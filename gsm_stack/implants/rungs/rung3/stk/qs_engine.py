"""M3 question-structure implant, ENGINE side (all-weights stack, prefill only).

install(Q, tok, side_path): wraps fastqwen.fwd_chunk. For every SUFFIX prefill chunk (prefix_kv given, T > 1 = the question
'Question: q\\nAnswer:' rows of Engine._prefill) it adds per-layer `links` (the engine's own residual-add mechanism):
  layer LRD : reads the layer input x (no change) -> compiled reader roles -> trained side map -> deltas
  layer L>LRD: x[tail] += delta[L]   at the 3 tail tokens found by a compiled token trigger ('\\n'-token, 'Answer', ':').
Decode steps (T == 1), the shared 8-shot prefix (prefix_kv None) and every other call are untouched. Not installed => the
code path is exactly the base engine.
What is weights: vocab lookup tables (digit / sep / alpha / stop / relation / sentence-end), the number DFA and the
last-sentence register (fixed), the trigger (token-id equality), the side map (trained: pool queries, U, V, c).
No host Python decides anything per row: all row-dependent quantities are tensor ops on the token ids / states.
"""
import math, torch, torch.nn.functional as Fn

STOP = {'a', 'an', 'the', 'of', 'per', 'each', 'more', 'less', 'fewer', 'extra', 'additional', 'other', 'and', 'in', 'for', 'to'}
REL = set('''each every per total totals altogether all together combined sum twice double doubled triple tripled half halves
third thirds quarter quarters fourth fifth times more less fewer than left remaining remain remains rest spent spend gave give
gives sold sell sells lost lose ate eat eats increase increased decrease decreased percent profit cost costs earn earns earned
save saves saved difference average split splits share shares shared both pair pairs dozen dozens week weeks weekly day days
daily hour hours hourly minute minutes month months monthly year years yearly another additional extra remaining first second
last only except instead before after already still how many much what if while round trip back return'''.split())


def tables(tok, V):
    T = {k: torch.zeros(V, dtype=torch.bool) for k in ('dig', 'sep', 'alpha', 'stop', 'rel', 'end', 'nl')}
    for i in range(len(tok)):
        s = tok.decode([i]); st = s.strip()
        if st.isdigit(): T['dig'][i] = True
        if st in (',', '.'): T['sep'][i] = True
        if st.isalpha():
            T['alpha'][i] = True; w = st.lower()
            if w in STOP: T['stop'][i] = True
            if w in REL: T['rel'][i] = True
        if st in ('%', '$'): T['rel'][i] = st == '%'
        if any(c in s for c in '.!?') and not st.replace('.', '').isdigit(): T['end'][i] = True
        if '\n' in s: T['nl'][i] = True
    T['ANS'] = tok.convert_tokens_to_ids('Answer'); T['COL'] = tok.convert_tokens_to_ids(':')
    T['QHEAD'] = len(tok('Question:', add_special_tokens=False).input_ids)
    return T


def reader(T, ids):
    """ids (N, S) right-padded suffix ids. -> M (N,5,S) bool role masks, tail (N,3) long, fire (N,) bool.
    Same semantics as off/qs.py roles() (tested by test_qs_engine.py)."""
    N, S = ids.shape; dv = ids.device
    g = lambda k: T[k].to(dv)[ids]
    dig, sep, alpha, stop, rel, end, nl = (g(k) for k in ('dig', 'sep', 'alpha', 'stop', 'rel', 'end', 'nl'))
    pat = torch.zeros(N, S, dtype=torch.bool, device=dv)
    pat[:, 1:-1] = nl[:, :-2] & (ids[:, 1:-1] == T['ANS']) & (ids[:, 2:] == T['COL'])       # pattern centred on 'Answer'
    fire = pat.any(1)
    t1 = torch.where(fire, pat.float().argmax(1), torch.zeros_like(fire, dtype=torch.long))   # first match
    tail = torch.stack([t1 - 1, t1, t1 + 1], 1).clamp(min=0)
    t0 = t1 - 1                                                     # first tail token; body = [QHEAD, t0)
    M = torch.zeros(N, 5, S, dtype=torch.bool, device=dv)
    innum = torch.zeros(N, dtype=torch.bool, device=dv); want = torch.zeros(N, dtype=torch.long, device=dv)
    last_end = torch.full((N,), T['QHEAD'] - 1, dtype=torch.long, device=dv)
    digp = torch.cat([dig[:, 1:], torch.zeros(N, 1, dtype=torch.bool, device=dv)], 1)
    for j in range(T['QHEAD'], S):
        act = fire & (j < t0)
        isn = dig[:, j] | (innum & sep[:, j] & digp[:, j])
        M[:, 4, j] = act
        M[:, 0, j] = act & isn
        o = act & ~isn
        noun = o & (want > 0) & alpha[:, j] & ~stop[:, j]
        M[:, 1, j] = noun
        want = torch.where(act & isn, torch.full_like(want, 4), torch.where(noun, torch.zeros_like(want), torch.where(o & (want > 0), want - 1, want)))
        innum = torch.where(act, isn, innum)
        M[:, 2, j] = o & rel[:, j]
        last_end = torch.where(o & end[:, j] & (j < t0 - 1), torch.full_like(last_end, j), last_end)
    pos = torch.arange(S, device=dv)[None]
    M[:, 3] = M[:, 4] & (pos > last_end[:, None])
    empty = ~M[:, :4].any(2)                                        # (N,4) role without tokens -> ALL
    M[:, :4] = torch.where(empty[:, :, None], M[:, 4:5].expand(-1, 4, -1), M[:, :4])
    return M, tail, fire


class Side:
    def __init__(s, sd, dev):
        f = lambda k: sd[k].to(dev).float() if k in sd else None
        s.c, s.qry, s.U, s.V = f('c'), f('qry'), f('U'), f('V')
        s.nw = s.c.shape[0]; s.D = s.c.shape[-1]

    def deltas(s, hq, M):
        B = hq.shape[0]; D = s.D
        d = s.c[:, None].expand(-1, B, -1, -1)
        if s.V is None: return d
        h = Fn.layer_norm(hq.float(), (D,))
        sc = torch.einsum('btd,rd->brt', h, s.qry) / math.sqrt(D)
        sc = sc.masked_fill(~M, float('-inf'))
        pooled = torch.einsum('brt,btd->brd', sc.softmax(-1), h)
        f = Fn.layer_norm(pooled, (D,)).reshape(B, 5 * D)
        return d + torch.einsum('br,lprd->lbpd', f @ s.U, s.V)


def install(Q, tok, side_path, LRD=6, scale=1.0):
    sd = torch.load(side_path, map_location='cpu')
    T = tables(tok, max(len(tok), 151936))
    orig = Q.fwd_chunk
    st = {}
    sides = {}

    def mk(L):
        def link(feat, x):
            if L == LRD:
                key = x.device
                if key not in sides: sides[key] = Side(sd, x.device)
                st['d'] = sides[key].deltas(x, st['M']) * scale
                return torch.zeros_like(x)
            N = x.shape[0]; bi = torch.arange(N, device=x.device)[:, None].expand(-1, 3)
            add = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
            add[bi, st['tail']] = st['d'][L - LRD - 1] * st['fire'][:, None, None]
            return add
        return link
    QL = {L: mk(L) for L in range(LRD, Q.NL)}

    def fwd_chunk(W, ids, pos0, cos_t, sin_t, prefix_kv=None, links=None, feat=None, **kw):
        if prefix_kv is None or ids.shape[1] < 4:
            return orig(W, ids, pos0, cos_t, sin_t, prefix_kv=prefix_kv, links=links, feat=feat, **kw)
        st['M'], st['tail'], st['fire'] = reader(T, ids)
        lk = dict(links) if links else {}
        for L, f in QL.items():
            if L in lk:
                g0 = lk[L]; lk[L] = (lambda g0, f: lambda fe, x: g0(fe, x) + f(fe, x))(g0, f)
            else: lk[L] = f
        try: return orig(W, ids, pos0, cos_t, sin_t, prefix_kv=prefix_kv, links=lk, feat=feat, **kw)
        finally: st.clear()
    Q.fwd_chunk = fwd_chunk
    return T
