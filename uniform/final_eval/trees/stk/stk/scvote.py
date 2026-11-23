"""task [19](a): the rung-2 voter applied OFFLINE to SC xN samples (N independent B-stack calls).

Per sample, from its tokens only:
  - reader registers with the serial stack's fixed tables (comp_serial: DFA TT/FLAG, symbol table): it reads the
    host-written tokens (B-stack writer tokens, wflag & 2, are skipped as the serial reader skips its own writes) until the
    answer closes; ok = valid (closed, or ends in a DONE state), bad = ~valid | sanity flag 1/2 (k_rm off, as rung 2),
    sym = up to 16 answer symbols (pad 12);
  - probe logit = the L18 verifier probe (verifier_probe_L18.pt) on the layer-18 input residual at the LAST token that
    moved the DFA into the '####' state before the close (the serial stack overwrites s_lg on each capture); -30 when none.
    The residual comes from a teacher-forced host forward (fastqwen.fwd_chunk with the shared prefix KV), so it is the host's
    residual on the sample text (bf16 numerics of a different kernel; B-stack forget masking after a fire is not replayed).
  - vote = comp_serial.SerialW._vote with NS = N, k_N = N, sanity filter on, probe weights sigmoid(lg) (vmode 1).
"""
import torch, torch.nn.functional as F
import fastqwen as Q
import san_engine as SE

KS, NSYM, PAD = 16, 13, 12


def tables(tok, V, cache):
    txt = [tok.decode([i]) for i in range(len(tok))] + [''] * max(0, V - len(tok))
    TT, FLAG, S = SE.build_tables(txt, cache)
    sym = torch.full((V,), -1, dtype=torch.long)
    for v, t in enumerate(txt):
        s_ = t.strip() if t.strip() == '-' else t
        if len(s_) == 1 and s_ in '0123456789': sym[v] = int(s_)
        elif s_ == '-' and t.lstrip() == '-': sym[v] = 10
        elif t == '.': sym[v] = 11
    done = torch.tensor([s[0] == 'X' for s in S]); numst = torch.tensor([s[0] in ('M', 'I', 'D', 'F') for s in S])
    return dict(TT=TT, FLAG=FLAG, sym=sym, done=done, numst=numst, W0=S.index(('W', 0)))


def read(T, ids, wfl):
    """-> ok, bad, sym (KS), capture position (index into ids) or -1."""
    d = 0; k = 0; syms = []; cap = -1; closed = False
    for j, t in enumerate(ids):
        if wfl[j] & 2: continue
        d1 = int(T['TT'][t, d])
        if d1 == T['W0'] and d != T['W0']: cap = j
        sy = int(T['sym'][t])
        if bool(T['numst'][d1]) and sy >= 0 and k < KS: syms.append(sy); k += 1
        if bool(T['done'][d1]) and not bool(T['done'][d]): closed = True
        d = d1
        if closed: break
    valid = closed or bool(T['done'][d])
    fl = int(T['FLAG'][d])
    bad = (not valid) or fl in (1, 2)
    return valid, bad, syms + [PAD] * (KS - len(syms)), cap


def vote(R_ok, R_bad, R_lg, R_sym):
    """comp_serial.SerialW._vote (k_N = NS, k_san on, k_vm = 1) on (B, NS) registers."""
    B, NS = R_ok.shape; dev = R_ok.device
    ok = R_ok
    sane = ok & ~R_bad
    anys = sane.any(1, keepdim=True)
    keep = ok & (sane | ~anys)
    Rh = F.one_hot(R_sym, NSYM).float()
    same = torch.einsum('bikc,bjkc->bij', Rh, Rh)
    eq = torch.relu(same - (KS - 1))
    w = torch.sigmoid(R_lg)
    kw = keep.float() * w
    score = torch.einsum('bij,bj->bi', eq, kw)
    dlt = score[:, :, None] - score[:, None, :]
    ar = torch.arange(NS, device=dev)
    earlier = (ar[:, None] < ar[None, :]).float()
    beat = torch.clamp(torch.relu(1e6 * dlt) + torch.relu(1 - 1e6 * dlt.abs()) * earlier, max=1.0)
    beat = torch.maximum(beat, (~keep).float()[:, None, :])
    beat = beat + torch.eye(NS, device=dev)[None]
    win = keep & (torch.relu(beat.sum(-1) - (NS - 1)) > 0.5)
    return win.any(1), win.float().argmax(1)


def val(sym_row):
    s = ''.join('0123456789-.'[x] for x in sym_row if x != PAD)
    return s or None


@torch.no_grad()
def run(eng, tok, C, items, qids, o, GS, rows, stk_dir, P):
    dev = eng.dev; W = eng.W
    T = tables(tok, W['E'].shape[0], f'{stk_dir}/dfa_tables.pt')
    pr = torch.load(f'{stk_dir}/verifier_probe_L18.pt')
    v_w = pr['w'].to(dev).float(); v_b = float(pr['b']); v_L = int(pr.get('layer', 18))
    pkv = Q.prefix_kv_of(W, eng.prefix_ids, eng.cos_t, eng.sin_t)
    Lp = int(eng.Lp)
    n = len(items); nok = 0; ncap = 0; nvalid = 0
    for j in range(n):
        toks = o['tokens'][j * GS:(j + 1) * GS]; wfs = o['wflag'][j * GS:(j + 1) * GS]
        rd = [read(T, t, w) for t, w in zip(toks, wfs)]
        q = list(qids[j]); Lq = len(q)
        seqs = [q + list(t) for t in toks]; L = max(map(len, seqs))
        ids = torch.full((GS, L), int(eng.eos), dtype=torch.long, device=dev)
        for s, sq in enumerate(seqs): ids[s, :len(sq)] = torch.tensor(sq, device=dev)
        _, hcap = Q.fwd_chunk(W, ids, Lp, eng.cos_t, eng.sin_t, prefix_kv=pkv, capture=v_L)
        lg = []
        for s, (valid, bad, sy, cap) in enumerate(rd):
            if cap >= 0:
                # residual of the step whose INPUT token is ids[Lq + cap]: position Lq + cap of the teacher-forced sequence
                lg.append(float(hcap[s, Lq + cap].float() @ v_w + v_b)); ncap += 1
            else:
                lg.append(-30.0)
            nvalid += int(valid)
        R_ok = torch.tensor([[r[0] for r in rd]], device=dev); R_bad = torch.tensor([[r[1] for r in rd]], device=dev)
        R_sym = torch.tensor([[r[2] for r in rd]], device=dev); R_lg = torch.tensor([lg], device=dev, dtype=torch.float32)
        has, wi = vote(R_ok, R_bad, R_lg, R_sym)
        ans = val(rd[int(wi)][2]) if bool(has) else None
        ok = int(C.correct(ans, items[j]['gold'])) if ans is not None else 0
        rows[j].update(scv_ok=ok, scv_last=ans, scv_lg=lg, scv_valid=[int(r[0]) for r in rd], scv_bad=[int(r[1]) for r in rd],
                       scv_sym=[val(r[2]) for r in rd])
        nok += ok
    P('scvote', f"n={n} acc_probe_vote={100 * nok / n:.2f} acc_majority={100 * sum(r['ok'] for r in rows) / n:.2f} "
                f"probe_captures={ncap}/{n * GS} valid={nvalid}/{n * GS}")
