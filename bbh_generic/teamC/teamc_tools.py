"""Team C generic tools (BBH held-out redo). Everything at inference is a compiled integer ReLU circuit (micromachines/hdl.py)
plus the solved linear reader / write directions of micromachines/machine.py; no Python logic runs per token.

  expr_gen : f(text) = exact value of an INTEGER arithmetic expression the host wrote, with + - * (also x, ×), unary minus and
             parentheses nested up to D=4 deep, any number of numbers (up to the K=72-token window), any integer sizes with all
             intermediate values below 10^12. Fires when the expression is followed by '=' (or ')='): writes ' ', [-], the
             digits, then an end write (the solved 'no more digits' veto by default). Silent on anything it cannot certify.
  calc_gen : the old calculator calc_s4 (exact a op b on signed decimals, also two-op chains), restricted to the cases expr_gen
             does not cover: the call contains a decimal point or a division. So the two never both fire.
  bool_gen : f(text) = truth value of a propositional expression over the literals True/False with not/and/or and
             parentheses (Python precedence), at most 16 words, nesting 3. Fires at '=' after the expression; writes ' True'/' False'.

Token classes are defined by RULES over the whole vocabulary (written below), not by lists mined from any benchmark:
  * expression punctuation: every vocabulary token whose text, spaces removed, is  ')'* [op] ['-'] '('* ['-']  with op in + - * ×
    (or the token ' x'), at most 4 brackets; space-only tokens. Digits are single tokens in Qwen3.
  * start guard (blacklist): the token right before the expression must not be something that could be PART of an operand or
    an operator we do not parse, else a prefix would be silently dropped ('3.5 + 2', '1,000 - 3', 'x - 2', '2^3 - 1', '8 / 2 + 1').
    The guard list = decimal/thousands separators, other operators (/ ^ % ** etc.), single letters (variables), unicode minus/dashes.
"""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', '..')); sys.path.insert(0, os.environ.get('WSM_REPO', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
import torch, torch.nn as nn
from micromachines.hdl import Net, Lin, lsum, compile_net, ReLUNet
import numpy as np
import micromachines.hdl as _hdl


def compile_net64(net, outs):
    """hdl.compile_net, but the weight arrays are float64 (expr_gen uses constants up to 1e14, not representable in float32)"""
    old = _hdl.np
    class _NP:                                            # numpy with float32 -> float64
        def __getattr__(self, k): return np.float64 if k == 'float32' else getattr(np, k)
    _hdl.np = _NP()
    try: return compile_net(net, outs)
    finally: _hdl.np = old
from micromachines import machine as MM
from micromachines.specs import SPECS
from micromachines.circuits import calc as _C
import calc_neg as _CN

_TOK = None
def tokenizer():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(os.environ.get('MODEL', 'Qwen/Qwen3-0.6B-Base'))
    return _TOK


def _single(tok, s):
    ids = tok(s, add_special_tokens=False)['input_ids']
    return len(ids) == 1


def vocab_texts(tok):
    return [tok.decode([i]) for i in range(len(tok))]


# ============================================================ token classes (rules) ============================================
D_MAX = 4
ETOK_RE = re.compile(r'^(\)*)([+\-*×]?)(-?)(\(*)(-?)$')


def expr_tokens(tok):
    """-> list of (text, parse) for every expression-punctuation token; parse = (n_close, op, pre_minus, n_open, post_minus)."""
    out = []
    for t in sorted(set(vocab_texts(tok))):
        if not t or '\n' in t or '\t' in t or not _single(tok, t): continue
        if t.strip() == '':
            if set(t) == {' '} and len(t) <= 4: out.append((t, (0, '', 0, 0, 0)))
            continue
        u = t.replace(' ', '')
        if u == 'x' and t == ' x': out.append((t, (0, '*', 0, 0, 0))); continue
        m = ETOK_RE.match(u)
        if not m: continue
        c, op, pre, o, post = m.groups()
        if len(c) > D_MAX or len(o) > D_MAX: continue
        if pre and not op: continue                      # a token '-(' is op '-' then '(' (regex puts '-' into op)
        out.append((t, (len(c), {'×': '*'}.get(op, op), int(bool(pre)), len(o), int(bool(post)))))
    return out


SEP_GUARD = {'.', ',', ' .', ' ,'}
# spelled-out arithmetic words: '5 plus 3 * 4 =' must not be answered with the value of '3 * 4' (judge round 1)
GUARD_WORDS = ['plus', 'minus', 'times', 'divided', 'by', 'over', 'multiplied', 'less', 'more', 'than', 'mod', 'modulo', 'per',
               'add', 'added', 'subtract', 'subtracted', 'multiply', 'divide']
GUARD_WORDS = GUARD_WORDS + [w.capitalize() for w in GUARD_WORDS]          # 'Plus 3 * 4 =' (judge expr round 2)
# An expression must be SEPARATED from what precedes it: either its first token starts with a space, or the token before
# it is one of these separators. Otherwise it is glued to a word or a bracket ('sqrt(4+5)', 'x2+1', ')x7+1') -> silent.
SEP_TOKENS = ['=', ' =', ':', ' :', '\n', '\n\n', ' \n', '.\n', ':\n', ',\n', '.\n\n', ':\n\n']
GUARD_CHARS = ['.', ',', '/', '^', '%', '**', '−', '–', '—', '_', '\\', '|', '&', '!', '~', '<', '>', '[', ']', '{', '}', '@', '#', "'", '`']


def guard_tokens(tok, etexts):
    """Tokens that may not stand right before an expression (they could belong to the first operand or be an unparsed operator):
    a single ASCII letter (a variable), or one of GUARD_CHARS, each with or without a leading space.
    The separators '.' and ',' (SEP_GUARD) only guard when the expression's first token is a digit right after them
    ('3.5', '1,000'); after a sentence ('is 5. 3 + 4 =', 'more, 5 + 3 =') they are ordinary punctuation."""
    cands = [c for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'] + GUARD_CHARS + ['..', '...'] + GUARD_WORDS
    out = set()
    for u in cands:
        for t in (u, ' ' + u):
            if t not in etexts and _single(tok, t): out.add(t)
    return sorted(out)


# ============================================================ expr_gen: stack 1 ================================================
K_EX, P_EX = 72, 13
BV = 10 ** 12                      # |every register| <= BV, else overflow -> silent
MG = 10 ** 14                      # gate constant (> any gated value); float64 keeps all of this exact (< 2^53)


CALL_CS, CALL_CE, CALL_OS, CALL_OE = '<tool_call>', '</tool_call>', '<tool_response>', '</tool_response>'


def expr_syms(tok, call=False):
    ET = expr_tokens(tok)
    G = guard_tokens(tok, {t for t, _ in ET})
    SP_ = [t for t in SEP_TOKENS if _single(tok, t)]
    syms = [str(d) for d in range(10)] + [f'E{i}' for i in range(len(ET))] + ['CE', 'CEC'] + [f'G{i}' for i in range(len(G))] + [f'P{i}' for i in range(len(SP_))]
    text = {**{str(d): str(d) for d in range(10)}, **{f'E{i}': t for i, (t, _) in enumerate(ET)},
            'CE': ('=', ' =', ')='), 'CEC': ')=', **{f'G{i}': t for i, t in enumerate(G)}, **{f'P{i}': t for i, t in enumerate(SP_)}}
    syms += ['ANY']; text['ANY'] = '<|endoftext|>'           # ANY: 1 on every real token, 0 before the text (reader set in attach_gen)
    if call:                                              # <tool_call>expr EXPR</tool_call><tool_response>VALUE</tool_response>
        syms += ['TC', 'NM', 'OSY']; text.update(CE=CALL_CE, TC=CALL_CS, NM='expr', OSY=CALL_OS)
    return ET, G, syms, text


def build_expr_stack1(ET, G, syms, K=K_EX, P=P_EX, D=D_MAX, call=False):
    NS = len(syms); SI = {s: i for i, s in enumerate(syms)}
    net = Net(K * NS); A_ = net.AND; relu = net.relu
    s = lambda o, x: net.inp(o * NS + SI[x]) if 0 <= o < K else Lin()
    clip = net.clip01
    def g(x, f, M=MG):                                    # x if f else 0   (f in {0,1}, |x| < M)
        return relu(x - M * (1 - f)) - relu(-x - M * (1 - f))
    def upd(reg, new, f, M=MG): return reg + g(new - reg, f, M)
    def clampv(x):                                        # clamp to [-BV, BV], and the overflow flag
        return relu(x + BV) - relu(x - BV) - BV, net.ge(x, BV + 1) + net.ge(-x, BV + 1)
    # ---- linear token features per offset ----
    F = {}
    for o in range(K):
        f = dict(dig=lsum(s(o, str(v)) for v in range(10)), close={k: Lin() for k in range(1, D + 1)}, opP=Lin(), opM=Lin(), opT=Lin(),
                 pre=Lin(), push={k: Lin() for k in range(1, D + 1)}, post=Lin(), et=Lin(), guard=lsum(s(o, f'G{i}') for i in range(len(G)) if G[i] not in SEP_GUARD),
                 sepg=lsum(s(o, f'G{i}') for i in range(len(G)) if G[i] in SEP_GUARD),
                 slead=lsum(s(o, f'E{i}') for i, (t, _) in enumerate(ET) if t.startswith(' ')),
                 sep=lsum(s(o, x) for x in syms if x.startswith('P')), any=s(o, 'ANY'))
        for i, (t, (c, op, pre, nop, post)) in enumerate(ET):
            x = s(o, f'E{i}'); f['et'] = f['et'] + x
            if c: f['close'][c] = f['close'][c] + x
            if op == '+': f['opP'] = f['opP'] + x
            if op == '-': f['opM'] = f['opM'] + x
            if op == '*': f['opT'] = f['opT'] + x
            if pre: f['pre'] = f['pre'] + x
            if nop: f['push'][nop] = f['push'][nop] + x
            if post: f['post'] = f['post'] + x
        f['ex'] = f['dig'] + f['et']
        F[o] = f
    F[0]['close'][1] = F[0]['close'][1] + s(0, 'CEC')     # ')=' closes one bracket before the '='
    # ---- span: offsets 1..t-1 are expression tokens, t is the start (guard-checked) ----
    sp = {o: relu(lsum(F[j]['ex'] for j in range(1, o + 1)) - (o - 1)) for o in range(1, K)}
    term = {t: A_(sp[t - 1], 1 - F[t]['ex']) for t in range(2, K)}
    if not call:
        start_ok = lsum(A_(term[t], 1 - F[t]['guard'] - A_(F[t]['sepg'], F[t - 1]['dig']), clip(F[t - 1]['slead'] + F[t]['sep'] + 1 - F[t]['any'])) for t in term)
        mine = None
    else:                                                 # the span starts right after '<tool_call>expr'
        start_ok = lsum(A_(term[t], s(t, 'NM'), s(t + 1, 'TC')) for t in term)
        mine = lsum(A_(s(t, 'NM'), s(t + 1, 'TC'), 1 - clip(lsum(s(u, 'TC') for u in range(1, t)))) for t in range(1, K - 1))
    # ---- the scan: a stack machine over the span, offset K-2 (oldest) .. 0 ('=' token: only its ')') ----
    INIT = dict(A=0, B=1, R=1, Rs=0, dep=0, exp=1, inn=0, err=0, hop=0, ovf=0, **{f'SA{j}': 0 for j in range(1, D + 1)}, **{f'SB{j}': 0 for j in range(1, D + 1)})
    BIGREG = ['A', 'B', 'R', 'Rs'] + [f'SA{j}' for j in range(1, D + 1)] + [f'SB{j}' for j in range(1, D + 1)]
    st = {k: Lin(c=v) for k, v in INIT.items()}
    for o in range(K - 2, -1, -1):
        f = F[o]
        if o < K - 2:                                     # restart if the token after this one (o+1) was outside the span
            keep = sp[o + 1]
            st = {k: g(v, keep, MG if k in BIGREG else 4 * K) + INIT[k] * (1 - keep) for k, v in st.items()}
        exp, dep = st['exp'], st['dep']
        err = [st['err']]
        # (1) closes: pop k levels at once
        anyc = lsum(f['close'].values())
        err.append(A_(anyc, exp))
        fk = {k: A_(f['close'][k], 1 - exp) for k in range(1, D + 1)}
        err += [relu(f['close'][k] - net.ge(dep, k)) for k in range(1, D + 1)]
        R0, A0, B0, SA, SB = st['R'], st['A'], st['B'], [None] + [st[f'SA{j}'] for j in range(1, D + 1)], [None] + [st[f'SB{j}'] for j in range(1, D + 1)]
        newR = R0 + lsum(g(lsum([A0] + [SA[j] for j in range(1, k)]), fk[k]) for k in range(1, D + 1))
        newA = A0 + lsum(g(SA[k] - A0, fk[k]) for k in range(1, D + 1))
        newB = B0 + lsum(g(SB[k] - B0, fk[k]) for k in range(1, D + 1))
        nSA = {j: SA[j] + lsum(g((SA[j + k] if j + k <= D else 0) - SA[j], fk[k]) for k in range(1, D + 1)) for j in range(1, D + 1)}
        nSB = {j: SB[j] + lsum(g((SB[j + k] if j + k <= D else 0) - SB[j], fk[k]) for k in range(1, D + 1)) for j in range(1, D + 1)}
        dep = dep - lsum(k * fk[k] for k in range(1, D + 1))
        newR, ov0 = clampv(newR)
        R0, A0, B0, SA, SB = newR, newA, newB, [None] + [nSA[j] for j in range(1, D + 1)], [None] + [nSB[j] for j in range(1, D + 1)]
        # (2) operator: binary after an operand, unary +/- where an operand is expected
        fP, fM, fT = A_(f['opP'], 1 - exp), A_(f['opM'], 1 - exp), A_(f['opT'], 1 - exp)
        uM = A_(f['opM'], exp); err.append(A_(f['opT'], exp))
        A1 = A0 + g(R0, fP + fM)
        R1 = R0 + g(B0 - R0, fP) + g(-B0 - R0, fM) - 2 * g(R0, uM)
        exp1 = exp + fP + fM + fT
        hop = st['hop'] + fP + fM + fT
        A1, ov1 = clampv(A1)
        # (3) unary minus before the opening brackets
        u = A_(f['pre'], exp1); err.append(A_(f['pre'], 1 - exp1))
        R2 = R1 - 2 * g(R1, u)
        # (4) push k levels: the new levels start a fresh sum whose base coefficient is the current running product R
        anyp = lsum(f['push'].values())
        err.append(A_(anyp, 1 - exp1))
        pk = {k: A_(f['push'][k], exp1) for k in range(1, D + 1)}
        err += [A_(f['push'][k], net.ge(dep, D - k + 1)) for k in range(1, D + 1)]
        tSA = {k: {j: (SA[j - k] if j > k else (A1 if j == k else Lin())) for j in range(1, D + 1)} for k in range(1, D + 1)}
        tSB = {k: {j: (SB[j - k] if j > k else (B0 if j == k else R2)) for j in range(1, D + 1)} for k in range(1, D + 1)}
        A2 = A1 + lsum(g(-A1, pk[k]) for k in range(1, D + 1))
        B2 = B0 + lsum(g(R2 - B0, pk[k]) for k in range(1, D + 1))
        SA2 = {j: SA[j] + lsum(g(tSA[k][j] - SA[j], pk[k]) for k in range(1, D + 1)) for j in range(1, D + 1)}
        SB2 = {j: SB[j] + lsum(g(tSB[k][j] - SB[j], pk[k]) for k in range(1, D + 1)) for j in range(1, D + 1)}
        dep = dep + lsum(k * pk[k] for k in range(1, D + 1))
        # (5) unary minus after the opening brackets (applies to the first term inside)
        u2 = A_(f['post'], exp1); err.append(A_(f['post'], 1 - exp1))
        R3 = R2 - 2 * g(R2, u2)
        # (6) a digit: R = Rs * (number so far), updated digit by digit (only digit x integer products, all exact)
        dg, inn = f['dig'], st['inn']
        err.append(relu(dg - exp1 - inn))                # a digit right after a finished operand
        first = A_(dg, 1 - inn)
        Rs = st['Rs'] + g(R3 - st['Rs'], first)
        mult = lsum(v * g(Rs, s(o, str(v))) for v in range(1, 10))
        R4 = R3 + g(g(10 * R3, inn) + mult - R3, dg)
        R4, ov2 = clampv(R4)
        exp4 = exp1 - A_(exp1, dg)
        st = dict(A=A2, B=B2, R=R4, Rs=Rs, dep=dep, exp=exp4, inn=dg, err=clip(lsum(err)), hop=clip(hop), ovf=clip(st['ovf'] + ov0 + ov1 + ov2),
                  **{f'SA{j}': SA2[j] for j in range(1, D + 1)}, **{f'SB{j}': SB2[j] for j in range(1, D + 1)})
        if o == 0: break
    V = st['A'] + st['R']
    V, ov3 = clampv(V)
    valid = A_(s(0, 'CE'), clip(start_ok), 1 - st['err'], 1 - clip(st['ovf'] + ov3), st['hop'], 1 - st['exp'], net.eq(st['dep'], 0), sp[1])
    neg = net.ge(-V, 1); rem = relu(V) + relu(-V)
    Rd = [None] * P
    for k in range(P - 1, -1, -1):
        d = lsum(relu(net.ge(rem, j * 10 ** k)) for j in range(1, 10)); Rd[k] = d; rem = rem - 10 ** k * d   # relu(): one clean 0/1 wire (keeps 10^k * wire exact in float64)
    nz = [net.ge(r, 1) for r in Rd]
    Ld = {m: relu(nz[m - 1] - lsum(nz[m:])) for m in range(2, P + 1)}; Ld[1] = 1 - lsum(Ld.values())
    S = [valid, A_(valid, neg)] + Rd + [Ld[m] for m in range(1, P + 1)] + [valid if mine is None else clip(mine)]
    return net, S, dict(valid=[valid], V=[V], err=[st['err']], dep=[st['dep']], exp=[st['exp']], hop=[st['hop']], start=[start_ok])


def build_expr_stack2(syms, K=K_EX, P=P_EX, call=False):
    """CE at offset k; the machine's lead (' ', or <tool_response> in call format) at k-1; its '-' at k-2 if negative;
    m digits written since (offsets 0..m-1)."""
    NS = len(syms); SI = {s: i for i, s in enumerate(syms)}
    net = Net(K * NS + 3 + 2 * P); A_ = net.AND; b = K * NS
    s = lambda o, x: net.inp(o * NS + SI[x]) if 0 <= o < K else Lin()
    sp_sym, mn_sym = ('OSY' if call else SPACE_SYM[0]), MINUS_SYM[0]
    mine = net.inp(b + 2 + 2 * P)
    Dg = lambda o: lsum(s(o, str(v)) for v in range(10))
    valid, neg = net.inp(b), net.inp(b + 1)
    R = [net.inp(b + 2 + i) for i in range(P)]; Ld = {n: net.inp(b + 2 + P + n - 1) for n in range(1, P + 1)}
    digit_parts, wd, fin = [], [], []
    for k in range(1, P + 4):
        for sg in (0, 1):
            m = k - 1 - sg
            if m < 0: continue
            parts = [s(k, 'CE'), s(k - 1, sp_sym)] + ([s(k - 2, mn_sym)] if sg else []) + [Dg(o) for o in range(m)] + [neg if sg else 1 - neg]
            e = A_(*parts)
            for n in range(1, P + 1):
                gg = A_(e, Ld[n])
                if m < n: digit_parts.append(net.sel(R[n - 1 - m], gg)); wd.append(gg)
                if m == n: fin.append(gg)
    sgn = A_(s(1, 'CE'), s(0, sp_sym), neg)
    cur, wdig, fin = lsum(digit_parts), lsum(wd), lsum(fin)
    out = {str(d): A_(valid, wdig, net.eq(cur, d)) for d in range(10)}
    out['MN'] = A_(valid, sgn); out['OE'] = A_(valid, fin); out['OS'] = A_(s(0, 'CE'), mine)
    if call: out['OE'] = out['OE'] + A_(s(1, 'CE'), s(0, sp_sym), mine, 1 - valid)      # a call it cannot answer: empty response
    return net, [out[t] for t in EX_OUT]


EX_OUT = [str(d) for d in range(10)] + ['MN', 'OS', 'OE']
SPACE_SYM, MINUS_SYM = [None], [None]


def make_expr_spec(oe_text='.\n', call=False):
    tok = tokenizer()
    ET, G, syms, text = expr_syms(tok, call)
    SPACE_SYM[0] = [f'E{i}' for i, (t, _) in enumerate(ET) if t == ' '][0]
    MINUS_SYM[0] = [f'E{i}' for i, (t, _) in enumerate(ET) if t == '-'][0]
    def build():
        n1, S, _ = build_expr_stack1(ET, G, syms, call=call); n2, O = build_expr_stack2(syms, call=call)
        return compile_net64(n1, S), compile_net(n2, O)
    return dict(build=build, syms=syms, out=EX_OUT, key='CE', K=K_EX, W=P_EX + 4, f64=True, ET=ET, G=G,
                text={**text, 'MN': '-', 'OS': CALL_OS if call else ' ', 'OE': CALL_OE if call else oe_text})


def expr_ref(e):
    """Python reference of what expr_gen should compute for the text before '=' ('' = must be silent). Used only by tests."""
    try:
        v = eval(e.replace('×', '*').replace(' x ', ' * '), {'__builtins__': {}})
    except Exception:
        return ''
    return str(v) if isinstance(v, int) and abs(v) <= BV else ''


# ============================================================ calc_gen ==========================================================
def build_calc_gen():
    net, S, _ = _CN.build_stack1(spaced=True)
    w = _C.Win(net)
    rng = range(1, _C.K)
    calctok = {o: w.dig(o) + net.AND(w.dot(o), w.dig(o - 1), w.dig(o + 1)) + w.isop(o) + w.sp(o) for o in rng}   # a '.' counts only as a decimal point (digits on both sides)
    span = {o: net.relu(lsum(calctok[j] for j in range(1, o + 1)) - (o - 1)) for o in rng}
    hasdot = net.clip01(lsum(net.AND(span[o], w.dot(o)) for o in rng))
    hasdiv = net.clip01(lsum(net.AND(span[o], w.s(o, '/') + w.s(o, 'N/')) for o in rng))
    S = [net.AND(S[0], net.clip01(hasdot + hasdiv))] + S[1:]
    n2, O, _ = _C.build_stack2(dotveto=True)
    return compile_net(net, S), compile_net(n2, O)


def build_calc_call():
    """calc_s4's circuits for <tool_call>calc A op B</tool_call>: valid only if the call's tokens start right after the name
    token 'calc' (symbol CSN); <tool_response> is written for every call to calc (delimited by the envelope tokens, any
    content), then the digits, then </tool_response> (empty response = the call is malformed or out of range)."""
    net, S, _ = _CN.build_stack1(spaced=True)
    w = _C.Win(net); rng = range(1, _C.K)
    calctok = {o: w.dig(o) + net.AND(w.dot(o), w.dig(o - 1), w.dig(o + 1)) + w.isop(o) + w.sp(o) for o in rng}   # a '.' counts only as a decimal point (digits on both sides)
    span = {o: net.relu(lsum(calctok[j] for j in range(1, o + 1)) - (o - 1)) for o in rng}
    named = net.clip01(lsum(net.AND(span[t - 1], 1 - calctok[t], w.s(t, 'CSN')) for t in range(2, _C.K)))
    S = [net.AND(S[0], named)] + S[1:]
    n2, O, _ = _C.build_stack2(dotveto=False)
    w2 = _C.Win(n2); valid2 = n2.inp(_C.K * _C.NS)
    # Is this the end of a call to calc? The call is delimited by the envelope, whatever tokens are inside: the most
    # recent '</tool_call>' is the current one and the name token 'calc' (CSN) sits right after '<tool_call>'. calc's
    # symbol table has no free slot for '<tool_call>', so the input symbol 'OE' (never read by calc's circuits; its
    # FIRST text is the written '</tool_response>') also lists '<tool_call>' as an alternative token.
    def mine_from(a):
        return n2.clip01(lsum(n2.AND(w2.s(t, 'CSN'), w2.s(t + 1, 'OE'), 1 - n2.clip01(lsum(w2.s(u, 'CE') for u in range(a, t))))
                              for t in range(a, _C.K - 1)))
    iOS, iOE = _C.OUT.index('OS'), _C.OUT.index('OE')
    O = list(O)
    O[iOS] = n2.AND(w2.s(0, 'CE'), mine_from(1))
    O[iOE] = O[iOE] + n2.AND(w2.s(0, 'OS'), w2.s(1, 'CE'), mine_from(2), 1 - valid2)
    return compile_net(net, S), compile_net(n2, O)


# ============================================================ bool_gen ==========================================================
# A scan (pushdown) evaluator, like expr_gen, over every vocabulary token of the form  [' '] ')'* '('* [WORD]  with
# WORD in True/False/true/false/not/and/or (so merged tokens '((' '))' '(not' '(True' ... are read, not skipped).
# Registers are 0/1: A = OR of finished and-terms at this level, R = AND of the current term (times the level's base B),
# par = parity of pending 'not's, gp = parity of the 'not's in front of the current bracket; frames (A, B, gp) per level.
K_BO, D_BO = 48, 4
BOOL_RE = re.compile(r'^ ?(\)*)(\(*)(True|False|true|false|not|and|or)?$')
BOOL_GUARD_WORDS = ['Not', 'And', 'Or', 'NOT', 'AND', 'OR', 'xor', 'nor', 'nand', 'XOR', 'if', 'else', 'elif', 'then', 'is', 'in']
BOOL_GUARD_SYMS = ['!', '~', '¬', '&&', '||', '&', '|', '^', '==', '!=', '∧', '∨', '=>', '->', '<-', '-', '+', '*', '<', '>', '<=', '>=', '**', '%', '/']   # other operators: the run may be an operand of them


def bool_tokens(tok):
    out = []
    for t in sorted(set(vocab_texts(tok))):
        if not t or not _single(tok, t): continue
        if t.strip() == '':
            if set(t) == {' '} and len(t) <= 4: out.append((t, (0, 0, '')))
            continue
        m = BOOL_RE.match(t)
        if not m or not t.strip(): continue
        c, o, w = m.groups()
        if len(c) > D_BO or len(o) > D_BO: continue
        out.append((t, (len(c), len(o), (w or '').lower())))
    return out


def bool_guard_tokens(tok, own):
    out = set()
    for u in BOOL_GUARD_WORDS + BOOL_GUARD_SYMS:
        for t in (u, ' ' + u):
            if t not in own and _single(tok, t): out.add(t)
    return sorted(out)


def build_bool_stack1(BT, GT, syms, K=K_BO, D=D_BO, call=False):
    NS = len(syms); SI = {s: i for i, s in enumerate(syms)}
    net = Net(K * NS); A_ = net.AND; relu = net.relu; clip = net.clip01
    s = lambda o, x: net.inp(o * NS + SI[x]) if 0 <= o < K else Lin()
    def sel(f, x, y): return A_(f, x) + A_(1 - f, y)            # f ? x : y   (all 0/1)
    F = {}
    for o in range(K):
        f = dict(close={k: Lin() for k in range(1, D + 1)}, open={k: Lin() for k in range(1, D + 1)}, T=Lin(), Fa=Lin(), N=Lin(), An=Lin(), Or=Lin(), bt=Lin(),
                 guard=lsum(s(o, f'G{i}') for i in range(len(GT))), sep=lsum(s(o, x) for x in syms if x.startswith('P')),
                 slead=lsum(s(o, f'W{i}') for i, (t, _) in enumerate(BT) if t.startswith(' ')), any=s(o, 'ANY'))
        for i, (t, (c, op, w)) in enumerate(BT):
            x = s(o, f'W{i}'); f['bt'] = f['bt'] + x
            for k in range(1, c + 1): f['close'][k] = f['close'][k] + x          # close[k]: the token has >= k ')'
            for k in range(1, op + 1): f['open'][k] = f['open'][k] + x
            key = {'true': 'T', 'false': 'Fa', 'not': 'N', 'and': 'An', 'or': 'Or'}.get(w)
            if key: f[key] = f[key] + x
        F[o] = f
    F[0]['close'][1] = F[0]['close'][1] + s(0, 'CEC')
    sp = {o: relu(lsum(F[j]['bt'] for j in range(1, o + 1)) - (o - 1)) for o in range(1, K)}
    term = {t: A_(sp[t - 1], 1 - F[t]['bt']) for t in range(2, K)}
    if not call: start_ok = lsum(A_(term[t], 1 - F[t]['guard'], clip(F[t - 1]['slead'] + F[t]['sep'] + 1 - F[t]['any'])) for t in term)
    else:
        start_ok = lsum(A_(term[t], s(t, 'NM'), s(t + 1, 'TC')) for t in term)
        mine = clip(lsum(A_(s(t, 'NM'), s(t + 1, 'TC'), 1 - clip(lsum(s(u, 'TC') for u in range(1, t)))) for t in range(1, K - 1)))
    INIT = dict(A=0, R=1, B=1, par=0, gp=0, exp=1, dep=0, err=0, hop=0, **{f'{x}{j}': 0 for x in ('SA', 'SB', 'SG') for j in range(1, D + 1)})
    st = {k: Lin(c=v) for k, v in INIT.items()}
    for o in range(K - 2, -1, -1):
        f = F[o]
        if o < K - 2:
            keep = sp[o + 1]
            st = {k: (sel(keep, v, INIT[k]) if k != 'dep' else net.sel(v, keep, M=D)) for k, v in st.items()}
        st = dict(st); err = [st['err']]
        for k in range(1, D + 1):                                           # k-th ')' of this token
            c = f['close'][k]
            err += [A_(c, st['exp']), A_(c, 1 - net.ge(st['dep'], 1))]
            fc = A_(c, 1 - st['exp'], net.ge(st['dep'], 1))
            G = clip(st['A'] + st['R'])
            Rp = sel(st['gp'], st['B'] - G, G)
            st['R'] = sel(fc, Rp, st['R']); st['A'] = sel(fc, st['SA1'], st['A']); st['B'] = sel(fc, st['SB1'], st['B']); st['gp'] = sel(fc, st['SG1'], st['gp'])
            for x in ('SA', 'SB', 'SG'):
                for j in range(1, D + 1): st[f'{x}{j}'] = sel(fc, st[f'{x}{j + 1}'] if j < D else Lin(), st[f'{x}{j}'])
            st['dep'] = st['dep'] - fc
        for k in range(1, D + 1):                                           # k-th '(' of this token
            c = f['open'][k]
            err += [A_(c, 1 - st['exp']), A_(c, net.ge(st['dep'], D))]
            fo = A_(c, st['exp'], 1 - net.ge(st['dep'], D))
            for x, cur in (('SA', 'A'), ('SB', 'B'), ('SG', 'gp')):
                for j in range(D, 1, -1): st[f'{x}{j}'] = sel(fo, st[f'{x}{j - 1}'], st[f'{x}{j}'])
                st[f'{x}1'] = sel(fo, st[cur], st[f'{x}1'])
            st['A'] = sel(fo, Lin(), st['A']); st['B'] = sel(fo, st['R'], st['B']); st['gp'] = sel(fo, st['par'], st['gp']); st['par'] = sel(fo, Lin(), st['par'])
            st['dep'] = st['dep'] + fo
        exp = st['exp']; lit = f['T'] + f['Fa']
        err += [A_(lit + f['N'], 1 - exp), A_(f['An'] + f['Or'], exp)]
        fl, fn, fa, fr = A_(lit, exp), A_(f['N'], exp), A_(f['An'], 1 - exp), A_(f['Or'], 1 - exp)
        val = A_(f['T'], 1 - st['par']) + A_(f['Fa'], st['par'])                 # literal XOR pending not-parity
        Rl = A_(st['R'], val)
        nR = sel(fl, Rl, st['R']); nR = sel(fr, st['B'], nR)
        nA = sel(fr, clip(st['A'] + st['R']), st['A'])
        npar = sel(fn, 1 - st['par'], sel(fl, Lin(), st['par']))
        st.update(A=nA, R=nR, par=npar, exp=exp - fl + fa + fr, hop=clip(st['hop'] + fa + fr + fn), err=clip(lsum(err)))
        if o == 0: break
    value = clip(st['A'] + st['R'])
    valid = A_(s(0, 'CE'), clip(start_ok), 1 - st['err'], st['hop'], 1 - st['exp'], net.eq(st['dep'], 0), sp[1])
    return net, [valid, A_(valid, value), mine if call else valid]


def build_bool_stack2(syms, K, dot, call=False):
    NS = len(syms); SI = {s: i for i, s in enumerate(syms)}
    net = Net(K * NS + 3); s = lambda o, x: net.inp(o * NS + SI[x]); A_ = net.AND
    valid, value, mine = net.inp(K * NS), net.inp(K * NS + 1), net.inp(K * NS + 2)
    if not call:
        ce = s(0, 'CE')
        outs = [A_(ce, value), A_(ce, valid, 1 - value)]
        if dot: outs.append(A_(s(1, 'CE'), s(0, 'VT') + s(0, 'VF'), valid))
        return net, outs
    at = A_(s(0, 'OSY'), s(1, 'CE'))                                  # right after the machine's <tool_response>
    return net, [A_(at, value), A_(at, valid, 1 - value), A_(s(0, 'CE'), mine),
                 A_(s(0, 'VT') + s(0, 'VF'), s(1, 'OSY'), s(2, 'CE'), valid) + A_(at, mine, 1 - valid)]


def make_bool_spec(dot=False, call=False):
    tok = tokenizer()
    BT = bool_tokens(tok)
    own = {t for t, _ in BT} | {'=', ' =', ')='}
    GT = bool_guard_tokens(tok, own)
    SP_ = [t for t in SEP_TOKENS if _single(tok, t)]
    syms = [f'W{i}' for i in range(len(BT))] + ['CE', 'CEC', 'VT', 'VF'] + [f'G{i}' for i in range(len(GT))] + [f'P{i}' for i in range(len(SP_))] + ['ANY'] + (['TC', 'NM', 'OSY'] if call else [])
    def build():
        n1, S = build_bool_stack1(BT, GT, syms, call=call); n2, O = build_bool_stack2(syms, K_BO, dot, call)
        return compile_net(n1, S), compile_net(n2, O)
    text = {**{f'W{i}': t for i, (t, _) in enumerate(BT)}, 'CE': ('=', ' =', ')='), 'CEC': ')=', 'VT': ' True', 'VF': ' False', 'VD': '.',
            **{f'G{i}': t for i, t in enumerate(GT)}, **{f'P{i}': t for i, t in enumerate(SP_)}, 'ANY': '<|endoftext|>'}
    if call:      # <tool_call>bool EXPR</tool_call><tool_response>True|False</tool_response>
        text.update(CE=CALL_CE, TC=CALL_CS, NM='bool', OSY=CALL_OS, VT='True', VF='False', OS=CALL_OS, OE=CALL_OE)
        return dict(build=build, syms=syms, out=['VT', 'VF', 'OS', 'OE'], key='CE', K=K_BO, W=3, text=text, BT=BT, GT=GT)
    return dict(build=build, syms=syms, out=['VT', 'VF'] + (['VD'] if dot else []), key='CE', K=K_BO, W=2, text=text, BT=BT, GT=GT)


def bool_ref(e):
    try: v = eval(e.replace('true', 'True').replace('false', 'False'), {'__builtins__': {}})
    except Exception: return ''
    return str(v) if isinstance(v, bool) else ''


# ============================================================ attach ===========================================================
@torch.no_grad()
def prep_call_tokens(model, tok):
    """The four envelope tokens share ONE untrained embedding row in the base checkpoint, so no reader can tell them apart.
    The shared fine-tune (teamA/ft.py) first gives them distinct random rows; this is the same initialization (same seed,
    same order), used for testing the call-format tools before/without fine-tuning."""
    E = model.get_input_embeddings().weight
    g = torch.Generator().manual_seed(1234)
    for t in (CALL_CS, CALL_CE, CALL_OS, CALL_OE):
        v = torch.randn(E.shape[1], generator=g)
        E[tok.convert_tokens_to_ids(t)] = (v / v.norm() * E.norm(dim=1).median().cpu()).to(E.device, E.dtype)

class F64(nn.Module):
    """run a compiled ReLU stack in float64 (expr_gen's registers go up to 1e14), return float32"""
    def __init__(self, net):
        super().__init__(); self.net = net.double()
    def forward(self, x): return self.net(x.double()).float()


def register(oe='dotnl', bool_dot=1):
    SPECS['expr_gen'] = make_expr_spec('.\n' if oe in ('veto', 'dotnl') else oe)
    SPECS['calc_gen'] = dict(SPECS['calc_s4'], build=build_calc_gen)
    SPECS['bool_gen'] = make_bool_spec(bool(bool_dot))
    # call-format versions (shared envelope <tool_call>NAME ARGS</tool_call><tool_response>RESULT</tool_response>)
    SPECS['expr_call'] = make_expr_spec(call=True)
    SPECS['bool_call'] = make_bool_spec(call=True)
    SPECS['calc_call'] = dict(SPECS['calc_s4'], build=build_calc_call,
                              text={**SPECS['calc_s4']['text'], 'CS': ' ', 'SP': ' ', 'CSN': 'calc', 'CE': CALL_CE, 'OS': CALL_OS, 'OE': (CALL_OE, CALL_CS)})


def attach_gen(model, tok, machines, veto=None, oe='dotnl', bool_dot=1):
    register(oe, bool_dot)
    A = MM.Attached(model, tok, machines=tuple(machines))
    for n in machines:
        if SPECS[n].get('f64'): A.nets[n].stack1 = F64(A.nets[n].stack1).to(model.device)
        if 'ANY' in SPECS[n]['syms']:                     # the 'any token' flag: weight 0, bias 1 (exactly 1 on every vocabulary row,
            i = SPECS[n]['syms'].index('ANY')              # 0 in the zero padding before the first token = start of text)
            A.nets[n].reader_w[i] = 0; A.nets[n].reader_b[i] = 1
    v = torch.load(veto)['v'] if veto else None
    if 'calc_gen' in A.nets and v is not None: A.set_veto('calc_gen', v, zero_os=True)
    if 'expr_gen' in A.nets and oe == 'veto' and v is not None: A.set_veto('expr_gen', v, zero_os=False)
    return A
