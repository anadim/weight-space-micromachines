"""chain2: a compiled micromachine (hdl ReLU circuit, forward-only) for exact + / - chains, beside calc_s4.

Generalizes MM13 (circuits/sumexpr.py, 'sum'): signed sums of numbers written in GSM style, e.g.
    'The total is $585.80 + $565 + $599.20 + $567 ='       'So 1,000 - 300 - 200 + 2,000 ='      'is -5 + 3 + 4 ='
  numbers   up to NL integer digits, optional comma groups of 3 (1,234,567), up to F decimals, optional '$' before
  operators + and - only (binary), unary minus on the first number;  NO brackets, no x / (a mixed or bracketed
            expression is silent: fire rule agreed with the combination agent)
  fires     only where calc_s4 is silent: >= 4 operands, or a comma-grouped number, or a leading unary minus with >= 3
            operands; and only when the span before '=' starts at a 'safe' token (':' / newline / '=' / ' is' / ' has' ...)
Writes, after the model's own lead token (' ' or ' $', or ' -' when the value is negative): ['-'] digits ['.' digits]
(exact; trailing fraction zeros dropped), then the solved end-of-number veto. Silent (all-zero output) everywhere else.

Two stacks + one hard attention hop exactly as every micromachine (machine.py): stack 1 at the '=' token computes the
state [valid, neg, |value| digit register, #integer digits one-hot, #fraction digits one-hot]; the hop copies it to the
writing positions; stack 2 reads the window (what has been written since '=') and the state, and emits one output.
"""
from micromachines.hdl import Net, Lin, lsum, compile_net

NL = 9                       # max integer digits of an operand
F = 4                        # max fraction digits of an operand (fixed point: register index = place + F)
PT = NL + F + 3              # digits of the |result| register (up to ~25 operands)
PI = PT - F                  # integer digits of the result
K = 96                       # window (tokens back from '=')
MRUN = NL + (NL - 1) // 3    # max tokens of an integer part (digits + commas)

SAFE_TEXT = (':', '\n', '.\n', ':\n', ',\n', ' \n', ' is', ' are', ' was', ' were', ' be', ' equals', ' gives', ' get', ' gets',
             ' got', ' has', ' have', ' had', ' makes', ' total', ' of', ' so', ' spends', ' spent', ' spend', ' caught',
             ' sold', ' bought', ' buys', ' needs', ' need', ' for', ' becomes', ' pays', ' paid', ' used', ' uses', ' earns',
             ' earned', ' costs', ' cost', ' sum', ' Total', ' together', ' altogether', ' Sum')
SAFES = [f'S{i}' for i in range(len(SAFE_TEXT))]
SYMS = list('0123456789') + ['PL', 'MI', 'MN', 'SP', 'DL', 'DOT', 'CM', 'CE'] + SAFES
NS = len(SYMS); SI = {s: i for i, s in enumerate(SYMS)}
OUT = list('0123456789') + ['DOT', 'MN', 'OS', 'OE']
N_STATE = 2 + PT + PI + (F + 1)
TEXT = {**{c: c for c in '0123456789'}, 'PL': (' +', '+'), 'MI': ' -', 'MN': '-', 'SP': ' ', 'DL': (' $', '$'), 'DOT': '.',
        'CM': ',', 'CE': ('=', ' ='), **dict(zip(SAFES, SAFE_TEXT)), 'OS': ' ', 'OE': ' '}


class Win:
    def __init__(self, net): self.net = net
    def s(self, o, sym): return self.net.inp(o * NS + SI[sym]) if 0 <= o < K else Lin()
    def D(self, o): return lsum(self.s(o, str(v)) for v in range(10))
    def val(self, o): return lsum(self.s(o, str(v)) * v for v in range(1, 10))
    def safe(self, o): return lsum(self.s(o, c) for c in SAFES)


def build_stack1():
    net = Net(K * NS); w = Win(net); A = net.AND
    xor = lambda x, y: x + y - 2 * A(x, y)
    # context-dependent classes: a comma / point is part of a number only between digits (comma: digit on both sides,
    # point: a digit right after it); otherwise it ends the expression
    CMf = {o: A(w.s(o, 'CM'), w.D(o + 1), w.D(o - 1)) if 1 <= o < K - 1 else Lin() for o in range(K + 2)}
    DOTf = {o: A(w.s(o, 'DOT'), w.D(o - 1)) if 1 <= o < K else Lin() for o in range(K + 2)}
    cm = lambda o: CMf.get(o, Lin()); dt = lambda o: DOTf.get(o, Lin())
    numtok = lambda o: w.D(o) + cm(o) + dt(o)
    cls = lambda o: numtok(o) + w.s(o, 'PL') + w.s(o, 'MI') + w.s(o, 'MN') + w.s(o, 'SP') + w.s(o, 'DL')
    opn = lambda o: w.s(o, 'SP') + w.s(o, 'DL')                     # may stand right before a number
    bop = lambda o: w.s(o, 'PL') + w.s(o, 'MI')
    # ---- span: offsets 1..t-1 expression tokens, offset t the start token ----
    sp = {o: net.relu(lsum(cls(j) for j in range(1, o + 1)) - (o - 1)) for o in range(1, K)}
    term = {t: A(sp[t - 1], 1 - cls(t)) for t in range(2, K)}
    termg = lambda t: term.get(t, Lin())
    has_term = lsum(term.values())
    start_ok = lsum(A(term[t], w.safe(t) + w.s(t, 'CE') + w.s(t, 'CM')) for t in term)      # (a non-number comma)
    end_ok = w.D(1)
    # ---- numbers: start s = leftmost token (a digit) ----
    st = {s: A(w.D(s), 1 - numtok(s + 1), sp[s]) for s in range(1, K - 1)}
    L3 = {s: A(st[s], opn(s + 1), bop(s + 2), w.D(s + 3)) for s in st}                     # 'N + $M', 'N - M'
    L3b = {s: A(st[s], bop(s + 1) + w.s(s + 1, 'MN'), w.D(s + 2)) for s in st}             # 'N+M', 'N-M'
    L1 = {s: A(st[s], termg(s + 1)) for s in st}                                            # first number, nothing before
    L2 = {s: A(st[s], opn(s + 1), termg(s + 2)) for s in st}                                # ' is $N'
    L2m = {s: A(st[s], w.s(s + 1, 'MI') + w.s(s + 1, 'MN'), termg(s + 2)) for s in st}      # ' is -N'
    first = {s: L1[s] + L2[s] + L2m[s] for s in st}
    pat_ok = {s: first[s] + L3[s] + L3b[s] for s in st}
    bad_pat = lsum(st[s] - pat_ok[s] for s in st)
    nfirst = lsum(first.values())
    bad_first = net.relu(nfirst - 1) + net.relu(1 - nfirst)                                # exactly one first number
    un = {s: L2m[s] for s in st}
    bneg = {s: A(L3[s], w.s(s + 2, 'MI')) + A(L3b[s], w.s(s + 1, 'MI') + w.s(s + 1, 'MN')) for s in st}
    negS = {s: un[s] + bneg[s] for s in st}                                                 # (exclusive)
    nops = lsum(L3.values()) + lsum(L3b.values())
    # ---- validity of commas / points ----
    bad_cm = lsum(A(cm(o), 1 - A(w.D(o - 1), w.D(o - 2), w.D(o - 3), 1 - w.D(o - 4))) for o in range(4, K - 1))
    bad_cm = bad_cm + lsum(A(cm(o), w.D(o + 1), w.D(o + 2), w.D(o + 3), w.D(o + 4)) for o in range(1, K - 4))
    bad_cm = bad_cm + lsum(cm(o) for o in range(1, 4))                                      # a comma group cut by '='
    bad_dot = lsum(A(dt(o), 1 - w.D(o + 1)) for o in range(1, K - 1))                      # '.5' / '5.' : not a number here
    # a point inside the left digit run of a point -> two points ; a comma right of a point -> bad
    fr = {}
    for o in range(1, K - 1):
        for j in range(1, F + 1):
            if o + j < K:
                fr[o, j] = A(w.D(o), *[w.D(o + i) for i in range(1, j)], dt(o + j))
    bad_frac = lsum(A(w.D(o), *[w.D(o + i) for i in range(1, F + 1)], dt(o + F + 1)) for o in range(1, K - F - 2))
    bad_frac = bad_frac + lsum(A(dt(o), *[w.D(o - i) for i in range(1, j)], cm(o - j) + dt(o - j))
                               for o in range(2, K - 1) for j in range(2, F + 3) if o - j >= 1)
    isfrac = {o: lsum(fr[o, j] for j in range(1, F + 1) if (o, j) in fr) for o in range(1, K - 1)}
    # integer digit place: m tokens (digits / commas) to its right before the number end or the point; place = m - commas
    DC = lambda o: w.D(o) + cm(o)
    run = {(o, m): A(*[DC(o - i) for i in range(1, m + 1)], 1 - DC(o - m - 1)) if o - m - 1 >= 0 else Lin()
           for o in range(1, K - 1) for m in range(0, MRUN + 1)}
    pv = {o: lsum(net.sel(m - lsum(cm(o - i) for i in range(1, m + 1)), run[o, m], M=MRUN) for m in range(0, MRUN + 1) if run[o, m].t)
          for o in range(1, K - 1)}
    isint = {o: w.D(o) - isfrac[o] for o in range(1, K - 1)}
    too_long = lsum(A(w.D(o), 1 - isfrac[o], net.ge(pv[o], NL)) for o in range(1, K - 1))
    too_long = too_long + lsum(A(*[DC(o + i) for i in range(0, MRUN + 1)]) for o in range(1, K - MRUN - 1))
    # sign of the number a digit belongs to: number tokens to its left (q of them), then its start at o + q
    QMAX = MRUN + F + 1
    negd = {}
    for o in range(1, K - 1):
        parts = []
        for q in range(0, QMAX + 1):
            if o + q + 1 >= K or (o + q) not in negS: continue
            same = A(*[numtok(o + i) for i in range(1, q + 1)], 1 - numtok(o + q + 1)) if q else A(1 - numtok(o + 1))
            parts.append(A(same, negS[o + q]))
        negd[o] = lsum(parts)
    # ---- signed column sums (register index c = place + F) ----
    Pc, Nc = [], []
    for c in range(NL + F):
        pp, nn = [], []
        for o in range(1, K - 1):
            if c >= F:
                flag = A(isint[o], net.eq(pv[o], c - F)) if pv[o].t or c == F else None
                if flag is None: continue
            else:
                j = F - c
                if (o, j) not in fr: continue
                flag = fr[o, j]
            pp.append(net.sel(w.val(o), flag, 1 - negd[o], sp[o])); nn.append(net.sel(w.val(o), flag, negd[o], sp[o]))
        Pc.append(lsum(pp)); Nc.append(lsum(nn))
    bound = 9 * (K // 3 + 1)
    Pd = net.normalize(Pc, bound)[:PT]; Nd = net.normalize(Nc, bound)[:PT]
    Pd += [Lin()] * (PT - len(Pd)); Nd += [Lin()] * (PT - len(Nd))
    D1, b1 = net.resolve_sub([x - y for x, y in zip(Pd, Nd)])
    D2, _ = net.resolve_sub([y - x for x, y in zip(Pd, Nd)])
    R = [net.sel(x, 1 - b1) + net.sel(y, b1) for x, y in zip(D1, D2)]                       # |value|, LSB first (10^-F units)
    neg = b1
    nzI = [net.ge(r, 1) for r in R[F:]]
    Ld = {n: net.relu(nzI[n - 1] - lsum(nzI[n:])) for n in range(2, PI + 1)}
    Ld[1] = 1 - lsum(Ld.values())
    nzF = [net.ge(r, 1) for r in R[:F]]                                                     # index i = place -(F-i)
    Lf = {n: A(nzF[F - n], *[1 - nzF[i] for i in range(0, F - n)]) if F - n > 0 else nzF[0] for n in range(1, F + 1)}
    Lf[0] = 1 - lsum(Lf.values())
    has_cm = net.clip01(lsum(cm(o) for o in range(1, K - 1)))
    un_lead = net.clip01(lsum(A(un[s], sp[s]) for s in st))
    mine = net.clip01(net.ge(nops, 3) + has_cm + A(un_lead, net.ge(nops, 2)))
    valid = net.relu(A(w.s(0, 'CE'), end_ok, has_term, start_ok, net.ge(nops, 1), mine)
                     - net.clip01(bad_pat) - net.clip01(bad_first) - net.clip01(too_long) - net.clip01(bad_cm)
                     - net.clip01(bad_dot) - net.clip01(bad_frac))
    S = [valid, neg] + R + [Ld[n] for n in range(1, PI + 1)] + [Lf[n] for n in range(0, F + 1)]
    probes = dict(valid=[valid], neg=[neg], R=R, nops=[nops], mine=[mine], bad_pat=[bad_pat], bad_first=[bad_first],
                  start_ok=[start_ok], has_term=[has_term], too_long=[too_long], bad_cm=[bad_cm], bad_dot=[bad_dot],
                  bad_frac=[bad_frac])
    return net, S, probes


def build_stack2():
    """'=' at offset k, the model's lead (' ' / ' $', or ' -' when negative) at k-1, [the machine's '-'], then the m
    tokens written so far (offsets m-1..0). Layout of the value: n integer digits, then (f > 0) '.' and f digits."""
    net = Net(K * NS + N_STATE); w = Win(net); A = net.AND; b = K * NS
    valid, neg = net.inp(b), net.inp(b + 1)
    R = [net.inp(b + 2 + i) for i in range(PT)]
    Ld = {n: net.inp(b + 2 + PT + n - 1) for n in range(1, PI + 1)}
    Lf = {f: net.inp(b + 2 + PT + PI + f) for f in range(0, F + 1)}
    dparts, wd, dotw, fin = [], [], [], []
    for n in range(1, PI + 1):
        for f in range(0, F + 1):
            T = n + (1 if f else 0) + f
            layout = ['D'] * n + (['DOT'] + ['D'] * f if f else [])
            for mode, s in (('a', 0), ('a', 1), ('b', 0)):
                for xdot in (0, 1):                        # xdot: the model wrote a '.' after the finished number
                    for m in range(0, T + 1):
                        if xdot and m != T: continue
                        k = m + 1 + s + xdot
                        if k >= K: continue
                        lead = (w.s(k - 1, 'SP') + w.s(k - 1, 'DL')) if mode == 'a' else w.s(k - 1, 'MI')
                        parts = [w.s(k, 'CE'), lead] + ([w.s(k - 2, 'MN')] if s else [])
                        for i in range(m):                 # written token i at offset m-1-i (+xdot)
                            o = m - 1 - i + xdot
                            parts.append(w.D(o) if layout[i] == 'D' else w.s(o, 'DOT'))
                        if xdot: parts.append(w.s(0, 'DOT'))
                        parts += [neg] if (mode == 'b' or s) else [1 - neg]
                        g = A(*parts, Ld[n], Lf[f])
                        if m == T: fin.append(g); continue
                        if layout[m] == 'DOT': dotw.append(g); continue
                        idx = F + n - 1 - m if m < n else F - (m - n)        # register index of the next digit
                        dparts.append(net.sel(R[idx], g)); wd.append(g)
    sgn = A(w.s(1, 'CE'), w.s(0, 'SP') + w.s(0, 'DL'), neg)                    # negative, mode a: write '-' first
    cur, wdig = lsum(dparts), lsum(wd)
    out = {str(d): A(valid, wdig, net.eq(cur, d)) for d in range(10)}
    out['DOT'] = A(valid, lsum(dotw)); out['MN'] = A(valid, sgn); out['OE'] = A(valid, lsum(fin)); out['OS'] = Lin()
    probes = {'cur': [cur], 'wdig': [wdig], 'fin': [lsum(fin)]}
    return net, [out[t] for t in OUT], probes


def build():
    n1, S, _ = build_stack1(); n2, O, _ = build_stack2()
    assert len(S) == N_STATE
    return compile_net(n1, S), compile_net(n2, O)


def configure(tok):
    """keep only the start words that are single tokens for this host's tokenizer; rebuild the symbol tables."""
    global SAFE_TEXT, SAFES, SYMS, NS, SI, TEXT
    SAFE_TEXT = tuple(t for t in SAFE_TEXT if len(tok(t, add_special_tokens=False)['input_ids']) == 1)
    SAFES = [f'S{i}' for i in range(len(SAFE_TEXT))]
    SYMS = list('0123456789') + ['PL', 'MI', 'MN', 'SP', 'DL', 'DOT', 'CM', 'CE'] + SAFES
    NS = len(SYMS); SI = {s: i for i, s in enumerate(SYMS)}
    TEXT = {**{c: c for c in '0123456789'}, 'PL': (' +', '+'), 'MI': ' -', 'MN': '-', 'SP': ' ', 'DL': (' $', '$'), 'DOT': '.',
            'CM': ',', 'CE': ('=', ' ='), **dict(zip(SAFES, SAFE_TEXT)), 'OS': ' ', 'OE': ' '}


def register(tok=None):
    if tok is not None: configure(tok)
    from micromachines.specs import SPECS
    SPECS['chain2'] = dict(build=build, syms=SYMS, out=OUT, key='CE', K=K, W=PT + 6, text=TEXT)
