"""Sum micromachine (MM13): exact signed integer sums written in the model's own natural style, e.g.
    'parentheses: 1 + 1 + 2 + 1 + 3 ='     'A = (-1 + 2 + 45) ='     '((9 - (-56)) + -9) ='     '3969 + -40 ='
No tags: it reads the text before '=' back to a start token (':', '=', a newline, ' is', ...). The model writes the space
(or ' -') after '=' itself; the machine then writes the sign and the digits and ends with a veto on digits.

Value = sum over numbers of (binary op sign) x (unary minus sign) x value; brackets are transparent, and a binary '-' in
front of a bracket that holds more than one number (which would need the sign distributed) makes the call invalid.
It answers only where the calculator (circuits.py) is silent: some bracket, or >= 3 operators, or a unary minus that is
not on the first number, or a leading unary minus with >= 2 operators. Anything else it cannot certify: silent.
"""
from ..hdl import Net, Lin, lsum, compile_net

SAFE_TEXT = (':', '\n', '.\n', ':\n', ',\n', ' \n', ' is', ' are', ' was', ' were', ' be', ' equals', ' gives', ' get', ' gets',
             ' got', ' has', ' have', ' had', ' makes', ' total', ' A', ' B', 'A', 'B')
SAFES = [f'S{i}' for i in range(len(SAFE_TEXT))]           # one symbol per start token (a single linear reader cannot pick the whole set)
SYMS = list('0123456789') + ['PL', 'MI', 'SP', 'O1', 'O2', 'OM', 'C1', 'C2', 'CE', 'MN', 'DOT', 'TI'] + SAFES
NS = len(SYMS); SI = {s: i for i, s in enumerate(SYMS)}
K = 80                      # window (tokens back from '=')
NL = 6                      # max digits of one number
P = NL + 3                  # digits of the result (up to ~25 numbers of NL digits)
OUT = list('0123456789') + ['MN', 'OS', 'OE']
N_STATE = 2 + 2 * P
TEXT = {**{c: c for c in '0123456789'}, 'PL': ' +', 'MI': ' -', 'SP': ' ', 'O1': (' (', '('), 'O2': (' ((', '(('), 'OM': (' (-', '(-'),
        'C1': (')', ' )'), 'C2': '))', 'CE': ('=', ' ='), **dict(zip(SAFES, SAFE_TEXT)), 'MN': '-', 'DOT': '.', 'TI': ' *', 'OS': ' ', 'OE': ' '}


class Win:
    def __init__(self, net): self.net = net
    def s(self, o, sym): return self.net.inp(o * NS + SI[sym]) if 0 <= o < K else Lin()
    def D(self, o): return lsum(self.s(o, str(v)) for v in range(10))
    def val(self, o): return lsum(self.s(o, str(v)) * v for v in range(1, 10))
    def cls(self, o): return self.D(o) + lsum(self.s(o, c) for c in ('PL', 'MI', 'SP', 'O1', 'O2', 'OM', 'C1', 'C2'))
    def opn(self, o): return lsum(self.s(o, c) for c in ('SP', 'MI', 'O1', 'O2', 'OM'))   # may stand right before a number
    def bop(self, o): return self.s(o, 'PL') + self.s(o, 'MI')
    def cl(self, o): return self.s(o, 'C1') + self.s(o, 'C2')
    def safe(self, o): return lsum(self.s(o, c) for c in SAFES)
    def par(self, o): return lsum(self.s(o, c) for c in ('O1', 'O2', 'OM', 'C1', 'C2'))


def build_stack1(nonneg=False):
    net = Net(K * NS); w = Win(net); A = net.AND
    xor = lambda x, y: x + y - 2 * A(x, y)
    # ---- the span: offsets 1..t-1 are expression tokens, offset t is the start token ----
    sp = {o: net.relu(lsum(w.cls(j) for j in range(1, o + 1)) - (o - 1)) for o in range(1, K)}
    sp[0] = Lin(c=1); spg = lambda o: sp.get(o, Lin())
    term = {t: A(sp[t - 1], 1 - w.cls(t)) for t in range(2, K)}
    termg = lambda t: term.get(t, Lin())
    has_term = lsum(term.values())                                           # 0 if the expression runs past the window
    start_ok = lsum(A(term[t], w.safe(t) + w.s(t, 'CE')) for t in term)
    end_ok = w.D(1) + w.cl(1)
    # ---- numbers: start s (most significant digit), length L ----
    st = {s: A(w.D(s), 1 - w.D(s + 1), sp[s]) for s in range(1, K - 1)}
    L3 = {s: A(st[s], w.opn(s + 1), w.bop(s + 2), w.D(s + 3) + w.cl(s + 3)) for s in st}       # '... x op [open] N'
    L1 = {s: A(st[s], termg(s + 1)) for s in st}                                                # first number, nothing before
    L2 = {s: A(st[s], w.opn(s + 1), termg(s + 2)) for s in st}                                  # first number, one opener
    L2b = {s: A(st[s], w.opn(s + 1), w.s(s + 2, 'O1') + w.s(s + 2, 'O2'), termg(s + 3)) for s in st}
    pat_ok = {s: L1[s] + L2[s] + L2b[s] + L3[s] for s in st}
    bad_pat = lsum(st[s] - pat_ok[s] for s in st)
    un = {s: A(st[s], w.s(s + 1, 'MI') + w.s(s + 1, 'OM')) for s in st}                        # unary minus on this number
    bneg = {s: A(L3[s], w.s(s + 2, 'MI')) for s in st}                                         # binary minus before it
    negS = {s: xor(un[s], bneg[s]) for s in st}
    ln = {(s, L): A(st[s], *[w.D(s - j) for j in range(1, L)], 1 - w.D(s - L)) for s in st for L in range(1, NL + 1) if s - L >= 0}
    too_long = lsum(A(*[w.D(o + j) for j in range(NL + 1)]) for o in range(1, K - NL - 1))
    # a binary '-' in front of a bracket: the bracket must close right after that one number (')' for '(' / '(-', '))' for '((')
    c1 = {s: lsum(A(ln[s, L], w.cl(s - L)) for L in range(1, NL + 1) if (s, L) in ln) for s in st}
    c2 = {s: lsum(A(ln[s, L], w.s(s - L, 'C2')) for L in range(1, NL + 1) if (s, L) in ln) for s in st}
    dist_bad = lsum(A(bneg[s], w.s(s + 1, 'O1') + w.s(s + 1, 'OM'), 1 - c1[s]) + A(bneg[s], w.s(s + 1, 'O2'), 1 - c2[s]) for s in st)
    # ---- when the calculator is silent: bracket / >= 3 ops / unary minus not on the first number / leading unary + >= 2 ops ----
    nops = lsum(L3.values())
    haspar = net.clip01(lsum(A(sp[o], w.par(o)) for o in range(1, K)))
    un_lead = net.clip01(lsum(A(un[s], L1[s] + L2[s] + L2b[s]) for s in st))
    un_non = net.clip01(lsum(A(un[s], L3[s]) for s in st))
    mine = net.clip01(haspar + net.ge(nops, 3) + un_non + A(un_lead, net.ge(nops, 2)))
    # ---- signed column sums: digit at offset o, place p, sign of its number ----
    same = {(o, q): A(*[w.D(o + j) for j in range(q + 1)], 1 - w.D(o + q + 1)) for o in range(1, K - 1) for q in range(NL) if o + q + 1 < K}
    negd = {o: lsum(A(same[o, q], negS[o + q]) for q in range(NL) if (o, q) in same and o + q in negS) for o in range(1, K - 1)}
    place = {(o, p): A(*[w.D(o - j) for j in range(p + 1)], 1 - w.D(o - p - 1)) for o in range(1, K - 1) for p in range(NL) if o - p - 1 >= 0}
    Pc, Nc = [], []
    for p in range(NL):
        Pc.append(lsum(net.sel(w.val(o), place[o, p], 1 - negd[o], sp[o]) for o in range(1, K - 1) if (o, p) in place))
        Nc.append(lsum(net.sel(w.val(o), place[o, p], negd[o], sp[o]) for o in range(1, K - 1) if (o, p) in place))
    bound = 9 * (K // 3 + 1)
    Pd = net.normalize(Pc, bound)[:P]; Nd = net.normalize(Nc, bound)[:P]
    Pd += [Lin()] * (P - len(Pd)); Nd += [Lin()] * (P - len(Nd))
    D1, b1 = net.resolve_sub([x - y for x, y in zip(Pd, Nd)])
    D2, _ = net.resolve_sub([y - x for x, y in zip(Pd, Nd)])
    R = [net.sel(x, 1 - b1) + net.sel(y, b1) for x, y in zip(D1, D2)]                       # |sum|, LSB first
    neg = b1
    nz = [net.ge(r, 1) for r in R]
    Ld = {n: net.relu(nz[n - 1] - lsum(nz[n:])) for n in range(2, P + 1)}                    # number of digits (one-hot)
    Ld[1] = 1 - lsum(Ld.values())
    valid = net.relu(A(w.s(0, 'CE'), end_ok, has_term, start_ok, net.ge(nops, 1), mine) - net.clip01(bad_pat) - net.clip01(too_long)
                     - net.clip01(dist_bad) - (neg if nonneg else 0))
    S = [valid, neg] + R + [Ld[n] for n in range(1, P + 1)]
    probes = {'valid': [valid], 'neg': [neg], 'R': R, 'ndig': [Ld[n] for n in range(1, P + 1)], 'nops': [nops], 'mine': [mine],
              'bad_pat': [bad_pat], 'dist_bad': [dist_bad], 'start_ok': [start_ok], 'has_term': [has_term], 'end_ok': [end_ok]}
    return net, S, probes


def build_stack2(force_os=False):
    """CE at offset k, the model's lead token (' ' or ' -') at k-1, then what has been written since (current token = offset 0).
    mode a: lead ' ', [the machine's '-' if negative], digits     mode b: lead ' -' (only if negative), digits.
    dot: the current token is a '.' the model wrote; right after the last digit that means 'the number is over' -> veto."""
    net = Net(K * NS + N_STATE); w = Win(net); A = net.AND; b = K * NS
    valid, neg = net.inp(b), net.inp(b + 1)
    R = [net.inp(b + 2 + i) for i in range(P)]; Ld = {n: net.inp(b + 2 + P + n - 1) for n in range(1, P + 1)}
    digit_parts, wd, fin = [], [], []
    for k in range(1, P + 4):
        for mode, s in (('a', 0), ('a', 1), ('b', 0)):
            for dot in (0, 1):
                m = k - 1 - s - dot                                                 # digits written so far
                if m < 0: continue
                lead = w.s(k - 1, 'SP') if mode == 'a' else w.s(k - 1, 'MI')
                parts = [w.s(k, 'CE'), lead] + ([w.s(k - 2, 'MN')] if s else []) + [w.D(o) for o in range(dot, dot + m)]
                parts += [w.s(0, 'DOT')] if dot else []
                parts += [neg] if (mode == 'b' or s) else [1 - neg]
                e = A(*parts)
                for n in range(1, P + 1):
                    g = A(e, Ld[n])
                    if m < n and not dot: digit_parts.append(net.sel(R[n - 1 - m], g)); wd.append(g)
                    if m == n: fin.append(g)
    sgn = A(w.s(1, 'CE'), w.s(0, 'SP'), neg)                                        # negative, mode a: write '-' first
    cur, wdig, fin = lsum(digit_parts), lsum(wd), lsum(fin)                         # at most one combination is active
    out = {str(d): A(valid, wdig, net.eq(cur, d)) for d in range(10)}
    out['MN'] = A(valid, sgn); out['OE'] = A(valid, fin)
    out['OS'] = A(w.s(0, 'CE'), valid) if force_os else Lin()       # force_os: answer at once, before the model rewrites the sum
    probes = {'sign': [sgn], 'cur': [cur], 'write digit': [wdig], 'fin': [fin], 'out': [out[t] for t in OUT]}
    return net, [out[t] for t in OUT], probes


def build(nonneg=False, force_os=False):
    n1, S, _ = build_stack1(nonneg); n2, O, _ = build_stack2(force_os)
    assert len(S) == N_STATE
    return compile_net(n1, S), compile_net(n2, O)


def sum_ref(expr):
    """Python reference for what the machine should write after 'expr =' ('' = silent)."""
    import re
    e = expr.strip()
    if not re.fullmatch(r'[\d+\-() ]+', e) or not re.search(r'\d', e): return ''
    try: v = eval(e)
    except Exception: return ''
    return str(v)


# ======================= MM15: small expression machine (BBH multistep sub-expressions) =======================
def build_expr_stack1():
    """'A = (-1 + 2 + 9 * 5) =' : 2..4 signed SINGLE-digit numbers, ops + - *, usual precedence, brackets only as one outer
    pair. Left-to-right scan with a running sum S and a running product term P ('* x' multiplies P; '+ x'/'- x' moves P
    into S and starts a new term), all exact integer ReLU arithmetic. Fires only with at least one '*' and (>= 3 numbers or
    a unary minus or the outer brackets), so plain 'a * b =' stays with the calculator."""
    net = Net(K * NS); w = Win(net); A = net.AND
    M = 10 ** 5
    gate = lambda x, f: net.relu(x - M * (1 - f)) - net.relu(-x - M * (1 - f))           # x if f else 0, |x| < M
    cls = lambda o: w.cls(o) + w.s(o, 'TI')
    bop = lambda o: w.bop(o) + w.s(o, 'TI')
    sp = {o: net.relu(lsum(cls(j) for j in range(1, o + 1)) - (o - 1)) for o in range(1, K)}
    term = {t: A(sp[t - 1], 1 - cls(t)) for t in range(2, K)}; termg = lambda t: term.get(t, Lin())
    has_term = lsum(term.values())
    start_ok = lsum(A(term[t], w.safe(t) + w.s(t, 'CE')) for t in term)
    end_ok = w.D(1) + w.s(1, 'C1')
    Dsp = {o: A(w.D(o), sp[o]) for o in range(1, K)}
    n = lsum(Dsp.values())
    adj = lsum(A(w.D(o), w.D(o + 1), sp[o + 1]) for o in range(1, K - 1))
    cnt = {o: lsum(w.D(u) for u in range(1, o)) for o in range(1, K)}
    st = {o: A(Dsp[o], 1 - w.D(o + 1)) for o in range(1, K - 1)}
    L3 = {o: A(st[o], w.opn(o + 1), bop(o + 2), w.D(o + 3) + w.cl(o + 3)) for o in st}
    L1 = {o: A(st[o], termg(o + 1)) for o in st}
    L2 = {o: A(st[o], w.opn(o + 1), termg(o + 2)) for o in st}
    L2b = {o: A(st[o], w.opn(o + 1), w.s(o + 2, 'O1') + w.s(o + 2, 'O2'), termg(o + 3)) for o in st}
    bad_pat = lsum(st[o] - L1[o] - L2[o] - L2b[o] - L3[o] for o in st)
    npar = lsum(A(sp[o], w.par(o)) for o in range(1, K))
    outer = net.clip01(lsum(A(L2[o], w.s(o + 1, 'O1') + w.s(o + 1, 'OM')) + A(L2b[o], w.s(o + 2, 'O1')) for o in st))
    par_ok = net.eq(npar, 0) + A(net.eq(npar, 2), w.s(1, 'C1'), outer)
    un = {o: A(st[o], w.s(o + 1, 'MI') + w.s(o + 1, 'OM')) for o in st}
    un_any = net.clip01(lsum(un.values()))
    # per number i (1 = rightmost): digit one-hot, unary sign, operator in front of it
    idx = {(o, i): A(st[o], net.eq(cnt[o], i - 1)) for o in st for i in range(1, 5)}
    dv = {(i, v): lsum(A(idx[o, i], w.s(o, str(v))) for o in st) for i in range(1, 5) for v in range(10)}
    ui = {i: lsum(A(idx[o, i], un[o]) for o in st) for i in range(1, 5)}
    opk = {(i, c): lsum(A(idx[o, i], L3[o], w.s(o + 2, c)) for o in st) for i in range(1, 5) for c in ('PL', 'MI', 'TI')}
    has_mul = net.clip01(lsum(opk[i, 'TI'] for i in range(1, 5)))
    sv = {i: lsum(v * A(dv[i, v], 1 - ui[i]) - v * A(dv[i, v], ui[i]) for v in range(1, 10)) for i in range(1, 5)}
    def times(X, i):                                                                       # X * (signed number i)
        return lsum(v * (gate(X, A(dv[i, v], 1 - ui[i])) - gate(X, A(dv[i, v], ui[i]))) for v in range(1, 10))
    Vc = {}
    for c in (2, 3, 4):                                                                    # c numbers: the leftmost is number c
        Pt, S = sv[c], Lin()
        for i in range(c - 1, 0, -1):
            mul, pl, mi = opk[i, 'TI'], opk[i, 'PL'], opk[i, 'MI']
            S = S + gate(Pt, 1 - mul)
            Pt = gate(times(Pt, i), mul) + gate(sv[i], pl) - gate(sv[i], mi)
        Vc[c] = S + Pt
    ncase = {c: net.eq(n, c) for c in (2, 3, 4)}
    V = lsum(gate(Vc[c], ncase[c]) for c in (2, 3, 4))
    neg = net.ge(-V, 1); rem = net.relu(V) + net.relu(-V)
    Rd = [Lin()] * P
    for k in range(4, -1, -1):
        d = lsum(net.ge(rem, j * 10 ** k) for j in range(1, 10)); Rd[k] = d; rem = rem - 10 ** k * d
    nz = [net.ge(r, 1) if isinstance(r, Lin) and r.t else Lin() for r in Rd]
    Ld = {m: net.relu(nz[m - 1] - lsum(nz[m:])) for m in range(2, P + 1)}; Ld[1] = 1 - lsum(Ld.values())
    mine = net.clip01(net.ge(n, 3) + un_any + net.ge(npar, 1))
    valid = net.relu(A(w.s(0, 'CE'), end_ok, has_term, start_ok, net.ge(n, 2), 1 - net.ge(n, 5), has_mul, mine, par_ok)
                     - net.clip01(bad_pat) - net.clip01(adj))
    S_ = [valid, neg] + Rd + [Ld[m] for m in range(1, P + 1)]
    return net, S_, {'valid': [valid], 'V': [V], 'n': [n], 'neg': [neg], 'R': Rd}


def build_expr():
    n1, S, _ = build_expr_stack1(); n2, O, _ = build_stack2(force_os=True)
    assert len(S) == N_STATE
    return compile_net(n1, S), compile_net(n2, O)
