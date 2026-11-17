"""CalcNet circuits: exact decimal + - * / as fixed ReLU networks over a window of token symbols.

Two feed-forward stacks, joined by one attention hop:
  stack 1  runs at the </tool_call> position: parses "A op B" and does the per-call work
  (attention copies that call state to later positions)
  stack 2  runs at every output position: decides the next output token
Numbers are fixed-point digit registers; every wire is a small integer.
"""
from ..hdl import Net, Lin, lsum, compile_net
from ..calcfmt import NI, F, D

NOP = NI + F                     # digits in an operand register (LSB-first, index = place + F)
OPLEN = NOP + 1                  # max chars of an operand (with the point, without the sign)
NEGOP = {'+': 'N+', '-': 'N-', '*': 'N*', '/': 'N/'}     # merged tokens "+-", "--", "*-", "/-": operator, then a negative B
SYMS = list('0123456789') + ['+', '-', '*', '/', '.', 'CS', 'CE', 'OS', 'OE'] + list(NEGOP.values()) + ['SP', 'CSN', 'X2']   # X2: a second multiply class (' x', ' ×'), ORed with '*' (Qwen3 port); SP: a lone space token (v4, spaced style); CSN: a second start-of-expression class (v5b)
NS = len(SYMS)
SI = {s: i for i, s in enumerate(SYMS)}
K = 2 * OPLEN + 4                # window: offsets 0..K-1 back from the current token
KU = 2 * NI                      # unified result index k = 1..KT (MSB first); place(k) = KU - k
KT = KU + 2 * F                  # units digit is k = KU, fraction is k = KU+1..KT
OUT = list('0123456789') + ['.', '-', 'OS', 'OE']
assert D <= F and NOP + D <= KT


class Win:
    """Symbol window accessors (offset 0 = current token)."""

    def __init__(self, net, base=0):
        self.net, self.base = net, base

    def s(self, o, sym):
        if sym == '*' and 0 <= o < K:                                        # Qwen3 port: either multiply class
            return self.net.inp(self.base + o * NS + SI['*']) + self.net.inp(self.base + o * NS + SI['X2'])
        if sym == 'CS' and 0 <= o < K:                                       # v5b: either start class opens an expression
            return self.net.inp(self.base + o * NS + SI['CS']) + self.net.inp(self.base + o * NS + SI['CSN'])
        return self.net.inp(self.base + o * NS + SI[sym]) if 0 <= o < K else Lin()

    def dig(self, o): return lsum(self.s(o, str(v)) for v in range(10))
    def dval(self, o): return lsum(self.s(o, str(v)) * v for v in range(1, 10))
    def dot(self, o): return self.s(o, '.')
    def isop(self, o): return lsum(self.s(o, c) for c in ['+', '-', '*', '/'] + list(NEGOP.values()))
    def negop(self, o): return lsum(self.s(o, c) for c in NEGOP.values())
    def sp(self, o): return self.s(o, 'SP')


def parse_operand(net, dig, dval, dot, term):
    """dig/dval/dot/term: dicts offset(1..OPLEN+1) -> wire, offset 1 = last char of the operand,
    term[n] = the terminator sits at offset n.  -> (register[NOP], valid)"""
    V = lambda n: lsum(2 * dig[o] + dot[o] for o in range(1, n)) - 2 * (n - 1)   # 0: all digits, -1: one point
    T0, T1 = {}, {}
    for n in range(2, OPLEN + 2):
        v = V(n)
        T0[n] = net.relu(term[n] + v)
        T1[n] = net.relu(2 * term[n] + v) - net.relu(2 * term[n] + v - 1) - T0[n]
    pt = {j: net.relu(dot[j] + lsum(dig[o] for o in range(1, j)) - (j - 1)) for j in range(2, F + 2)}
    valid = lsum([T0[n] for n in range(2, NI + 2)] +
                 [net.AND(pt[j], T1[n]) for j in pt for n in range(j + 2, min(j + NI + 1, OPLEN + 1) + 1)])
    nopt = 1 - lsum(pt.values())
    inop = {o: net.relu(lsum(dig[u] + dot[u] for u in range(1, o + 1)) - (o - 1)) for o in range(1, OPLEN + 1)}
    reg = []
    for r in range(NOP):
        parts = []
        if r >= F: parts.append(net.sel(dval[r - F + 1], nopt, inop[r - F + 1]))
        for j in pt:
            f = j - 1
            off = f - (F - r) + 1 if r < F else j + 1 + (r - F)
            if 1 <= off <= OPLEN and not (r < F and F - r > f):
                parts.append(net.sel(dval[off], pt[j], inop[off]))
        reg.append(lsum(parts))
    return reg, valid


def therm(net, reg): return [[net.ge(x, u) for u in range(1, 10)] for x in reg]   # therm[i][u-1] = [digit >= u]


def mul_cols(net, X, tY, ncols):
    """column sums of X*Y; X digits (scalars), tY thermometer of Y."""
    cols = [[] for _ in range(ncols)]
    for i, x in enumerate(X):
        for j, t in enumerate(tY):
            if i + j < ncols:
                cols[i + j] += [net.sel(x, t[u]) for u in range(9)]
    return [lsum(c) for c in cols]


def build_stack1(spaced=False, nonneg=False):
    net = Net(K * NS); w = Win(net)
    # ---- operand B sits right before </tool_call>; the operator terminates it ----
    rng = range(1, OPLEN + 2)
    # v4 spaced style ('21 / 7 =' -> 2 1 ' /' ' ' 7 ' ='): B may be preceded by a lone space before the operator
    termB = {o: w.isop(o) + (net.AND(w.sp(o), w.isop(o + 1)) if spaced else 0) for o in rng}
    Breg, validB = parse_operand(net, {o: w.dig(o) for o in rng}, {o: w.dval(o) for o in rng},
                                 {o: w.dot(o) for o in rng}, termB)
    opat = {k: net.relu(w.isop(k) + lsum(w.dig(o) + w.dot(o) for o in range(1, k)) - (k - 1)) for k in range(2, OPLEN + 2)}
    opsp = {k: net.relu(w.sp(k) + w.isop(k + 1) + lsum(w.dig(o) + w.dot(o) for o in range(1, k)) - k) for k in range(2, OPLEN + 2)} if spaced else {}
    op = {c: lsum(net.AND(opat[k], w.s(k, c) + w.s(k, NEGOP[c])) for k in opat) + lsum(net.AND(opsp[k], w.s(k + 1, c)) for k in opsp) for c in '+-*/'}
    sB = lsum(net.AND(opat[k], w.negop(k)) for k in opat)
    # ---- align operand A to the operator, then parse it (terminator: <tool_call>, or "-" then <tool_call>) ----
    al = lambda f, m: lsum(net.AND(opat[k], f(k + m)) for k in opat) + lsum(net.AND(opsp[k], f(k + 1 + m)) for k in opsp)
    a_dig = {m: al(w.dig, m) for m in rng}; a_dot = {m: al(w.dot, m) for m in rng}
    a_cs = {m: al(lambda o: w.s(o, 'CS'), m) for m in range(1, OPLEN + 3)}
    a_min = {m: al(lambda o: w.s(o, '-'), m) for m in rng}
    termS = {n: net.relu(a_min[n] + a_cs[n + 1] + lsum(a_dig[o] + a_dot[o] for o in range(1, n)) - n) for n in rng}
    sA = lsum(termS[n] for n in range(2, OPLEN + 2))
    Areg, validA = parse_operand(net, a_dig, {m: lsum(net.sel(w.dval(k + m), opat[k]) for k in opat) + lsum(net.sel(w.dval(k + 1 + m), opsp[k]) for k in opsp) for m in rng},
                                 a_dot, {n: a_cs[n] + termS[n] for n in rng})
    chainv = None
    if spaced:   # v4: 'x + 0.30 + 0.20 =' -- the space before A follows another operator: a chain, not a fresh call
        a_op = {m: al(w.isop, m) for m in range(1, OPLEN + 3)}
        chn = {n: net.relu(a_cs[n] + a_op[n + 1] + lsum(a_dig[o] + a_dot[o] for o in range(1, n)) - n) for n in range(2, OPLEN + 2)}
        chain = net.clip01(lsum(chn.values()))
        validA_raw = validA
        validA = net.AND(validA, 1 - chain)
        # v5: two-operation chains 'a op1 b op2 c =' (operands up to NB chars) are reduced to ONE operation for the exact core
        NB = 12
        pos = [(opat[k], k) for k in opat if k <= NB + 1] + [(opsp[k], k + 1) for k in opsp if k <= NB + 1]
        z = {}                                                                # z[j]: the last char of a sits at absolute offset j
        for f_k, pk in pos:
            for n in range(2, NB + 2):
                j = pk + n + 2
                if j + NB + 2 < K: z.setdefault(j, []).append(net.AND(f_k, chn[n]))
        z = {j: lsum(v) for j, v in z.items()}
        rA = range(1, NB + 2)
        g = lambda f, m: lsum(net.AND(z[j], f(j + m - 1)) for j in z)
        c_dig = {m: g(w.dig, m) for m in rA}; c_dot = {m: g(w.dot, m) for m in rA}
        c_cs = {m: g(lambda o: w.s(o, 'CS'), m) for m in range(1, NB + 3)}; c_op = {m: g(w.isop, m) for m in range(1, NB + 3)}
        c_dval = {m: lsum(net.sel(w.dval(j + m - 1), z[j]) for j in z) for m in rA}
        pad = lambda d: {m: d.get(m, Lin()) for m in rng}
        Creg, validC = parse_operand(net, pad(c_dig), pad(c_dval), pad(c_dot), pad(c_cs))
        chain3 = net.clip01(lsum(net.relu(c_cs[n] + c_op[n + 1] + lsum(c_dig[o] + c_dot[o] for o in range(1, n)) - n) for n in range(2, NB + 2)))
        op1 = {c: lsum(net.AND(z[j], w.s(j - 1, c)) for j in z) for c in '+-*/'}
        isch = net.clip01(lsum(z.values()))
        pm1, pm2 = op1['+'] + op1['-'], op['+'] + op['-']
        # partial results: a (+|-) b, and one product X*Y with X, Y chosen by the case
        P_add = net.resolve([x + y for x, y in zip(Creg, Areg)])
        Q1, qneg = net.resolve_sub([x - y for x, y in zip(Creg, Areg)])
        P_sum = [net.sel(x, op1['+']) + net.sel(y, op1['-']) for x, y in zip(P_add, Q1 + [Lin()])]
        xa = net.clip01(op1['*'] + net.AND(op1['/'], op['*']))                # X = a, else b
        Xr = [net.sel(x, xa) + net.sel(y, 1 - xa) for x, y in zip(Creg, Areg)]
        Yr = [net.sel(x, op1['*']) + net.sel(y, 1 - op1['*']) for x, y in zip(Areg, Breg)]   # Y = b if op1 is *, else c
        Pm = net.normalize(mul_cols(net, Yr, therm(net, Xr), 2 * NOP - 1), NOP * 81)[:2 * NOP]
        Preg = Pm[F:F + NOP]
        prod_ok = 1 - net.clip01(lsum(Pm[:F] + Pm[F + NOP:]))               # no lost decimals, no overflow
        sum_ok = 1 - net.clip01(P_sum[NOP] if len(P_sum) > NOP else Lin())
        cL = net.AND(pm1, pm2); cPm = net.AND(pm1, op['*']); cM = op1['*']; cDm = net.AND(op1['/'], op['*']); cDd = net.AND(op1['/'], op['/'])
        A2 = [net.sel(p_, cL) + net.sel(a_, net.clip01(cPm + cDd)) + net.sel(q_, net.clip01(cM + cDm)) for p_, a_, q_ in zip(P_sum[:NOP], Creg, Preg)]
        B2 = [net.sel(c_, net.clip01(cL + cM)) + net.sel(q_, net.clip01(cPm + cDd)) + net.sel(b_, cDm) for c_, q_, b_ in zip(Breg, Preg, Areg)]
        op2n = {c: net.AND(op[c], net.clip01(cL + cM)) + net.AND(op1[c], cPm) + (net.clip01(cDm + cDd) if c == '/' else 0) for c in '+-*/'}
        needs_p = net.clip01(cPm + cM + cDm + cDd)
        ok_case = net.clip01(cL + cPm + cM + cDm + cDd)
        chainv = net.AND(isch, validA_raw, validC, 1 - chain3, ok_case, 1 - sB, 1 - sA, 1 - net.AND(needs_p, 1 - prod_ok), 1 - net.AND(cL, 1 - sum_ok),
                         1 - net.AND(cL, op1['-'], qneg))                     # a - b < 0 inside a chain: not handled -> reject
        sA2 = Lin()
        # the reduced call replaces the plain one when the chain is valid
        Areg = [net.sel(x, chainv) + net.sel(y, 1 - chainv) for x, y in zip(A2, Areg)]
        Breg = [net.sel(x, chainv) + net.sel(y, 1 - chainv) for x, y in zip(B2, Breg)]
        op = {c: net.AND(op2n[c], chainv) + net.AND(op[c], 1 - chainv) for c in '+-*/'}
        sA = net.AND(sA2, chainv) + net.AND(sA, 1 - chainv)
        validA = validA + chainv
    tA, tB = therm(net, Areg), therm(net, Breg)

    # ---- add / subtract ----
    S_add = net.resolve([a + b for a, b in zip(Areg, Breg)])                # NOP+1 digits
    D1, neg = net.resolve_sub([a - b for a, b in zip(Areg, Breg)])
    D2, _ = net.resolve_sub([b - a for a, b in zip(Areg, Breg)])
    S_sub = [net.sel(x, 1 - neg) + net.sel(y, neg) for x, y in zip(D1, D2)]
    # ---- multiply ----
    cols_mul = mul_cols(net, Breg, tA, 2 * NOP - 1)
    S_mul = net.normalize(cols_mul, NOP * 81)[:2 * NOP]

    # ---- signs: which magnitude operation is it, and is the result negative? ----
    xor = lambda x, y: x + y - 2 * net.AND(x, y)
    isas = op['+'] + op['-']
    eff_sub = xor(sA, xor(sB, op['-']))                                        # the two terms have opposite signs
    use = {'+': net.AND(isas, 1 - eff_sub), '-': net.AND(isas, eff_sub), '*': op['*']}

    # ---- unified result register, MSB first: R[k], k = 1..KT, place = KU - k ----
    def at(digs, place, shift):                                               # digit of a fixed-point list
        i = place + shift
        return digs[i] if 0 <= i < len(digs) else None
    R = {}
    for k in range(1, KT + 1):
        pl = KU - k
        parts = [net.sel(x, use[c]) for c, x in (('+', at(S_add, pl, F)), ('-', at(S_sub, pl, F)), ('*', at(S_mul, pl, 2 * F))) if x is not None]
        R[k] = lsum(parts)
    nzR = net.clip01(lsum(R.values()))
    qnz = net.cmp_ge([Lin()] * D + Areg, Breg)                                # truncated quotient is non-zero
    sAB = xor(sA, sB)
    neg_out = lsum([net.AND(use['+'], sA, nzR), net.AND(use['-'], xor(sA, neg), nzR), net.AND(op['*'], sAB, nzR), net.AND(op['/'], sAB, qnz)])
    nz = {k: net.ge(R[k], 1) for k in range(1, KU)}
    lead = {k: net.relu(nz[k] - lsum(nz[j] for j in range(1, k))) for k in nz}
    k0pm = dict(lead); k0pm[KU] = 1 - lsum(lead.values())
    tz = {k: 1 - net.clip01(lsum(R[j] for j in range(k, KT + 1))) for k in range(KU + 1, KT + 1)}

    # ---- division: multiples of B, and the index of the first quotient digit ----
    def units_tens(j, t):                                                     # digit of B -> (j*b % 10, j*b // 10), affine in the thermometer
        u = lsum(((j * v) % 10 - (j * (v - 1)) % 10) * t[v - 1] for v in range(1, 10))
        tn = lsum(((j * v) // 10 - (j * (v - 1)) // 10) * t[v - 1] for v in range(1, 10))
        return u, tn
    jB = {1: Breg + [Lin()]}
    for j in range(2, 10):
        ut = [units_tens(j, t) for t in tB]
        cols = [lsum([ut[i][0] if i < NOP else 0, ut[i - 1][1] if i >= 1 else 0]) for i in range(NOP + 1)]
        jB[j] = net.resolve(cols)[:NOP + 1]
    T = {0: Lin()}
    for kd in range(1, NOP):                                                  # [first kd digits of A' >= B']
        Nk = [Areg[NOP - kd + i] for i in range(kd)]
        T[kd] = net.cmp_ge(Nk, Breg)
    k0d = {kd: T[kd] - T[kd - 1] for kd in range(1, NOP)}; k0d[NOP] = 1 - T[NOP - 1]
    bzero = 1 - net.clip01(lsum(Breg))
    isdiv = op['/']
    off = KU - NOP                                                            # division index kd -> unified k = kd + off
    k0 = {}
    for k in range(1, KU + 1):
        parts = [net.AND(k0pm[k], 1 - isdiv)]
        if 1 <= k - off <= NOP: parts.append(net.AND(k0d[k - off], isdiv))
        k0[k] = lsum(parts)
    # v3: a token the calculator has no symbol for, inside the call (e.g. '(' '%' '$' 'x'), rejects the call: never a silent misparse
    cs_at = {o: w.s(o, 'CS') for o in range(1, K)}
    anysym = lambda o: lsum(w.s(o, x) for x in SYMS)
    unk = net.clip01(lsum(net.AND(1 - net.clip01(lsum(cs_at[u] for u in range(1, o + 1))), net.clip01(lsum(cs_at[j] for j in range(o + 1, K))), 1 - anysym(o))
                          for o in range(1, K - 1)))
    valid = net.relu(net.AND(w.s(0, 'CE'), validA, validB, 1 - unk) - net.AND(isdiv, bzero) - (neg_out if nonneg else 0))   # v6 nonneg: silent on a negative result

    S = [valid, op['/'], neg_out] + [R[k] for k in range(1, KT + 1)] + [k0[k] for k in range(1, KU + 1)] \
        + [tz[k] for k in range(KU + 1, KT + 1)] + Areg + [t[u] for t in tB for u in range(9)] + [jB[j][i] for j in range(1, 10) for i in range(NOP + 1)]
    probes = {'A register (LSB first)': Areg, 'B register (LSB first)': Breg, 'sign A': [sA], 'sign B': [sB],
              'op one-hot [+,-,*,/]': [op[c] for c in '+-*/'], 'valid A': [validA], 'valid B': [validB], 'call valid': [valid],
              'magnitude op [add,sub,mul]': [use['+'], use['-'], use['*']], '|A|+|B| digits': S_add, '||A|-|B|| digits': S_sub,
              'A<B borrow': [neg], 'multiply column sums': cols_mul, 'multiply digits': S_mul,
              'result register R (MSB first)': [R[k] for k in range(1, KT + 1)], 'first digit index k0 (one-hot)': [k0[k] for k in range(1, KU + 1)],
              'result negative': [neg_out], 'first quotient digit index, division (one-hot)': [k0d[k] for k in range(1, NOP + 1)]}
    return net, S, probes


class CState:
    """Accessors for the copied call state inside stack 2."""

    def __init__(self, net, base):
        self.net = net; p = base
        self.valid, self.isdiv, self.neg = (net.inp(p + i) for i in range(3)); p += 3
        self.R = {k: net.inp(p + k - 1) for k in range(1, KT + 1)}; p += KT
        self.k0 = {k: net.inp(p + k - 1) for k in range(1, KU + 1)}; p += KU
        self.tz = {k: net.inp(p + k - KU - 1) for k in range(KU + 1, KT + 1)}; p += KT - KU
        self.A = [net.inp(p + i) for i in range(NOP)]; p += NOP
        self.tB = [[net.inp(p + 9 * i + u) for u in range(9)] for i in range(NOP)]; p += 9 * NOP
        self.jB = {j: [net.inp(p + (j - 1) * (NOP + 1) + i) for i in range(NOP + 1)] for j in range(1, 10)}; p += 9 * (NOP + 1)
        self.n = p - base


N_STATE = 3 + KT + KU + (KT - KU) + NOP + 9 * NOP + 9 * (NOP + 1)


def build_stack2(dotveto=False):
    net = Net(K * NS + N_STATE); w = Win(net); C = CState(net, K * NS)
    assert C.n == N_STATE
    PM = KT + 2                                                               # max chars of a result (point, sign)
    # ---- where are we in the output?  </tool_call> at offset k, <|box_start|> at k-1, s non-digit chars emitted ----
    eks = {}
    for k in range(1, min(PM + 2, K)):
        U = 3 * (w.s(k, 'CE') + w.s(k - 1, 'OS') - 2) + lsum(3 * w.dig(o) + 2 * (w.dot(o) + w.s(o, '-')) - 3 for o in range(k - 1))
        for s in range(3):
            if k - 1 - s >= 0:
                eks[k, s] = net.relu(U + s + 1) - 2 * net.relu(U + s) + net.relu(U + s - 1)
    act = lsum(eks.values()); p0 = eks[1, 0]
    td = {}
    for (k, s), e in eks.items(): td.setdefault(k - 1 - s, []).append(e)
    td = {n: lsum(v) for n, v in td.items()}                                  # one-hot: digits emitted so far
    hp = net.relu(lsum(s * e for (k, s), e in eks.items()) - C.neg)           # the point has been emitted
    kc = {}
    for k0, f in C.k0.items():
        for n, t in td.items():
            if k0 + n <= KT + 1: kc.setdefault(k0 + n, []).append(net.AND(f, t))
    kc = {k: lsum(v) for k, v in kc.items()}                                  # one-hot: index of the next digit
    kcg = lambda k: kc.get(k, Lin())
    frac = lsum(kcg(k) for k in range(KU + 1, KT + 2))

    # ---- + - * : the digits are in the call state ----
    digit_pm = lsum(net.sel(C.R[k], kcg(k)) for k in range(1, KT + 1))
    end_pm = lsum([net.AND(kcg(k), C.tz[k]) for k in range(KU + 1, KT + 1)] + [kcg(KT + 1)])

    # ---- / : one long-division step; the digits emitted so far are the quotient prefix P ----
    off = KU - NOP
    Pr = []
    for i in range(NOP):
        nd = net.relu(lsum(w.dig(o) for o in range(i + 1)) - i)
        da = net.relu(lsum(w.dig(o) for o in range(i + 2)) + 2 * lsum(w.dot(o) for o in range(i + 1)) - (i + 2))
        Pr.append(net.sel(w.dval(i), nd) + net.sel(w.dval(i + 1), da))
    Tdig = net.normalize(mul_cols(net, Pr, C.tB, NOP), NOP * 81)[:NOP]        # (P * B') mod 10^NOP
    Nk = []                                                                   # first kd digits of the numerator, LSB first
    for i in range(NOP + 1):
        Nk.append(lsum(net.sel(C.A[NOP - kd + i], kcg(kd + off)) for kd in range(1, NOP + D + 2) if 0 <= NOP - kd + i < NOP))
    X, _ = net.resolve_sub([Nk[i] - (Tdig[i - 1] if i >= 1 else 0) for i in range(NOP + 1)])   # 10*rem + next digit
    q = lsum(net.cmp_ge(X, C.jB[j]) for j in range(1, 10))
    B = C.jB[1][:NOP]
    end_div = lsum([net.AND(kcg(KU + D - s), 1 - net.cmp_ge([Lin()] * s + X, B)) for s in range(D)] + [kcg(KU + D + 1)])

    digit = digit_pm + net.sel(q, C.isdiv)
    endfrac = net.AND(1 - C.isdiv, end_pm) + net.AND(C.isdiv, end_div)
    c_end = net.AND(frac, endfrac, 1 - w.dot(0))
    c_dot = net.AND(kcg(KU + 1), 1 - hp, 1 - endfrac)
    c_sign = net.AND(p0, C.neg)
    g = net.AND(act, C.valid)
    # v6 dotveto: the model wrote '.' after a finished number (a sentence period): write no fraction digit, veto digits instead
    c_end2 = net.clip01(net.AND(frac, endfrac, w.dot(0)) + net.AND(w.dot(0), net.relu(hp - 1))) if dotveto else Lin()   # (a second point always ends it)
    out = {str(d): net.AND(g, 1 - c_end - c_end2 - c_dot - c_sign, net.eq(digit, d)) for d in range(10)}
    out['.'] = net.AND(g, c_dot); out['-'] = net.AND(g, c_sign)
    out['OE'] = net.AND(g, c_end + c_end2)                     # v4b: on a call it cannot answer the machine writes nothing at all (not even the end)
    out['OS'] = net.AND(w.s(0, 'CE'), net.clip01(lsum(w.s(o, 'CS') for o in range(1, K))), C.valid)   # v3b: silent on a call it cannot answer -> the model fills the box itself
    probes = {'active': [act], 'digits emitted (one-hot)': [td[n] for n in sorted(td)], 'next digit index kcur (one-hot)': [kcg(k) for k in range(1, KT + 2)],
              'point already emitted': [hp], 'quotient prefix P (LSB first)': Pr, 'P*B mod 10^n': Tdig, 'numerator prefix': Nk,
              '10*remainder + next digit (X)': X, 'quotient digit q': [q], 'chosen digit': [digit], 'fraction is finished': [endfrac],
              'emit </box>': [c_end], 'emit point': [c_dot], 'emit sign': [c_sign], 'output one-hot': [out[t] for t in OUT]}
    return net, [out[t] for t in OUT], probes


def build(spaced=False, nonneg=False, dotveto=False):
    n1, S, _ = build_stack1(spaced, nonneg); n2, O, _ = build_stack2(dotveto)
    assert len(S) == N_STATE
    return compile_net(n1, S), compile_net(n2, O)


def build_probes():
    """Named internal wires, for inspection: -> (compiled probe net, [(name, width)]) for each stack."""
    res = []
    for net, _, probes in (build_stack1(), build_stack2()):
        names = [(k, len(v)) for k, v in probes.items()]
        res.append((compile_net(net, [x if isinstance(x, Lin) else Lin(c=x) for v in probes.values() for x in v]), names))
    return res


if __name__ == '__main__':
    import time
    t = time.time(); c1, c2 = build()
    for name, c in (('stack1', c1), ('stack2', c2)):
        print(name, 'inputs', c['n_in'], 'units', c['n_units'], 'layers', len(c['layers']), 'widths', [len(b) for _, _, b in c['layers']],
              'dense params', sum(W.size for _, W, _ in c['layers']), 'nonzeros', sum(int((W != 0).sum()) for _, W, _ in c['layers']))
    print('window K', K, 'state', N_STATE, f'{time.time()-t:.1f}s')
