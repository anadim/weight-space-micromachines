"""calc_s4's stack 1 (micromachines/circuits/calc.py, build_stack1), copied unchanged except for one Team-C addition marked
below: a negative second operand written with its own sign token after a spaced operator ('3.5 * -2', '8.8 / -0.4').
Generated from the repo file by a text patch; diff against calc.py to audit."""
from micromachines.hdl import Net, Lin, lsum
from micromachines.circuits.calc import *            # noqa: F401,F403  (K, NS, SYMS, Win, parse_operand, therm, mul_cols, ...)
from micromachines.circuits.calc import K, NS, SYMS, OPLEN, NOP, KU, KT, NEGOP, Win, parse_operand, therm, mul_cols
from micromachines.calcfmt import NI, F, D


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
    # Team C: a negative B written with its own sign token after a spaced operator ('8.8 / -0.4' -> ... ' /' ' -' 0 . 4):
    # the '-' at k is B's sign and the operator sits at k+1. An operator token directly preceded by another operator is
    # therefore never the main operator.
    opneg = {k: net.relu(w.s(k, '-') + w.isop(k + 1) + lsum(w.dig(o) + w.dot(o) for o in range(1, k)) - k) for k in range(2, OPLEN + 2)} if spaced else {}
    opat = {k: net.AND(v, 1 - w.isop(k + 1)) for k, v in opat.items()} if spaced else opat
    op = {c: lsum(net.AND(opat[k], w.s(k, c) + w.s(k, NEGOP[c])) for k in opat) + lsum(net.AND(opsp[k], w.s(k + 1, c)) for k in opsp)
             + lsum(net.AND(opneg[k], w.s(k + 1, c)) for k in opneg) for c in '+-*/'}
    sB = lsum(net.AND(opat[k], w.negop(k)) for k in opat) + lsum(opneg.values())
    # ---- align operand A to the operator, then parse it (terminator: <tool_call>, or "-" then <tool_call>) ----
    al = lambda f, m: lsum(net.AND(opat[k], f(k + m)) for k in opat) + lsum(net.AND(opsp[k], f(k + 1 + m)) for k in opsp) + lsum(net.AND(opneg[k], f(k + 1 + m)) for k in opneg)
    a_dig = {m: al(w.dig, m) for m in rng}; a_dot = {m: al(w.dot, m) for m in rng}
    a_cs = {m: al(lambda o: w.s(o, 'CS'), m) for m in range(1, OPLEN + 3)}
    a_min = {m: al(lambda o: w.s(o, '-'), m) for m in rng}
    termS = {n: net.relu(a_min[n] + a_cs[n + 1] + lsum(a_dig[o] + a_dot[o] for o in range(1, n)) - n) for n in rng}
    sA = lsum(termS[n] for n in range(2, OPLEN + 2))
    Areg, validA = parse_operand(net, a_dig, {m: lsum(net.sel(w.dval(k + m), opat[k]) for k in opat) + lsum(net.sel(w.dval(k + 1 + m), opsp[k]) for k in opsp) + lsum(net.sel(w.dval(k + 1 + m), opneg[k]) for k in opneg) for m in rng},
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
        pos = [(opat[k], k) for k in opat if k <= NB + 1] + [(opsp[k], k + 1) for k in opsp if k <= NB + 1] + [(opneg[k], k + 1) for k in opneg if k <= NB + 1]
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


