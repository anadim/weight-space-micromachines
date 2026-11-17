"""Boolean-evaluator micromachine (MM3). Stack 1 runs at the call end: it checks the grammar and evaluates the
expression one nesting level at a time, deepest first. Stack 2 writes the box."""
from hdl import Net, Lin, lsum, compile_net
from boolfmt import LMAX, DMAX

WORDS = ['T', 'F', 'N', 'A', 'R', 'B', 'C']               # True False not and or ( )
SYMS = [w + v for w in WORDS for v in '01'] + ['CS', 'CE', 'OS', 'OE']    # each word with / without a leading space
NS = len(SYMS); SI = {s: i for i, s in enumerate(SYMS)}
K = LMAX + 2
OUT = ['T0', 'F0', 'OS', 'OE']
N_STATE = 2                                               # valid, value


def build_stack1():
    net = Net(K * NS); s = lambda o, x: net.inp(o * NS + SI[x]) if 0 <= o < K else Lin()
    raw = {w: {o: s(o, w + '0') + s(o, w + '1') for o in range(1, LMAX + 1)} for w in WORDS}
    anyw = {o: lsum(raw[w][o] for w in WORDS) for o in range(1, LMAX + 1)}
    E = {o: net.AND(*[anyw[u] for u in range(1, o + 1)]) for o in range(1, LMAX + 1)}     # offset o is inside the expression
    haslen = lsum(net.AND(s(n + 1, 'CS'), E[n]) for n in range(1, LMAX + 1))
    W = {w: {o: net.AND(raw[w][o], E[o]) for o in range(1, LMAX + 1)} for w in WORDS}
    T, F, N, A, R, B, C = (W[w] for w in WORDS)
    O = range(1, LMAX + 1)
    # ---- grammar ----
    S = {o: lsum(C[u] - B[u] for u in range(1, o + 1)) for o in O}                         # closes minus opens, from the right
    lev = {o: (S[o - 1] if o > 1 else Lin()) + C[o] for o in O}                            # nesting level; parentheses carry the level of their inside
    bad = [net.ge(-S[o], 1) for o in O] + [net.ge(lev[o], DMAX + 1) for o in O] + [net.ge(S[LMAX], 1)]
    for o in O:
        left_op = (N[o + 1] + A[o + 1] + R[o + 1] + B[o + 1] if o < LMAX else Lin()) + net.AND(E[o], s(o + 1, 'CS'))
        left_val = T[o + 1] + F[o + 1] + C[o + 1] if o < LMAX else Lin()
        bad.append(net.AND(left_op, A[o] + R[o] + C[o]))                                    # an operand must start here
        bad.append(net.AND(left_val, T[o] + F[o] + N[o] + B[o]))                            # an operator or ")" must come here
    bad.append(N[1] + A[1] + R[1] + B[1])                                                   # last word must end an operand
    valid = net.AND(s(0, 'CE'), haslen, 1 - net.clip01(lsum(bad)))
    # ---- evaluate, deepest level first ----
    L = {d: {o: net.eq(lev[o], d) for o in O} for d in range(DMAX + 2)}
    G = {}                                                                                  # G[d][r]: value of the group closed by ")" at r (level d)
    for d in range(DMAX, -1, -1):
        atom = {o: net.AND(T[o] + F[o], L[d][o]) + (net.AND(C[o], L[d + 1][o]) if d < DMAX else 0) for o in O}
        aval = {o: net.AND(T[o], L[d][o]) + (G[d + 1][o] if d < DMAX else 0) for o in O}
        notd = {o: net.AND(N[o], L[d][o]) for o in O}
        ord_ = {o: net.AND(R[o], L[d][o]) for o in O}
        andd = {o: net.AND(A[o], L[d][o]) for o in O}
        bnd = {o: net.AND(B[o] + C[o], L[d][o]) for o in O}
        block = {o: atom[o] + ord_[o] + andd[o] + bnd[o] for o in O}                        # level-d items that end a run of "not"
        f, term_ok = {}, {}
        for a in O:
            cnt = lsum(net.AND(notd[q], *[1 - block[u] for u in range(a + 1, q)]) for q in range(a + 1, LMAX + 1))
            odd = lsum(net.eq(cnt, k) for k in range(1, LMAX, 2))
            f[a] = net.AND(aval[a], 1 - odd) + net.AND(atom[a], 1 - aval[a], odd)           # value after the "not"s in front
        same = {}
        for lo in O:
            for hi in range(lo + 1, LMAX + 1):
                same[lo, hi] = net.AND(*[1 - ord_[u] - bnd[u] for u in range(lo + 1, hi)]) if hi > lo + 1 else Lin(c=1)
        for a in O:                                                                         # every factor of a's and-term is true
            false_in_term = lsum(net.AND(same[min(a, b), max(a, b)], atom[b], 1 - f[b]) for b in O if b != a)
            term_ok[a] = net.AND(f[a], 1 - net.clip01(false_in_term))
        if d == 0:
            value = net.clip01(lsum(term_ok.values()))
        else:
            G[d] = {}
            for r in O:
                inside = [net.AND(term_ok[a], net.AND(C[r], L[d][r]), *[1 - bnd[u] for u in range(r + 1, a)]) for a in range(r + 1, LMAX + 1)]
                G[d][r] = net.clip01(lsum(inside)) if inside else Lin()
    return net, [valid, net.AND(valid, value)]


def build_stack2():
    net = Net(K * NS + N_STATE); s = lambda o, x: net.inp(o * NS + SI[x])
    valid, value = net.inp(K * NS), net.inp(K * NS + 1)
    at_open = net.AND(s(0, 'OS'), s(1, 'CE'))
    after_val = net.AND(s(0, 'T0') + s(0, 'F0'), s(1, 'OS'), s(2, 'CE'))
    t = net.AND(at_open, value); fl = net.AND(at_open, valid, 1 - value)
    return net, [t, fl, s(0, 'CE'), at_open - t - fl + after_val]


def build():
    n1, S = build_stack1(); n2, O = build_stack2()
    return compile_net(n1, S), compile_net(n2, O)


if __name__ == '__main__':
    c1, c2 = build()
    for name, c in (('stack1', c1), ('stack2', c2)):
        print(name, 'inputs', c['n_in'], 'units', c['n_units'], 'layers', len(c['layers']), 'nonzeros', sum(int((W != 0).sum()) for _, W, _ in c['layers']) + int((c['out'][1] != 0).sum()))
