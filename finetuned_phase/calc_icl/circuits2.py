"""Swap-tracker micromachine (MM2) circuits. Same shape as the calculator:
  stack 1  at a state-box end: read the digits of the box into a register
  (attention copies the register from the most recent box end)
  stack 2  after a call: apply 'new arrangement' / 'swap i j' / 'read i' and emit the new box, one token per step
"""
from hdl import Net, Lin, lsum, compile_net
from swapfmt import NMAX

SYMS = list('0123456789') + ['CS', 'CE', 'OS', 'OE']
NS = len(SYMS); SI = {s: i for i, s in enumerate(SYMS)}
K = NMAX + 5                      # window: offsets 0..K-1
OUT = list('0123456789') + ['OS', 'OE']
N_STATE = 1 + 2 * NMAX            # has-state flag, length one-hot, digits


class Win:
    def __init__(self, net): self.net = net
    def s(self, o, sym): return self.net.inp(o * NS + SI[sym]) if 0 <= o < K else Lin()
    def dig(self, o): return lsum(self.s(o, str(v)) for v in range(10))
    def dval(self, o): return lsum(self.s(o, str(v)) * v for v in range(1, 10))


def build_stack1():
    net = Net(K * NS); w = Win(net)
    nfl = {n: net.AND(w.s(0, 'OE'), w.s(n + 1, 'OS'), *[w.dig(o) for o in range(1, n + 1)]) for n in range(1, NMAX + 1)}
    P = {i: lsum(net.sel(w.dval(n + 1 - i), nfl[n]) for n in range(i, NMAX + 1)) for i in range(1, NMAX + 1)}
    S = [lsum(nfl.values())] + [nfl[n] for n in range(1, NMAX + 1)] + [P[i] for i in range(1, NMAX + 1)]
    return net, S, {'state length (one-hot)': [nfl[n] for n in range(1, NMAX + 1)], 'state digits': [P[i] for i in range(1, NMAX + 1)]}


def build_stack2():
    net = Net(K * NS + N_STATE); w = Win(net); c = lambda i: net.inp(K * NS + i)
    has = c(0); nold = {n: c(n) for n in range(1, NMAX + 1)}; P = {i: c(NMAX + i) for i in range(1, NMAX + 1)}
    # position in the output: call end at offset k, box start at k-1, t = k-1 digits emitted
    e = {k: net.AND(w.s(k, 'CE'), w.s(k - 1, 'OS'), *[w.dig(o) for o in range(k - 1)]) for k in range(1, NMAX + 2)}
    t = {k - 1: e[k] for k in e}; act = lsum(e.values())
    A1 = lsum(net.sel(w.dval(k + 1), e[k]) for k in e)                       # last digit of the call
    A2 = lsum(net.sel(w.dval(k + 2), e[k]) for k in e)                       # the digit before it (0 if absent)
    one = lsum(net.AND(e[k], w.dig(k + 1), w.s(k + 2, 'CS')) for k in e)     # call has one digit
    two = lsum(net.AND(e[k], w.dig(k + 1), w.dig(k + 2), w.s(k + 3, 'CS')) for k in e)
    j = {v: net.eq(A1, v) for v in range(1, NMAX + 1)}; i = {v: net.eq(A2, v) for v in range(1, NMAX + 1)}
    N = lsum(n * nold[n] for n in nold)
    in1 = net.AND(net.ge(A1, 1), net.ge(N - A1, 0)); in2 = net.AND(net.ge(A2, 1), net.ge(N - A2, 0))
    v_init = net.AND(one, lsum(j.values()))
    v_swap = net.AND(two, has, in1, in2, 1 - net.eq(A1 - A2, 0))
    v_read = net.AND(two, has, in1, net.eq(A2, 0))
    Pi = lsum(net.sel(P[q], i[q]) for q in P); Pj = lsum(net.sel(P[q], j[q]) for q in P)
    # is there a digit to emit at this step, and which
    more_init = lsum(net.AND(v_init, t[s], j[v]) for s in t for v in j if v > s)
    more_swap = lsum(net.AND(v_swap, t[s], nold[n]) for s in t for n in nold if n > s)
    more_read = net.AND(v_read, t[0])
    val = lsum([(s + 1) * net.AND(v_init, t[s], j[v]) for s in t for v in j if v > s]
               + [net.sel(P[p], v_swap, t[p - 1], 1 - i[p] - j[p], M=9) for p in P]
               + [net.sel(Pj, v_swap, t[p - 1], i[p]) for p in P] + [net.sel(Pi, v_swap, t[p - 1], j[p]) for p in P]
               + [net.sel(Pj, more_read)])
    emit = more_init + more_swap + more_read
    out = {str(d): net.AND(emit, net.eq(val, d)) for d in range(10)}
    out['OE'] = act - emit
    out['OS'] = net.AND(w.s(0, 'CE'), net.clip01(lsum(w.s(o, 'CS') for o in range(1, K))))   # v3: open a box only after this machine's own call
    probes = {'active': [act], 'digits emitted (one-hot)': [t[s] for s in sorted(t)], 'call digits [first, last]': [A2, A1],
              'kind [new, swap, read]': [v_init, v_swap, v_read], 'old state': [P[q] for q in P], 'chosen digit': [val], 'output one-hot': [out[x] for x in OUT]}
    return net, [out[x] for x in OUT], probes


def build():
    n1, S, _ = build_stack1(); n2, O, _ = build_stack2()
    assert len(S) == N_STATE
    return compile_net(n1, S), compile_net(n2, O)


if __name__ == '__main__':
    c1, c2 = build()
    for name, c in (('stack1', c1), ('stack2', c2)):
        print(name, 'inputs', c['n_in'], 'units', c['n_units'], 'layers', len(c['layers']), 'nonzeros', sum(int((W != 0).sum()) for _, W, _ in c['layers']) + int((c['out'][1] != 0).sum()))
