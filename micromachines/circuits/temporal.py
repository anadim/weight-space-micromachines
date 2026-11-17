"""Natural-format schedule machine for BBH temporal_sequences (fork C). Reads the question itself:
  '... woke up at 7am. ... saw X ... from 7am to 11am. ... was closed after 9pm. ... Options:\\n(A) 6pm to 9pm\\n(B) ...\\nA:'
and, when the model writes 'So the answer is (', writes the letter of the one option interval that no sighting covers
(after waking, before closing). Hours are digit tokens + 'am'/'pm'; every hour is 0..23 on a wire. A 24-slot busy mask is
filled from the sightings; an option is free iff none of its hours is busy. Exactly one free option, else silent."""
from ..hdl import Net, Lin, lsum, compile_net

LET = ['A', 'B', 'C', 'D']
SYMS = list('0123456789') + ['AM', 'PM', 'SP', 'FROM', 'TO', 'WOKE', 'CLOSED', 'RP', 'OA', 'OB', 'OC', 'OD', 'CE', 'TA', 'TI', 'PO', 'CS']
NS = len(SYMS); SI = {s: i for i, s in enumerate(SYMS)}
K = 212
OUT = LET
N_STATE = 1 + len(LET)
TEXT = {**{c: c for c in '0123456789'}, **{L: L for L in LET}, 'AM': 'am', 'PM': 'pm', 'SP': ' ', 'FROM': ' from', 'TO': ' to', 'WOKE': ' woke', 'CLOSED': ' closed',
        'RP': ')', 'OA': '(A', 'OB': '(B', 'OC': '(C', 'OD': '(D', 'CE': 'A', 'TA': ' answer', 'TI': ' is', 'PO': ' (', 'CS': ' Today'}


class Win:
    def __init__(self, net): self.net = net
    def s(self, o, sym): return self.net.inp(o * NS + SI[sym]) if 0 <= o < K else Lin()
    def D(self, o): return lsum(self.s(o, str(v)) for v in range(10))
    def val(self, o): return lsum(self.s(o, str(v)) * v for v in range(1, 10))


def time_ref(q):
    import re
    H = lambda s: (int(re.match(r'\d+', s).group()) % 12) + (12 if s.endswith('pm') else 0)
    sight = [(H(a), H(b)) for a, b in re.findall(r'from (\d+[ap]m) to (\d+[ap]m)', q)]
    wake = re.search(r'woke up at (\d+[ap]m)', q); close = re.search(r'closed after (\d+[ap]m)', q)
    free = []
    for L, a, b in re.findall(r'\(([A-D])\) (\d+[ap]m) to (\d+[ap]m)', q):
        a, b = H(a), H(b)
        if any(not (b <= s or a >= t) for s, t in sight): continue
        if wake and a < H(wake.group(1)): continue
        if close and b > H(close.group(1)): continue
        free.append(L)
    return free[0] if len(free) == 1 and sight else None


def build_stack1():
    net = Net(K * NS); w = Win(net); A = net.AND
    P = list(range(1, K - 8))
    # only the current question counts: everything after its ' Today' (a short question would otherwise let the window
    # reach the previous exemplar's sightings and options)
    csa = {o: w.s(o, 'CS') for o in range(1, K)}
    # inside[o] = no ' Today' at offsets 1..o and one further back: o lies in the latest question
    ins = {o: net.AND(1 - net.clip01(lsum(csa[u] for u in range(1, o + 1))), net.clip01(lsum(csa[u] for u in range(o + 1, K)))) for o in range(1, K)}
    has_cs = net.clip01(lsum(csa.values()))
    ap = {p: w.s(p, 'AM') + w.s(p, 'PM') for p in range(1, K)}
    two = {p: A(ap[p], w.D(p + 2)) for p in P}                                # two-digit hour
    one = {p: A(ap[p], 1 - w.D(p + 2)) for p in P}
    hour_ok = {p: A(ins[p], ap[p], w.D(p + 1), net.clip01(A(one[p], w.s(p + 2, 'SP')) + A(two[p], w.s(p + 3, 'SP')))) for p in P}
    n = {p: w.val(p + 1) + 10 * net.sel(w.val(p + 2), w.D(p + 2)) for p in P}           # the number before am/pm
    ng = {p: net.sel(n[p], hour_ok[p], M=99) for p in P}                             # 0 unless p ends a well-formed hour
    hv = {p: ng[p] - 12 * A(hour_ok[p], net.eq(ng[p], 12)) + 12 * A(hour_ok[p], w.s(p, 'PM')) for p in P}   # 0..23 (12am = 0, 12pm = 12)
    at = lambda p, d, sym: w.s(p + d, sym)                                            # helper
    # interval ending at the am/pm token p: '... X ' to' ' ' Y am/pm'; X's am/pm sits at q = p + 4 + two(p)
    ends = {}
    for p in P:
        for t in (0, 1):
            q = p + 4 + t
            if q >= K - 8: continue
            c_end = A(hour_ok[p], two[p] if t else one[p], at(p, 3 + t, 'TO'))
            for u in (0, 1):
                pre = q + 3 + u                                                   # the token before X's leading space
                c = A(c_end, hour_ok.get(q, Lin()), two.get(q, Lin()) if u else one.get(q, Lin()))
                ends.setdefault(p, []).append((c, q, pre))
    X, kind = {}, {}
    for p, cs in ends.items():
        X[p] = lsum(net.sel(hv[q], c, M=111) for c, q, pre in cs)
        kind[p, 'S'] = lsum(A(c, w.s(pre, 'FROM')) for c, q, pre in cs)
        for L in LET:
            kind[p, L] = lsum(A(c, w.s(pre, 'RP'), w.s(pre + 1, 'O' + L)) for c, q, pre in cs)
    inr = {(p, h): A(net.ge(h - X[p] + 1, 1), net.ge(hv[p] - h, 1)) for p in ends for h in range(24)}     # X <= h < Y
    # wake ' woke' ' up' ' at' ' ' d am  (woke at p+5+two) ; closed ' closed' ' after' ' ' d pm (closed at p+4+two)
    wk = {p: A(hour_ok[p], net.clip01(A(one[p], w.s(p + 5, 'WOKE')) + A(two[p], w.s(p + 6, 'WOKE')))) for p in P}
    cl = {p: A(hour_ok[p], net.clip01(A(one[p], w.s(p + 4, 'CLOSED')) + A(two[p], w.s(p + 5, 'CLOSED')))) for p in P}
    busy = {}
    for h in range(24):
        parts = [A(kind[p, 'S'], inr[p, h]) for p in ends]
        parts += [A(wk[p], net.ge(hv[p] - h, 1)) for p in P]                     # before waking
        parts += [A(cl[p], net.ge(h - hv[p] + 1, 1)) for p in P]                 # at or after closing
        busy[h] = net.clip01(lsum(parts))
    busy_in = {p: net.clip01(lsum(A(busy[h], inr[p, h]) for h in range(24))) for p in ends}
    free = {L: lsum(A(kind[p, L], 1 - busy_in[p]) for p in ends) for L in LET}
    nopt = {L: lsum(kind[p, L] for p in ends) for L in LET}
    nsight = lsum(kind[p, 'S'] for p in ends)
    ok = A(w.s(0, 'CE'), has_cs, net.ge(nsight, 1), net.eq(lsum(free.values()), 1), *[1 - net.ge(nopt[L], 2) for L in LET])
    return net, [ok] + [A(ok, net.clip01(free[L])) for L in LET]


def build_stack2():
    net = Net(K * NS + N_STATE); w = Win(net); A = net.AND
    ok = net.inp(K * NS); f = {L: net.inp(K * NS + 1 + i) for i, L in enumerate(LET)}
    e = A(w.s(0, 'PO'), w.s(1, 'TI'), w.s(2, 'TA'))                              # '... answer is (' -> the letter
    return net, [A(ok, e, f[L]) for L in LET]


def build():
    n1, S = build_stack1(); n2, O = build_stack2()
    assert len(S) == N_STATE
    return compile_net(n1, S), compile_net(n2, O)
