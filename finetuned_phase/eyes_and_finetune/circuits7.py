"""Walker micromachine (MM7) for BBH navigate: <walk>I;I;...;I</walk> -> box: Yes (back at the start) or No.
  I = F n | B n | L n | R n   step n (1-2 digits) forward / backward / left / right of the current facing
    | TL | TR | TA            turn left / right / around
Heading after each turn is kept mod 4; every step adds n to x or y in its absolute direction."""
from hdl import Net, Lin, lsum, compile_net
from seqmachine import Win, inside_flags, build_stack2_seq

SYMS = list('0123456789') + ['F', 'B', 'L', 'R', 'TL', 'TR', 'TA', ';', 'Yes', 'No', 'CS', 'CE', 'OS', 'OE']
NS = len(SYMS); SI = {s: i for i, s in enumerate(SYMS)}
K = 72
CONTENT = ['Yes', 'No']
OUT = CONTENT + ['OS', 'OE']
LMAX = 1
N_STATE = LMAX * len(CONTENT)
REL = {'F': 0, 'R': 1, 'B': 2, 'L': 3}


def walk_ref(call):
    import re
    items = call.split(';')
    if not call or any(not re.fullmatch(r'[FBLR]\d{1,2}|TL|TR|TA', it) for it in items): return ''
    if len(call) > 0 and sum(1 + (len(it) - 1 if it[0] in 'FBLR' else 0) for it in items) + len(items) - 1 > K - 4: return ''
    h, x, y = 0, 0, 0
    for it in items:
        if it in ('TL', 'TR', 'TA'): h = (h + {'TR': 1, 'TL': 3, 'TA': 2}[it]) % 4; continue
        d = (REL[it[0]] + h) % 4; n = int(it[1:])
        x += n if d == 1 else -n if d == 3 else 0; y += n if d == 0 else -n if d == 2 else 0
    return 'Yes' if x == 0 and y == 0 else 'No'


def build_stack1():
    net = Net(K * NS); w = Win(net, SI, NS, K)
    inside, cs_at = inside_flags(net, w, K)
    O = [o for o in inside]
    dig = lambda o: w.any(o, list('0123456789'))
    dval = lambda o: lsum(w.s(o, str(d)) * d for d in range(1, 10))
    end = lambda o: w.s(o, ';') + w.s(o, 'CE')                  # an item ends with ';' or the call end
    letter = {o: w.any(o, ['F', 'B', 'L', 'R']) for o in O}
    turn = {o: w.any(o, ['TL', 'TR', 'TA']) for o in O}
    # step size n at a letter: one digit then end, or two digits then end
    p1 = {o: net.AND(letter[o], inside[o], dig(o - 1), end(o - 2)) for o in O if o >= 2}
    p2 = {o: net.AND(letter[o], inside[o], dig(o - 1), dig(o - 2), end(o - 3)) for o in O if o >= 3}
    N = {o: (net.sel(dval(o - 1), p1[o]) if o in p1 else Lin()) + (net.sel(10 * dval(o - 1) + dval(o - 2), p2[o], M=99) if o in p2 else Lin()) for o in O}
    # heading before position o: (right turns + 3*left + 2*around) since the call start, mod 4
    hd = {}
    for o in O:
        c = lsum(net.AND(inside[u], w.s(u, 'TR')) + 3 * net.AND(inside[u], w.s(u, 'TL')) + 2 * net.AND(inside[u], w.s(u, 'TA')) for u in O if u > o)
        q = lsum(net.ge(c, 4 * m) for m in range(1, 3 * (K // 2) // 4 + 2))
        hmod = c - 4 * q
        hd[o] = {k: net.eq(hmod, k) for k in range(4)}
    X, Y = [], []
    for o in O:
        if o not in p1 and o not in p2: continue
        for a in range(4):
            fl = lsum(net.AND(w.s(o, L), hd[o][k], inside[o]) for L, rel in REL.items() for k in range(4) if (rel + k) % 4 == a)
            v = net.sel(N[o], fl, M=99)
            if a == 0: Y.append(v)
            if a == 2: Y.append(-1 * v)
            if a == 1: X.append(v)
            if a == 3: X.append(-1 * v)
    home = net.AND(net.eq(lsum(X), 0), net.eq(lsum(Y), 0))
    # shape: every inside token belongs to an item; at least one item
    n_items = lsum(p1.values()) + lsum(p2.values()) + lsum(net.AND(turn[o], inside[o], end(o - 1)) for o in O if o >= 1)
    n_tok = lsum(inside[o] for o in O)
    used = 2 * lsum(p1.values()) + 3 * lsum(p2.values()) + lsum(net.AND(turn[o], inside[o], end(o - 1)) for o in O if o >= 1) + n_items - 1
    ok = net.AND(w.s(0, 'CE'), net.ge(n_items, 1), net.eq(n_tok - used, 0))
    return net, [net.AND(ok, home), net.AND(ok, 1 - home)]


def build():
    n1, S = build_stack1(); n2, O = build_stack2_seq(SYMS, SI, NS, K, CONTENT, LMAX)
    assert len(S) == N_STATE
    return compile_net(n1, S), compile_net(n2, O)


if __name__ == '__main__':
    c1, c2 = build()
    for name, c in (('stack1', c1), ('stack2', c2)):
        print(name, 'units', c['n_units'], 'layers', len(c['layers']), 'nonzeros', sum(int((W != 0).sum()) for _, W, _ in c['layers']) + int((c['out'][1] != 0).sum()))
