"""Shape micromachine (MM11) for BBH geometric_shapes: <shape>SVG path d</shape> -> box: one code
   2 line, 3 triangle, 5..8 pentagon..octagon, K kite, T other quadrilateral (trapezoid), S sector (arcs and lines),
   E ellipse (arcs only).
Points are 'x,y' with two decimals. Vertices = distinct points. A quadrilateral is a kite when two pairs of adjacent
sides have equal length (within 4%), computed exactly on coordinates rounded to tenths."""
from hdl import Net, Lin, lsum, compile_net
from seqmachine import Win, inside_flags, build_stack2_seq

SYMS = list('0123456789') + ['.', ',', 'S_', 'M', 'Mx', 'L', 'A'] + list('KTSE') + ['CS', 'CE', 'OS', 'OE']   # S_ lone space; M first move; Mx ' M'; L ' L'; A ' A'
NS = len(SYMS); SI = {s: i for i, s in enumerate(SYMS)}
K = 232
NP = 18                     # points at most
CODES = ['2', '3', '5', '6', '7', '8', 'K', 'T', 'S', 'E']
CONTENT = CODES
OUT = CONTENT + ['OS', 'OE']
LMAX = 1
N_STATE = len(CONTENT)
TOL = 0.04


def shape_ref(d):
    import re
    if not re.fullmatch(r'M [\d.,]+(?: [MLA] [\d., ]+)*', d.strip()): return ''
    has_a = ' A ' in f' {d} '; has_l = ' L ' in f' {d} '
    if has_a: return 'S' if has_l else 'E'
    pts = re.findall(r'(\d{1,2}\.\d\d),(\d{1,2}\.\d\d)', d)
    if len(pts) != len(re.findall(r'[\d.]+,[\d.]+', d)) or not 2 <= len(pts) <= NP: return ''
    distinct = []
    for p in pts:
        if p not in distinct: distinct.append(p)
    n = len(distinct)
    if n in (2, 3, 5, 6, 7, 8): return str(n)
    if n != 4: return ''
    v = [(round(float(x) * 10), round(float(y) * 10)) for x, y in distinct]      # tenths (the circuit truncates; see below)
    v = [(int(x.split('.')[0]) * 10 + int(x.split('.')[1][0]), int(y.split('.')[0]) * 10 + int(y.split('.')[1][0])) for x, y in distinct]
    s = [(v[i][0] - v[(i + 1) % 4][0]) ** 2 + (v[i][1] - v[(i + 1) % 4][1]) ** 2 for i in range(4)]
    eqs = lambda a, b: abs(a - b) * 100 <= TOL * 100 * (a + b) / 2 * 1 and True
    eq = lambda a, b: 100 * abs(a - b) <= 2 * (a + b)                              # |a-b| <= 4% of the mean
    kite = (eq(s[0], s[1]) and eq(s[2], s[3])) or (eq(s[3], s[0]) and eq(s[1], s[2]))
    return 'K' if kite else 'T'


def build_stack1():
    net = Net(K * NS); w = Win(net, SI, NS, K)
    inside, cs_at = inside_flags(net, w, K)
    O = [o for o in inside]
    dig = lambda o: w.any(o, list('0123456789'))
    dv = lambda o: lsum(w.s(o, str(d)) * d for d in range(1, 10))
    nond = lambda o: 1 - dig(o)
    # a point: [i1] i2 . f1 f2 , [j1] j2 . g1 g2   anchored at the ',' (offset m); integer parts 1-2 digits
    pt, X, Y, Xt, Yt = {}, {}, {}, {}, {}
    for m in O:
        if m < 5 or m + 6 >= K: continue
        yb1 = net.AND(dig(m - 1), w.s(m - 2, '.'), dig(m - 3), dig(m - 4), nond(m - 5))            # y = d.dd
        yb2 = net.AND(dig(m - 1), dig(m - 2), w.s(m - 3, '.'), dig(m - 4), dig(m - 5), nond(m - 6))  # y = dd.dd
        xa1 = net.AND(dig(m + 1), dig(m + 2), w.s(m + 3, '.'), dig(m + 4), nond(m + 5))
        xa2 = net.AND(dig(m + 1), dig(m + 2), w.s(m + 3, '.'), dig(m + 4), dig(m + 5), nond(m + 6))
        pt[m] = net.AND(w.s(m, ','), inside[m], yb1 + yb2, xa1 + xa2)
        # values in tenths: x = (int part)*10 + first decimal
        xt = net.sel(10 * dv(m + 4) + dv(m + 2), net.AND(pt[m], xa1), M=999) + net.sel(100 * dv(m + 5) + 10 * dv(m + 4) + dv(m + 2), net.AND(pt[m], xa2), M=999)
        yt = net.sel(10 * dv(m - 1) + dv(m - 3), net.AND(pt[m], yb1), M=999) + net.sel(100 * dv(m - 1) + 10 * dv(m - 2) + dv(m - 4), net.AND(pt[m], yb2), M=999)
        X[m], Y[m] = xt, yt
        # hundredths digits, for exact equality of points
        Xt[m] = net.sel(dv(m + 1), pt[m]); Yt[m] = net.sel(net.sel(dv(m - 4), net.AND(pt[m], yb1)) + net.sel(dv(m - 5), net.AND(pt[m], yb2)), pt[m])
    M_ = list(pt)
    ordn = {m: lsum(pt[u] for u in M_ if u > m) for m in M_}
    npts = lsum(pt.values())
    Px, Py, Pxh, Pyh, ex = {}, {}, {}, {}, {}
    for p in range(NP):
        sel_p = {m: net.AND(pt[m], net.eq(ordn[m], p)) for m in M_}
        Px[p] = lsum(net.sel(X[m], sel_p[m], M=999) for m in M_); Py[p] = lsum(net.sel(Y[m], sel_p[m], M=999) for m in M_)
        Pxh[p] = lsum(net.sel(Xt[m], sel_p[m]) for m in M_); Pyh[p] = lsum(net.sel(Yt[m], sel_p[m]) for m in M_)
        ex[p] = net.ge(npts, p + 1)
    same = {}
    for p in range(NP):
        for q in range(p):
            same[q, p] = net.AND(net.eq(Px[p] - Px[q], 0), net.eq(Py[p] - Py[q], 0), net.eq(Pxh[p] - Pxh[q], 0), net.eq(Pyh[p] - Pyh[q], 0))
    dist = {p: net.relu(ex[p] - lsum(same[q, p] for q in range(p))) for p in range(NP)}
    nd = lsum(dist.values())
    # first four distinct points, in order: rank among distinct
    drank = {p: lsum(dist[q] for q in range(p)) for p in range(NP)}
    V = {}
    for i in range(4):
        f = {p: net.AND(dist[p], net.eq(drank[p], i)) for p in range(NP)}
        V[i] = (lsum(net.sel(Px[p], f[p], M=999) for p in range(NP)), lsum(net.sel(Py[p], f[p], M=999) for p in range(NP)))
    def sq(v):                                                   # v integer in [-999, 999]: v^2 = sum (2k-1)[|v| >= k]
        return lsum((2 * k - 1) * (net.ge(v, k) + net.ge(-v, k)) for k in range(1, 1000))
    S = [sq(V[i][0] - V[(i + 1) % 4][0]) + sq(V[i][1] - V[(i + 1) % 4][1]) for i in range(4)]
    eqs = lambda a, b: net.AND(net.ge(2 * (a + b) - 100 * (a - b), 0), net.ge(2 * (a + b) - 100 * (b - a), 0))
    kite = net.clip01(net.AND(eqs(S[0], S[1]), eqs(S[2], S[3])) + net.AND(eqs(S[3], S[0]), eqs(S[1], S[2])))
    has_a = net.clip01(lsum(net.AND(w.s(o, 'A'), inside[o]) for o in O)); has_l = net.clip01(lsum(net.AND(w.s(o, 'L'), inside[o]) for o in O))
    starts_m = lsum(net.AND(cs_at[j], w.s(j - 1, 'M')) for j in range(2, K))
    ok = net.AND(w.s(0, 'CE'), starts_m, 1 - net.ge(npts, NP + 1))
    poly = net.AND(ok, 1 - has_a)
    out = {'2': net.AND(poly, net.eq(nd, 2)), '3': net.AND(poly, net.eq(nd, 3)), '5': net.AND(poly, net.eq(nd, 5)), '6': net.AND(poly, net.eq(nd, 6)),
           '7': net.AND(poly, net.eq(nd, 7)), '8': net.AND(poly, net.eq(nd, 8)),
           'K': net.AND(poly, net.eq(nd, 4), kite), 'T': net.AND(poly, net.eq(nd, 4), 1 - kite),
           'S': net.AND(ok, has_a, has_l), 'E': net.AND(ok, has_a, 1 - has_l)}
    return net, [out[c] for c in CODES]


def build():
    n1, S = build_stack1(); n2, O = build_stack2_seq(SYMS, SI, NS, K, CONTENT, LMAX)
    assert len(S) == N_STATE
    return compile_net(n1, S), compile_net(n2, O)


if __name__ == '__main__':
    import time
    t = time.time(); c1, c2 = build()
    for name, c in (('stack1', c1), ('stack2', c2)):
        print(name, 'units', c['n_units'], 'layers', len(c['layers']), 'nonzeros', sum(int((W != 0).sum()) for _, W, _ in c['layers']) + int((c['out'][1] != 0).sum()))
    print(f'{time.time()-t:.1f}s')
