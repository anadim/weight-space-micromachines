"""Calendar micromachine (MM10) for BBH date_understanding.
  <date>MM/DD/YYYY+N<u></date>  or  -N<u>    ->  box: MM/DD/YYYY
  u = D days (N <= 999), W weeks (N <= 99), M months (N <= 99; the day is clamped to the month's length),
      Y years (N <= 999; Feb 29 becomes Feb 28 in a non-leap year). Years 1000..2999, Gregorian calendar."""
import datetime
from hdl import Net, Lin, lsum, compile_net
from seqmachine import Win, inside_flags, build_stack2_seq

SYMS = list('0123456789') + ['/', '+', '-', 'D', 'W', 'M', 'Y', 'CS', 'CE', 'OS', 'OE']
NS = len(SYMS); SI = {s: i for i, s in enumerate(SYMS)}
K = 20
CONTENT = list('0123456789') + ['/']
OUT = CONTENT + ['OS', 'OE']
LMAX = 10
N_STATE = LMAX * len(CONTENT)
ML = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
CUM = [sum(ML[:m]) for m in range(12)]                        # days before month m+1 (non-leap)
YMIN, YMAX = 1000, 2999


def date_ref(call):
    import re, calendar
    m = re.fullmatch(r'(\d\d)/(\d\d)/(\d\d\d\d)([+-])(\d{1,3})([DWMY])', call)
    if not m: return ''
    M, D, Y, sg, N, u = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4), int(m.group(5)), m.group(6)
    if not (YMIN <= Y <= YMAX and 1 <= M <= 12): return ''
    if not 1 <= D <= calendar.monthrange(Y, M)[1]: return ''
    if u in 'WM' and N > 99: return ''
    s = 1 if sg == '+' else -1
    if u in 'DW':
        d = datetime.date(Y, M, D) + datetime.timedelta(days=s * N * (7 if u == 'W' else 1))
        Y2, M2, D2 = d.year, d.month, d.day
    elif u == 'M':
        tm = 12 * Y + M - 1 + s * N; Y2, M2 = tm // 12, tm % 12 + 1; D2 = min(D, calendar.monthrange(Y2, M2)[1]) if YMIN <= Y2 <= YMAX else D
    else:
        Y2, M2 = Y + s * N, M; D2 = 28 if (M == 2 and D == 29 and not calendar.isleap(Y2)) else D
    if not YMIN <= Y2 <= YMAX: return ''
    return f'{M2:02d}/{D2:02d}/{Y2:04d}'


def build_stack1():
    net = Net(K * NS); w = Win(net, SI, NS, K)
    dig = lambda o: w.any(o, list('0123456789'))
    dv = lambda o: lsum(w.s(o, str(d)) * d for d in range(1, 10))
    unit = {u: w.s(1, u) for u in 'DWMY'}
    # the shape for N with nd digits: CS M M / D D / Y Y Y Y sign N.. unit CE
    pat, vals = {}, {}
    for nd in (1, 2, 3):
        s = 2 + nd
        parts = [w.s(0, 'CE'), w.any(1, list('DWMY'))] + [dig(o) for o in range(2, 2 + nd)] + [w.any(s, ['+', '-'])]
        parts += [dig(s + 1), dig(s + 2), dig(s + 3), dig(s + 4), w.s(s + 5, '/'), dig(s + 6), dig(s + 7), w.s(s + 8, '/'), dig(s + 9), dig(s + 10), w.s(s + 11, 'CS')]
        pat[nd] = net.AND(*parts)
    gather = lambda f: lsum(net.sel(f(nd), pat[nd], M=999) for nd in (1, 2, 3))
    Nv = gather(lambda nd: lsum(dv(2 + i) * 10 ** i for i in range(nd)))
    plus = lsum(net.AND(pat[nd], w.s(2 + nd, '+')) for nd in (1, 2, 3)); minus = lsum(net.AND(pat[nd], w.s(2 + nd, '-')) for nd in (1, 2, 3))
    y4, y3, y2, y1 = (gather(lambda nd, k=k: dv(2 + nd + k)) for k in (1, 2, 3, 4))
    d2, d1 = gather(lambda nd: dv(2 + nd + 6)), gather(lambda nd: dv(2 + nd + 7))
    m2, m1 = gather(lambda nd: dv(2 + nd + 9)), gather(lambda nd: dv(2 + nd + 10))
    ok0 = lsum(pat.values())
    M = 10 * m1 + m2; D = 10 * d1 + d2; c = 10 * y1 + y2; yy = 10 * y3 + y4; Y = 100 * c + yy
    Mh = {m: net.AND(ok0, net.eq(M, m)) for m in range(1, 13)}
    # leap years: by the year's digits
    def mod4(v, top):
        q = lsum(net.ge(v, 4 * k) for k in range(1, top // 4 + 1)); return v - 4 * q
    r = mod4(yy, 99); rc = mod4(c, 29)
    def leap_of(j):                                            # is Y + j a leap year?  (|j| <= 12)
        t = yy + j
        wrap = net.ge(t, 100) - net.ge(-t, 1)                  # +1 / -1 when the century changes
        rj = {k: net.clip01(lsum(net.eq(r, (k - j) % 4 + 4 * i) for i in range(1) )) for k in range(4)}
        z4 = rj[0]
        yy0 = net.eq(t, 0) + net.eq(t, 100)
        cj0 = net.clip01(lsum(net.AND(net.eq(rc, a), net.eq(wrap, b)) for a in range(4) for b in (-1, 0, 1) if (a + b) % 4 == 0))
        return net.relu(z4 - net.AND(z4, yy0, 1 - cj0))
    L = {j: leap_of(j) for j in range(-12, 13)}
    leap = L[0]
    mlen = lambda Mh_, lp: lsum(ML[m - 1] * Mh_[m] for m in range(1, 13)) + net.AND(Mh_[2], lp) if isinstance(Mh_, dict) else None
    Dok = net.AND(ok0, net.ge(D, 1), net.ge(mlen(Mh, leap) - D, 0), net.ge(c, 10), 1 - net.ge(c, 30), net.clip01(lsum(Mh.values())))
    # ---------------- D / W: days
    uDW = unit['D'] + unit['W']
    delta = net.sel(Nv, unit['D'], M=999) + 7 * net.sel(Nv, net.AND(unit['W'], 1 - net.ge(Nv, 100)), M=999)
    doy = lsum((CUM[m - 1]) * Mh[m] for m in range(1, 13)) + lsum(net.AND(Mh[m], leap) for m in range(3, 13)) + D      # 1-based
    o = doy - 1 + net.sel(delta, plus, M=999 * 7) - net.sel(delta, minus, M=999 * 7)
    S = {0: Lin()}
    for k in range(1, 5): S[k] = S[k - 1] + 365 + L[k - 1]
    for k in range(-1, -5, -1): S[k] = S[k + 1] - 365 - L[k]
    kst = lsum(net.ge(o - S[k], 0) for k in range(-3, 5)) - 4          # Y' = Y + kst, kst in [-4, 4]
    kh = {k: net.eq(kst, k) for k in range(-4, 5)}
    doy2 = o + 1 - lsum(net.sel(S[k] + 1500, kh[k], M=3000) for k in range(-4, 5)) + 1500          # o + 1 - S[k*]
    leap2 = lsum(net.AND(kh[k], L[k]) for k in range(-4, 5))
    start = {m: CUM[m - 1] + (Lin() if m <= 2 else leap2) for m in range(1, 13)}
    Mn = 1 + lsum(net.ge(doy2 - start[m] - 1, 0) for m in range(2, 13))
    Mnh = {m: net.eq(Mn, m) for m in range(1, 13)}
    Dn = doy2 - lsum(net.sel(start[m], Mnh[m], M=400) for m in range(1, 13))
    Yd = Y + kst
    # ---------------- M: months
    NM = net.sel(Nv, net.AND(unit['M'], 1 - net.ge(Nv, 100)), M=999)
    tmr = M - 1 + net.sel(NM, plus, M=99) - net.sel(NM, minus, M=99) + 120                   # months since Jan of Y - 10
    q = lsum(net.ge(tmr - 12 * j, 0) for j in range(1, 21))                                  # Y' = Y - 10 + q
    jh = {j: net.eq(q - 10, j) for j in range(-10, 11)}
    Mm = tmr - 12 * q + 1
    Mmh = {m: net.eq(Mm, m) for m in range(1, 13)}
    leapm = lsum(net.AND(jh[j], L[j]) for j in range(-10, 11) if j in L)
    lenm = lsum(ML[m - 1] * Mmh[m] for m in range(1, 13)) + net.AND(Mmh[2], leapm)
    Dm = D - net.relu(D - lenm)
    Ym = Y + q - 10
    # ---------------- Y: years
    NY = net.sel(Nv, unit['Y'], M=999)
    Yy = Y + net.sel(NY, plus, M=999) - net.sel(NY, minus, M=999)
    # ---------------- choose by unit, then digits
    Mo = net.sel(Mn, uDW, M=12) + net.sel(Mm, unit['M'], M=12) + net.sel(M, unit['Y'], M=12)
    Do = net.sel(Dn, uDW, M=31) + net.sel(Dm, unit['M'], M=31) + net.sel(D, unit['Y'], M=31)
    Yo = net.sel(Yd, uDW, M=4000) + net.sel(Ym, unit['M'], M=4000) + net.sel(Yy, unit['Y'], M=4000)
    th = lsum(net.ge(Yo, 1000 * k) for k in range(1, 4)); r1 = Yo - 1000 * th
    hu = lsum(net.ge(r1, 100 * k) for k in range(1, 10)); r2 = r1 - 100 * hu
    te = lsum(net.ge(r2, 10 * k) for k in range(1, 10)); on = r2 - 10 * te
    # year digits are needed for the leap check of a Y shift: Feb 29 -> Feb 28 unless Y' is leap
    yyo = 10 * te + on; cco = 10 * th + hu
    ro = mod4(yyo, 99); rco = mod4(cco, 39)
    leapo = net.relu(net.eq(ro, 0) - net.AND(net.eq(ro, 0), net.eq(yyo, 0), 1 - net.eq(rco, 0)))
    feb29 = net.AND(unit['Y'], Mh[2], net.eq(D, 29), 1 - leapo)
    Do = Do - feb29
    mt = net.ge(Mo, 10); dt_ = lsum(net.ge(Do, 10 * k) for k in range(1, 4))
    digits = [mt, Mo - 10 * mt, None, dt_, Do - 10 * dt_, None, th, hu, te, on]
    unit_ok = net.AND(uDW + unit['M'] + unit['Y'], 1 - net.AND(unit['W'] + unit['M'], net.ge(Nv, 100)))
    ok = net.AND(Dok, unit_ok, net.ge(Yo, YMIN), 1 - net.ge(Yo, YMAX + 1), plus + minus)
    slots = []
    for t, dg in enumerate(digits):
        for v, sym in enumerate(CONTENT):
            if dg is None: slots.append(ok if sym == '/' else Lin())
            elif sym == '/': slots.append(Lin())
            else: slots.append(net.AND(ok, net.eq(dg, int(sym))))
    return net, slots


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
