"""navnat (BBH navigate, official CoT): a coordinate state machine in natural text.

At the ':' that ends a step line '(k) <instruction>:' it reads the previous line's state '(X, Y), facing the <+|-> <x|y>-axis.'
and the instruction (Turn left/right/around | Always face forward | Take N step(s) [forward|backward|left|right]) and writes
the new state ' (X, Y), facing the <sign> <axis>' itself (exact signed integer arithmetic). v2: at 'Since' writes ' (X, Y)' (final position). Also writes ' not' / ' (' after
'Since (X, Y) is' and ' Yes' / ' No' after '... we are not|indeed where we started. So the answer is'.
Moves are relative to the facing direction (left = facing turned 90 degrees counter-clockwise).
"""
from ..hdl import Net, Lin, lsum, compile_net

TEXT = {**{c: c for c in '0123456789'}, 'OPEN': ' (', 'OPENM': ' (-', 'COMMA': ',', 'SP': ' ', 'MIS': ' -', 'CLOSE': '),',
        'FACING': ' facing', 'THE': ' the', 'POS': ' positive', 'NEG': ' negative', 'AXX': ' x', 'AXY': ' y', 'AXIS': '-axis',
        'NL': '.\n', 'LP': '(', 'RP': ')', 'TURN': ' Turn', 'LEFT': ' left', 'RIGHT': ' right', 'AROUND': ' around',
        'ALWAYS': ' Always', 'FACE': ' face', 'FORWARD': ' forward', 'BACKWARD': ' backward', 'TAKE': ' Take', 'STEP': ' step',
        'STEPS': ' steps', 'KEY': ':', 'SINCE': 'Since', 'IS': ' is', 'NOT': ' not', 'INDEED': ' indeed', 'ARE': ' are', 'WE': ' we',
        'WHERE': ' where', 'STARTED': ' started', 'DOT': '.', 'SO': ' So', 'ANSWER': ' answer',
        'W_YES': ' Yes', 'W_NO': ' No'}
SYMS = [k for k in TEXT if not k.startswith('W_')]
NS = len(SYMS); SI = {s: i for i, s in enumerate(SYMS)}
K = 32
NX = 4                       # digits of a coordinate
OUT = ['OPEN', 'OPENM'] + list('0123456789') + ['COMMA', 'SP', 'MIS', 'CLOSE', 'FACING', 'THE', 'POS', 'NEG', 'AXX', 'AXY', 'NOT', 'W_YES', 'W_NO', 'RP']
N_STATE = 1 + 2 * (1 + NX + NX) + 4
DIRS = 4                     # facing index: 0 +x (E), 1 +y (N), 2 -x (W), 3 -y (S); turning left = +1


class Win:
    def __init__(self, net): self.net = net
    def s(self, o, x): return self.net.inp(o * NS + SI[x]) if 0 <= o < K else Lin()
    def D(self, o): return lsum(self.s(o, str(v)) for v in range(10))
    def val(self, o): return lsum(self.s(o, str(v)) * v for v in range(1, 10))


def build_stack1():
    net = Net(K * NS); w = Win(net); A = net.AND; s = w.s
    # instruction shapes, offsets 1.. back from ':'  -> (length, checks, turn r one-hot dict, move r one-hot dict or None, N digit offsets)
    shapes = []
    for tk, r in (('LEFT', 1), ('AROUND', 2), ('RIGHT', 3)):
        shapes.append((2, [s(1, tk), s(2, 'TURN')], r, None, []))
    shapes.append((3, [s(1, 'FORWARD'), s(2, 'FACE'), s(3, 'ALWAYS')], 0, None, []))
    for nd in (1, 2):
        for tk, r in (('FORWARD', 0), ('BACKWARD', 2), ('LEFT', 1), ('RIGHT', 3)):
            shapes.append((4 + nd, [s(1, tk), s(2, 'STEP') + s(2, 'STEPS')] + [w.D(3 + j) for j in range(nd)] + [s(3 + nd, 'SP'), s(4 + nd, 'TAKE')], 0, r, [3 + j for j in range(nd)]))
        shapes.append((3 + nd, [s(1, 'STEP') + s(1, 'STEPS')] + [w.D(2 + j) for j in range(nd)] + [s(2 + nd, 'SP'), s(3 + nd, 'TAKE')], 0, 0, [2 + j for j in range(nd)]))
    combos = []
    for L, chk, rt, rm, noff in shapes:
        for nl in (1, 2):
            of = L + 8 + nl                                                      # offset of the previous line's ' facing'
            line = [s(L + 1, 'RP')] + [w.D(L + 2 + j) for j in range(nl)] + [s(L + 2 + nl, 'LP'), s(L + 3 + nl, 'NL'), s(L + 4 + nl, 'AXIS'),
                    s(of - 1, 'THE'), s(of, 'FACING'), s(of + 1, 'CLOSE')]
            for ny in range(1, 4):
                for nx in range(1, 4):
                    yb, xb = of + 2, of + 4 + ny
                    if xb + nx >= K: continue
                    coord = [w.D(yb + j) for j in range(ny)] + [s(yb + ny, 'SP') + s(yb + ny, 'MIS'), s(yb + ny + 1, 'COMMA')] + \
                            [w.D(xb + j) for j in range(nx)] + [s(xb + nx, 'OPEN') + s(xb + nx, 'OPENM')]
                    c = A(s(0, 'KEY'), *chk, *line, *coord)
                    combos.append(dict(c=c, of=of, yb=yb, ny=ny, xb=xb, nx=nx, rt=rt, rm=rm, noff=noff))
    valid = net.clip01(lsum(x['c'] for x in combos))
    sel = net.sel
    Xd = [lsum(sel(w.val(x['xb'] + p), x['c']) for x in combos if p < x['nx']) for p in range(3)]
    Yd = [lsum(sel(w.val(x['yb'] + p), x['c']) for x in combos if p < x['ny']) for p in range(3)]
    Xs = lsum(A(x['c'], s(x['xb'] + x['nx'], 'OPENM')) for x in combos)
    Ys = lsum(A(x['c'], s(x['yb'] + x['ny'], 'MIS')) for x in combos)
    F = [lsum(A(x['c'], s(x['of'] - 2, sg), s(x['of'] - 3, ax)) for x in combos) for sg, ax in (('POS', 'AXX'), ('POS', 'AXY'), ('NEG', 'AXX'), ('NEG', 'AXY'))]
    RT = [lsum(x['c'] for x in combos if x['rt'] == r) for r in range(4)]                       # turn by r (0: none)
    RM = [lsum(x['c'] for x in combos if x['rm'] == r) for r in range(4)]                       # move toward facing+r (none if no move)
    Nd = [lsum(sel(w.val(x['noff'][p]), x['c']) for x in combos if p < len(x['noff'])) for p in range(2)]      # noff[0] = units digit (offsets grow leftwards)
    F2 = [lsum(A(F[i], RT[r]) for i in range(4) for r in range(4) if (i + r) % 4 == j) for j in range(4)]
    MV = [lsum(A(F[i], RM[r]) for i in range(4) for r in range(4) if (i + r) % 4 == j) for j in range(4)]

    def signed_add(Vd, Vs, plus, minus):                                    # V + N*(plus - minus) -> (neg, digits LSB first, ndigits one-hot)
        Pc = [sel(Vd[p], 1 - Vs) + (sel(Nd[p], plus) if p < 2 else 0) for p in range(3)] + [Lin()]
        Mc = [sel(Vd[p], Vs) + (sel(Nd[p], minus) if p < 2 else 0) for p in range(3)] + [Lin()]
        Pd = net.resolve(Pc)[:NX]; Md = net.resolve(Mc)[:NX]
        D1, b1 = net.resolve_sub([x - y for x, y in zip(Pd, Md)]); D2, _ = net.resolve_sub([y - x for x, y in zip(Pd, Md)])
        R = [sel(x, 1 - b1) + sel(y, b1) for x, y in zip(D1, D2)]
        nz = [net.ge(r, 1) for r in R]
        Ln = {n: net.relu(nz[n - 1] - lsum(nz[n:])) for n in range(2, NX + 1)}; Ln[1] = 1 - lsum(Ln.values())
        return b1, R, [Ln[n] for n in range(1, NX + 1)]
    xn, XR, XL = signed_add(Xd, Xs, MV[0], MV[2])
    yn, YR, YL = signed_add(Yd, Ys, MV[1], MV[3])
    S = [valid, xn] + XR + XL + [yn] + YR + YL + F2
    assert len(S) == N_STATE
    return net, S


def build_stack2():
    net = Net(K * NS + N_STATE); w = Win(net); A = net.AND; s = w.s; b = K * NS
    I = lambda i: net.inp(b + i)
    valid, xn = I(0), I(1); XR = [I(2 + i) for i in range(NX)]; XL = [I(2 + NX + i) for i in range(NX)]
    o = 2 + 2 * NX; yn = I(o); YR = [I(o + 1 + i) for i in range(NX)]; YL = [I(o + 1 + NX + i) for i in range(NX)]
    F2 = [I(o + 1 + 2 * NX + i) for i in range(4)]
    out = {t: [] for t in OUT}; digit_parts, wd = [], []
    for j in range(0, 2 * NX + 8):
        at = A(s(j, 'KEY'), *[1 - s(u, 'KEY') - s(u, 'NL') for u in range(j)], valid)    # ':' j tokens back, nothing ended since
        for nx in range(1, NX + 1):
            for ny in range(1, NX + 1):
                g = A(at, XL[nx - 1], YL[ny - 1])
                if j == 0: out['OPENM'].append(A(g, xn)); out['OPEN'].append(A(g, 1 - xn))
                elif j <= nx: digit_parts.append(net.sel(XR[nx - j], g)); wd.append(g)
                elif j == nx + 1: out['COMMA'].append(g)
                elif j == nx + 2: out['MIS'].append(A(g, yn)); out['SP'].append(A(g, 1 - yn))
                elif j <= nx + 2 + ny: digit_parts.append(net.sel(YR[ny - (j - nx - 2)], g)); wd.append(g)
                elif j == nx + ny + 3: out['CLOSE'].append(g)
                elif j == nx + ny + 4: out['FACING'].append(g)
                elif j == nx + ny + 5: out['THE'].append(g)
                elif j == nx + ny + 6: out['POS'].append(A(g, F2[0] + F2[1])); out['NEG'].append(A(g, F2[2] + F2[3]))
                elif j == nx + ny + 7: out['AXX'].append(A(g, F2[0] + F2[2])); out['AXY'].append(A(g, F2[1] + F2[3]))
    # v2 (Qwen3 copies 'Since (0, 0)' from the prompt): at 'Since' the machine writes the final position itself,
    # ' (X, Y)' from the state of the last step line (the attention hop still holds the last ':' state)
    for j in range(0, 2 * NX + 4):
        at = A(s(j, 'SINCE'), *[1 - s(u, 'KEY') - s(u, 'NL') - s(u, 'SINCE') for u in range(j)], valid)
        for nx in range(1, NX + 1):
            for ny in range(1, NX + 1):
                g = A(at, XL[nx - 1], YL[ny - 1])
                if j == 0: out['OPENM'].append(A(g, xn)); out['OPEN'].append(A(g, 1 - xn))
                elif j <= nx: digit_parts.append(net.sel(XR[nx - j], g)); wd.append(g)
                elif j == nx + 1: out['COMMA'].append(g)
                elif j == nx + 2: out['MIS'].append(A(g, yn)); out['SP'].append(A(g, 1 - yn))
                elif j <= nx + 2 + ny: digit_parts.append(net.sel(YR[ny - (j - nx - 2)], g)); wd.append(g)
                elif j == nx + ny + 3: out['RP'].append(g)
    cur, wdig = lsum(digit_parts), lsum(wd)
    for d in range(10): out[str(d)].append(A(wdig, net.eq(cur, d)))
    # 'Since (X, Y) is' -> ' not' unless both are 0 (then ' (' to go on '(0, 0)')
    for ny in range(1, 4):
        for nx in range(1, 4):
            yb = 2; xb = 4 + ny
            parts = [s(0, 'IS'), s(1, 'RP')] + [w.D(yb + j) for j in range(ny)] + [s(yb + ny, 'SP') + s(yb + ny, 'MIS'), s(yb + ny + 1, 'COMMA')] + \
                    [w.D(xb + j) for j in range(nx)] + [s(xb + nx, 'OPEN') + s(xb + nx, 'OPENM'), s(xb + nx + 1, 'SINCE')]
            m = A(*parts)
            zero = A(m, s(2, '0'), s(xb, '0'), s(yb + ny, 'SP'), s(xb + nx, 'OPEN')) if nx == 1 and ny == 1 else Lin()
            out['NOT'].append(m - zero); out['OPEN'].append(zero)
    base = [s(0, 'IS'), s(1, 'ANSWER'), s(2, 'THE'), s(3, 'SO'), s(4, 'DOT'), s(5, 'STARTED'), s(6, 'WE'), s(7, 'WHERE')]
    out['W_NO'].append(A(*base, s(8, 'NOT'), s(9, 'ARE'), s(10, 'WE'))); out['W_YES'].append(A(*base, s(8, 'INDEED'), s(9, 'ARE'), s(10, 'WE')))
    return net, [net.clip01(lsum(out[t])) if out[t] else Lin() for t in OUT]


def build():
    n1, S = build_stack1(); n2, O = build_stack2()
    return compile_net(n1, S), compile_net(n2, O)
