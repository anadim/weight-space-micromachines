"""Team B generic tools as hand-written ReLU circuits (micromachines/hdl.py), attached with micromachines/machine.py.

All three share one shape (the calculator pattern, scaled up to a whole argument list):
  * the host writes  <tool_call>NAME ARGS</tool_call>  (shared envelope of all teams, protocol rule 14);
  * stack 1 runs over a window of the last K tokens; at the '</tool_call>' token it parses ARGS (only the tokens after the
    most recent '<tool_call>') and computes the whole response as J token slots (one-hot over the tool's output tokens);
  * the attention hop copies that state from the most recent '</tool_call>' (key);
  * stack 2 (the shared emitter) writes slot j when the '</tool_call>' is j tokens back:
        <tool_response> RESULT </tool_response>     then goes silent.
  * a tool answers only calls whose first token after '<tool_call>' is its NAME (interval / walk / truth); for any other
    call it is silent (the host's logits are untouched);
  * every call to it is checked: if anything in ARGS does not parse (a token outside the tool's call alphabet, a number
    out of range, a count mismatch between items and separators), it writes the empty response
    '<tool_response></tool_response>' (the envelope's error).
The readers key only on the call grammar: digits, separators, and a few generic words (am/pm; turn/left/right/around/
forward/backward; says/true/false). No names, places, or task phrasing.

interval_gen : 'interval A-B busy C-D, E-F, ...' -> maximal free sub-intervals of [A, B) ('X-Y, Z-W' or 'none'), <= 4 gaps
turtle_gen   : 'walk turn left, forward 7, left 3, ...' -> '(X, Y) facing <north|west|south|east>' (start (0,0) north)
truth_gen    : 'truth 1 true; 2 says 1 false; ...' -> 'k true|false|unknown' for every mentioned k (k <= 12; same/opposite
               constraints used in both directions, any order, several per person; contradiction -> error)
Reference semantics: tools_ref.py.
"""
import os, sys
sys.path.insert(0, os.environ.get('WSM_REPO', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from micromachines.hdl import Net, Lin, lsum, compile_net
from micromachines.specs import SPECS

DIG = [str(d) for d in range(10)]
CALLTXT = {'CS': '<tool_call>', 'CE': '</tool_call>', 'OS': '<tool_response>', 'OE': '</tool_response>'}


class Win:
    def __init__(self, net, SI, K): self.net, self.SI, self.K, self.NS = net, SI, K, len(SI)
    def s(self, o, x): return self.net.inp(o * self.NS + self.SI[x]) if 0 <= o < self.K else Lin()
    def D(self, o): return lsum(self.s(o, d) for d in DIG)
    def val(self, o): return lsum(self.s(o, str(v)) * v for v in range(1, 10))


def call_scope(net, w, K):
    """inside[o] = offset o lies after the most recent '<tool_call>' (offset 0 = current token); has = a '<tool_call>' is in the window."""
    cs = {o: w.s(o, 'CS') for o in range(1, K)}
    inside = {o: 1 - net.clip01(lsum(cs[u] for u in range(1, o + 1))) for o in range(1, K)}
    return inside, net.clip01(lsum(cs.values()))


def n_inside(inside):
    """number of tokens between '<tool_call>' and the current token (NAME included)"""
    return lsum(inside.values())


def n_foreign(net, w, inside, K):
    """number of tokens inside the call that are not in the tool's call alphabet (every symbol flag zero)"""
    return lsum(net.AND(inside[o], 1 - lsum(net.inp(o * w.NS + i) for i in range(w.NS))) for o in range(1, K))


def serialize(net, fields, J, V):
    """fields: list of (length (integer Lin), toks: list over char c of {v: binary Lin}). -> slot[j][v] (binary Lin)."""
    slot = [[[] for _ in range(V)] for _ in range(J)]
    start = Lin()
    for L, toks in fields:
        if toks:
            E = [net.eq(start, j) for j in range(J)]
            for c, tv in enumerate(toks):
                for j in range(c, J):
                    for v, x in tv.items():
                        if isinstance(x, Lin) and not x.t and x.c == 0: continue
                        slot[j][v].append(net.AND(E[j - c], x))
        start = start + L
    return [[lsum(slot[j][v]) for v in range(V)] for j in range(J)]


def emitter(K, NS, ce_idx, J, V):
    """stack 2: write slot j when the key '</tool_call>' is j tokens back (and no later one)."""
    net = Net(K * NS + J * V)
    ce = lambda o: net.inp(o * NS + ce_idx)
    sl = lambda j, v: net.inp(K * NS + j * V + v)
    outs = [[] for _ in range(V)]
    for j in range(J):
        at = net.relu(ce(j) - lsum(ce(u) for u in range(j)))
        for v in range(V): outs[v].append(net.AND(at, sl(j, v)))
    return net, [net.clip01(lsum(x)) for x in outs]


def name_ok(net, w, inside, K, nxt):
    """the token right after the most recent '<tool_call>' is this tool's NAME, and the token after it is one of nxt
    (so 'truthful' or 'interval of ...' are not calls of truth / interval). nxt=None: no condition (walk: any
    'walk ...' call is ours, so 'walk up 3' gets the error response instead of silence, where the host would write its own)"""
    return net.clip01(lsum(net.AND(w.s(o + 1, 'CS'), w.s(o, 'NAME'), *([lsum(w.s(o - 1, x) for x in nxt)] if nxt else []), inside[o]) for o in range(1, K - 1)))


def finish(net, slots, ok, named, has, OI):
    """valid call -> the result slots; own NAME but invalid ARGS -> '<tool_response></tool_response>'; else silent."""
    good = net.AND(ok, named, has); err = net.AND(named, has, 1 - ok)
    out = []
    for j, row in enumerate(slots):
        for v, x in enumerate(row):
            e = err if (j == 0 and v == OI['W_OS']) or (j == 1 and v == OI['W_OE']) else None
            out.append(net.AND(x, good) + (e if e is not None else 0))
    return out


def const_tok(names, p, OI):
    return [{OI[n]: p} for n in names]


# ======================================================================================================================
# interval_gen
# ======================================================================================================================
I_SYMS = DIG + ['AM', 'PM', 'DASH', 'COMMA', 'SP', 'CS', 'CE', 'OS', 'OE', 'NAME', 'BUSY']
I_OUT = ['W' + d for d in DIG] + ['W_AM', 'W_PM', 'W_DASH', 'W_COMMA', 'W_SP', 'W_OS', 'W_OE', 'W_NONE']
I_TEXT = {**{d: d for d in DIG}, **{'W' + d: d for d in DIG}, 'NAME': 'interval', 'BUSY': ' busy', 'AM': 'am', 'PM': 'pm', 'DASH': '-', 'COMMA': ',', 'SP': ' ',
          **CALLTXT, 'W_AM': 'am', 'W_PM': 'pm', 'W_DASH': '-', 'W_COMMA': ',', 'W_SP': ' ', 'W_OS': '<tool_response>',
          'W_OE': '</tool_response>', 'W_NONE': 'none'}
I_K, I_J, I_H, I_G = 128, 40, 101, 4          # window, output slots, time slots 0..100, max gaps


def tok_ampm(h):
    if h in (0, 24): return ['1', '2', 'am']
    if h == 12: return ['1', '2', 'pm']
    return list(str(h % 12)) + ['pm' if h > 12 else 'am']


def interval_stack1():
    K = I_K; SI = {s: i for i, s in enumerate(I_SYMS)}; OI = {s: i for i, s in enumerate(I_OUT)}
    net = Net(K * len(I_SYMS)); w = Win(net, SI, K); A = net.AND; s = w.s; sel = net.sel
    inside, has = call_scope(net, w, K)
    U = lambda o: s(o, 'AM') + s(o, 'PM')
    valid, first_src, S, E, ampmL, ntok_iv = {}, {}, {}, {}, {}, []   # ntok_iv: tokens of each interval incl. '-' and the space before it
    for o in range(1, K - 1):
        dash = A(s(o, 'DASH'), inside[o])
        LA1 = A(U(o + 1), w.D(o + 2), 1 - w.D(o + 3)); LA2 = A(U(o + 1), w.D(o + 2), w.D(o + 3), 1 - w.D(o + 4))
        LP1 = A(w.D(o + 1), 1 - w.D(o + 2)); LP2 = A(w.D(o + 1), w.D(o + 2), 1 - w.D(o + 3))
        RA1 = A(w.D(o - 1), U(o - 2)); RA2 = A(w.D(o - 1), w.D(o - 2), U(o - 3))
        RP1 = A(w.D(o - 1), 1 - w.D(o - 2) - U(o - 2)); RP2 = A(w.D(o - 1), w.D(o - 2), 1 - w.D(o - 3) - U(o - 3))
        aL, aR = LA1 + LA2, RA1 + RA2
        match = A(aL, aR) + A(LP1 + LP2, RP1 + RP2)
        ok = A(dash, match)
        nL = sel(w.val(o + 2), LA1) + sel(w.val(o + 2) + 10 * w.val(o + 3), LA2, M=99) + sel(w.val(o + 1), LP1) + sel(w.val(o + 1) + 10 * w.val(o + 2), LP2, M=99)
        nR = sel(w.val(o - 1), RA1) + sel(10 * w.val(o - 1) + w.val(o - 2), RA2, M=99) + sel(w.val(o - 1), RP1) + sel(10 * w.val(o - 1) + w.val(o - 2), RP2, M=99)
        bad = A(aL, 1 - net.ge(nL, 1)) + A(aL, net.ge(nL, 13)) + A(aR, 1 - net.ge(nR, 1)) + A(aR, net.ge(nR, 13))
        v = A(ok, 1 - net.clip01(bad))
        pmL = A(LA1 + LA2, s(o + 1, 'PM')); pmR = A(RA1, s(o - 2, 'PM')) + A(RA2, s(o - 3, 'PM'))
        tL, tR = A(aL, net.eq(nL, 12)), A(aR, net.eq(nR, 12))
        Sv = nL - 12 * tL + 12 * pmL
        Ev = nR - 12 * tR + 12 * pmR + 24 * A(tR, 1 - pmR)
        valid[o] = v; S[o] = sel(Sv, v, M=99); E[o] = sel(Ev, v, M=99); ampmL[o] = aL
        ntok_iv.append(A(v, LA1) * 4 + A(v, LA2) * 5 + A(v, LP1) * 3 + A(v, LP2) * 4 + A(v, RA1) * 2 + A(v, RA2) * 3 + A(v, RP1) + A(v, RP2) * 2)
        first_src[o] = (dash, ok)
    os_ = list(valid)
    first = {o: A(valid[o], 1 - net.clip01(lsum(valid[u] for u in os_ if u > o))) for o in os_}
    busyf = {o: valid[o] - first[o] for o in os_}
    win, busy = [], []
    for h in range(I_H):
        wp, bp = [], []
        for o in os_:
            TS = net.relu(h - S[o] + 1) - net.relu(h - S[o])            # S <= h
            TE = net.relu(E[o] - h) - net.relu(E[o] - h - 1)            # h < E
            inr = net.relu(TS + TE - 1)
            wp.append(A(first[o], inr)); bp.append(A(busyf[o], inr))
        win.append(lsum(wp)); busy.append(net.clip01(lsum(bp)))
    free = [A(win[h], 1 - busy[h]) for h in range(I_H)] + [Lin()]
    start = [A(free[h], 1 - free[h - 1]) if h else free[0] for h in range(I_H)]
    endx = {h: A(free[h - 1], 1 - free[h]) for h in range(1, I_H + 1)}
    cnt, run = [], Lin()
    for h in range(I_H): run = run + start[h]; cnt.append(run)
    ngaps = cnt[-1]
    anyfirst = net.clip01(lsum(first.values()))
    ampm = lsum(A(first[o], ampmL[o]) for o in os_)
    ndash = lsum(A(s(o, 'DASH'), inside[o]) for o in range(1, K))
    nval = lsum(valid.values())
    ncomma = lsum(A(s(o, 'COMMA'), inside[o]) for o in range(1, K))
    nbusy = lsum(A(s(o, 'BUSY'), inside[o]) for o in range(1, K))
    cover = net.eq(n_inside(inside) - 1 - nbusy - ncomma - lsum(ntok_iv), 0)          # every token belongs to the grammar
    okc = A(has, anyfirst, 1 - net.ge(ngaps, I_G + 1), net.eq(ndash - nval, 0), net.eq(nval - ncomma, 2), net.eq(nbusy, 1), cover,
            1 - net.ge(n_foreign(net, w, inside, K), 1))

    def time_field(onehot):                      # onehot: {h: binary Lin}
        toks, L = [], Lin()
        for h, x in onehot.items():
            xa = A(x, ampm); xp = x - xa
            for xx, tk in ((xa, tok_ampm(h) if h <= 24 else None), (xp, list(str(h)))):
                if tk is None: continue
                L = L + xx * len(tk)
                for c, t in enumerate(tk):
                    while len(toks) <= c: toks.append({})
                    key = OI['W_AM'] if t == 'am' else OI['W_PM'] if t == 'pm' else OI['W' + t]
                    toks[c][key] = toks[c].get(key, Lin()) + xx
        return L, toks

    fields = [(Lin(c=1), const_tok(['W_OS'], Lin(c=1), OI))]
    for k in range(1, I_G + 1):
        pk = net.ge(ngaps, k)
        GS = {h: A(start[h], net.eq(cnt[h], k)) for h in range(I_H)}
        GE = {h: A(endx[h], net.eq(cnt[h - 1], k)) for h in range(1, I_H + 1)}
        fields.append(time_field(GS))
        fields.append((pk, const_tok(['W_DASH'], pk, OI)))
        fields.append(time_field(GE))
        if k < I_G:
            pn = net.ge(ngaps, k + 1); fields.append((2 * pn, const_tok(['W_COMMA', 'W_SP'], pn, OI)))
    none = 1 - net.ge(ngaps, 1)
    fields.append((none, const_tok(['W_NONE'], none, OI)))
    fields.append((Lin(c=1), const_tok(['W_OE'], Lin(c=1), OI)))
    return net, finish(net, serialize(net, fields, I_J, len(I_OUT)), okc, name_ok(net, w, inside, K, ['SP']), has, OI)


def build_interval():
    n1, S = interval_stack1()
    n2, O = emitter(I_K, len(I_SYMS), I_SYMS.index('CE'), I_J, len(I_OUT))
    return compile_net(n1, S), compile_net(n2, O)


# ======================================================================================================================
# turtle_gen
# ======================================================================================================================
T_SYMS = DIG + ['TURN', 'LEFT', 'RIGHT', 'AROUND', 'FWD', 'BWD', 'SP', 'COMMA', 'CS', 'CE', 'OS', 'OE', 'NAME']
T_OUT = ['W' + d for d in DIG] + ['W_LP', 'W_LPM', 'W_COMMA', 'W_SP', 'W_MIN', 'W_RP', 'W_FACING', 'W_N', 'W_W', 'W_S', 'W_E', 'W_OS', 'W_OE']
T_TEXT = {**{d: d for d in DIG}, **{'W' + d: d for d in DIG}, 'NAME': 'walk', 'TURN': ' turn', 'LEFT': ' left', 'RIGHT': ' right', 'AROUND': ' around',
          'FWD': ' forward', 'BWD': ' backward', 'SP': ' ', 'COMMA': ',', **CALLTXT, 'W_LP': '(', 'W_LPM': '(-', 'W_COMMA': ',',
          'W_SP': ' ', 'W_MIN': ' -', 'W_RP': ')', 'W_FACING': ' facing', 'W_N': ' north', 'W_W': ' west', 'W_S': ' south',
          'W_E': ' east', 'W_OS': '<tool_response>', 'W_OE': '</tool_response>'}
T_K, T_J, T_NC = 128, 18, 24                  # window, output slots, max commands (turn sum bound 3*T_NC)


def digits4(net, x):
    """x integer 0..9999 -> (digit wires [thousands, hundreds, tens, units], length one-hot {1..4})"""
    g = [net.ge(x, 10 ** i) for i in (1, 2, 3)]
    d3 = lsum(net.ge(x, 1000 * k) for k in range(1, 10)); r = x - 1000 * d3
    d2 = lsum(net.ge(r, 100 * k) for k in range(1, 10)); r = r - 100 * d2
    d1 = lsum(net.ge(r, 10 * k) for k in range(1, 10)); d0 = r - 10 * d1
    return [d3, d2, d1, d0], {1: 1 - g[0], 2: g[0] - g[1], 3: g[1] - g[2], 4: g[2]}


def number_field(net, x, OI):
    A = net.AND
    dg, Lh = digits4(net, x)
    oh = [{v: net.eq(d, v) for v in range(10)} for d in dg]
    toks = [{} for _ in range(4)]
    for n, ln in Lh.items():
        for c in range(n):
            src = oh[4 - n + c]
            for v in range(10):
                toks[c][OI['W%d' % v]] = toks[c].get(OI['W%d' % v], Lin()) + A(ln, src[v])
    return 1 + net.ge(x, 10) + net.ge(x, 100) + net.ge(x, 1000), toks


def turtle_stack1():
    K = T_K; SI = {s: i for i, s in enumerate(T_SYMS)}; OI = {s: i for i, s in enumerate(T_OUT)}
    net = Net(K * len(T_SYMS)); w = Win(net, SI, K); A = net.AND; s = w.s; sel = net.sel
    inside, has = call_scope(net, w, K)
    endc = lambda o: s(o, 'COMMA') + s(o, 'CE')
    tv, mv, ncmd, ntok = {}, {}, [], []
    for o in range(1, K - 1):
        tL = A(inside[o], s(o + 1, 'TURN'), s(o, 'LEFT'), endc(o - 1))
        tA = A(inside[o], s(o + 1, 'TURN'), s(o, 'AROUND'), endc(o - 1))
        tR = A(inside[o], s(o + 1, 'TURN'), s(o, 'RIGHT'), endc(o - 1))
        tv[o] = tL + 2 * tA + 3 * tR; ncmd += [tL, tA, tR]; ntok += [2 * (tL + tA + tR)]
        for r, word in enumerate(['FWD', 'LEFT', 'BWD', 'RIGHT']):
            one = A(inside[o], s(o, word), 1 - s(o + 1, 'TURN'), s(o - 1, 'SP'), w.D(o - 2), endc(o - 3))
            two = A(inside[o], s(o, word), 1 - s(o + 1, 'TURN'), s(o - 1, 'SP'), w.D(o - 2), w.D(o - 3), endc(o - 4))
            mv[o, r] = sel(w.val(o - 2), one) + sel(10 * w.val(o - 2) + w.val(o - 3), two, M=99)
            ncmd += [one, two]; ntok += [3 * one, 4 * two]
    HT = 3 * T_NC
    amt = [[] for _ in range(4)]
    for o in range(1, K - 1):
        H = lsum(tv[u] for u in tv if u > o)
        hd = [lsum(net.eq(H, 4 * m + k) for m in range(HT // 4 + 1)) for k in range(4)]
        for r in range(4):
            for k in range(4):
                amt[(k + r) % 4].append(sel(mv[o, r], hd[k], M=99))
    Hf = lsum(tv.values())
    hf = [lsum(net.eq(Hf, 4 * m + k) for m in range(HT // 4 + 1)) for k in range(4)]
    N_, W_, S_, E_ = [lsum(a) for a in amt]
    X, Y = E_ - W_, N_ - S_
    nx, ny = net.ge(-X, 1), net.ge(-Y, 1)
    ax, ay = net.relu(X) + net.relu(-X), net.relu(Y) + net.relu(-Y)
    ncomma = lsum(A(s(o, 'COMMA'), inside[o]) for o in range(1, K))
    cover = net.eq(n_inside(inside) - 1 - ncomma - lsum(ntok), 0)                     # every token belongs to a command
    okc = A(has, net.eq(lsum(ncmd) - ncomma, 1), 1 - net.ge(lsum(ncmd), T_NC + 1), cover, 1 - net.ge(n_foreign(net, w, inside, K), 1))
    one = Lin(c=1)
    fields = [(one, const_tok(['W_OS'], one, OI)),
              (one, [{OI['W_LP']: 1 - nx, OI['W_LPM']: nx}]),
              number_field(net, ax, OI),
              (one, const_tok(['W_COMMA'], one, OI)),
              (one, [{OI['W_SP']: 1 - ny, OI['W_MIN']: ny}]),
              number_field(net, ay, OI),
              (one, const_tok(['W_RP'], one, OI)), (one, const_tok(['W_FACING'], one, OI)),
              (one, [{OI['W_N']: hf[0], OI['W_W']: hf[1], OI['W_S']: hf[2], OI['W_E']: hf[3]}]),
              (one, const_tok(['W_OE'], one, OI))]
    return net, finish(net, serialize(net, fields, T_J, len(T_OUT)), okc, name_ok(net, w, inside, K, None), has, OI)


def build_turtle():
    n1, S = turtle_stack1()
    n2, O = emitter(T_K, len(T_SYMS), T_SYMS.index('CE'), T_J, len(T_OUT))
    return compile_net(n1, S), compile_net(n2, O)


# ======================================================================================================================
# truth_gen
# ======================================================================================================================
R_SYMS = DIG + ['TRUE', 'FALSE', 'SAYS', 'SEMI', 'SP', 'CS', 'CE', 'OS', 'OE', 'NAME']
R_OUT = ['W' + d for d in DIG] + ['W_TRUE', 'W_FALSE', 'W_UNK', 'W_COMMA', 'W_SP', 'W_OS', 'W_OE']
R_TEXT = {**{d: d for d in DIG}, **{'W' + d: d for d in DIG}, 'TRUE': ' true', 'FALSE': ' false', 'SAYS': ' says', 'SEMI': ';',
          'SP': ' ', **CALLTXT, 'NAME': 'truth', 'W_TRUE': ' true', 'W_FALSE': ' false', 'W_UNK': ' unknown', 'W_COMMA': ',', 'W_SP': ' ',
          'W_OS': '<tool_response>', 'W_OE': '</tool_response>'}
R_K, R_J, R_NV = 128, 62, 12


def truth_stack1():
    K = R_K; SI = {s: i for i, s in enumerate(R_SYMS)}; OI = {s: i for i, s in enumerate(R_OUT)}
    net = Net(K * len(R_SYMS)); w = Win(net, SI, K); A = net.AND; s = w.s; sel = net.sel
    inside, has = call_scope(net, w, K)
    cstart = lambda o: A(s(o, 'SP'), s(o + 1, 'SEMI') + s(o + 1, 'NAME'))
    num = lambda o, n: w.val(o) if n == 1 else 10 * w.val(o + 1) + w.val(o)      # digits at o..o+n-1, MSB at the larger offset
    digs = lambda o, n: [w.D(o + i) for i in range(n)]
    cl, subj, ref, isc, iss, bad, ntok = {}, {}, {}, {}, {}, [], []
    for o in range(1, K - 1):
        word = A(inside[o], s(o, 'TRUE') + s(o, 'FALSE'), s(o - 1, 'SEMI') + s(o - 1, 'CE'))
        pats = []
        for ns in (1, 2):
            pats.append(('c', A(word, *digs(o + 1, ns), cstart(o + 1 + ns)), num(o + 1, ns), Lin(), ns + 2))
            for nr in (1, 2):
                b = o + nr + 3
                pats.append(('s', A(word, *digs(o + 1, nr), s(o + nr + 1, 'SP'), s(o + nr + 2, 'SAYS'), *digs(b, ns), cstart(b + ns)),
                             num(b, ns), num(o + 1, nr), ns + nr + 4))
        cl[o] = lsum(p[1] for p in pats)
        subj[o] = lsum(sel(v, f, M=99) for _, f, v, _, _ in pats)
        ref[o] = lsum(sel(r, f, M=99) for k, f, _, r, _ in pats if k == 's')
        isc[o] = lsum(f for k, f, _, _, _ in pats if k == 'c'); iss[o] = cl[o] - isc[o]
        ntok += [f * L for _, f, _, _, L in pats]
        bad += [A(cl[o], 1 - net.ge(subj[o], 1)), A(cl[o], net.ge(subj[o], R_NV + 1)), A(iss[o], 1 - net.ge(ref[o], 1)), A(iss[o], net.ge(ref[o], R_NV + 1))]
    os_ = list(cl); NV = range(1, R_NV + 1)
    se = {(o, k): net.eq(subj[o], k) for o in os_ for k in NV}
    re_ = {(o, j): net.eq(ref[o], j) for o in os_ for j in NV}
    ment = {k: net.clip01(lsum(se[o, k] + re_[o, k] for o in os_)) for k in NV}          # mentioned anywhere
    fT = {k: net.clip01(lsum(A(se[o, k], isc[o], s(o, 'TRUE')) for o in os_)) for k in NV}
    fF = {k: net.clip01(lsum(A(se[o, k], isc[o], s(o, 'FALSE')) for o in os_)) for k in NV}
    PT = {(k, j): lsum(A(se[o, k], re_[o, j], s(o, 'TRUE')) for o in os_) for k in NV for j in NV}
    PF = {(k, j): lsum(A(se[o, k], re_[o, j], s(o, 'FALSE')) for o in os_) for k in NV for j in NV}
    ET = {(k, j): net.clip01(PT[k, j] + PT[j, k]) for k in NV for j in NV}               # 'same' edge, either direction
    EF = {(k, j): net.clip01(PF[k, j] + PF[j, k]) for k in NV for j in NV}               # 'opposite' edge
    known = {k: net.clip01(fT[k] + fF[k]) for k in NV}; val = dict(fT)
    for _ in range(R_NV):
        nk, nv = {}, {}
        for k in NV:
            nk[k] = net.clip01(fT[k] + fF[k] + lsum(A(ET[k, j], known[j]) + A(EF[k, j], known[j]) for j in NV))
            nv[k] = net.clip01(fT[k] + lsum(A(ET[k, j], val[j]) + A(EF[k, j], known[j] - val[j]) for j in NV))
        known, val = nk, nv
    xor = lambda a, b: a + b - 2 * A(a, b)
    conflict = [A(fT[k], fF[k]) + A(fF[k], val[k]) for k in NV]
    conflict += [A(ET[k, j], known[k], known[j], xor(val[k], val[j])) + A(EF[k, j], known[k], known[j], 1 - xor(val[k], val[j]))
                 for k in NV for j in NV if j >= k]
    nsemi = lsum(A(s(o, 'SEMI'), inside[o]) for o in range(1, K))
    cover = net.eq(n_inside(inside) - 1 - nsemi - lsum(ntok), 0)                         # every token belongs to a clause
    okc = A(has, net.eq(lsum(cl.values()) - nsemi, 1), 1 - net.clip01(lsum(bad)), 1 - net.clip01(lsum(conflict)), cover,
            1 - net.ge(n_foreign(net, w, inside, K), 1))
    fields = [(Lin(c=1), const_tok(['W_OS'], Lin(c=1), OI))]
    for k in NV:
        p = ment[k]
        fields.append((p * len(str(k)), const_tok(['W' + c for c in str(k)], p, OI)))
        fields.append((p, [{OI['W_TRUE']: A(p, val[k]), OI['W_FALSE']: A(p, known[k], 1 - val[k]), OI['W_UNK']: A(p, 1 - known[k])}]))
        if k < R_NV:
            later = net.clip01(lsum(ment[u] for u in range(k + 1, R_NV + 1)))
            q = A(p, later); fields.append((2 * q, const_tok(['W_COMMA', 'W_SP'], q, OI)))
    fields.append((Lin(c=1), const_tok(['W_OE'], Lin(c=1), OI)))
    return net, finish(net, serialize(net, fields, R_J, len(R_OUT)), okc, name_ok(net, w, inside, K, ['SP']), has, OI)


def build_truth():
    n1, S = truth_stack1()
    n2, O = emitter(R_K, len(R_SYMS), R_SYMS.index('CE'), R_J, len(R_OUT))
    return compile_net(n1, S), compile_net(n2, O)


# ---------------- registration (same spec format as micromachines/specs.py; nothing there is changed) ----------------
SPECS['interval_gen'] = dict(build=build_interval, syms=I_SYMS, out=I_OUT, key='CE', K=I_K, W=I_J + 2, text=I_TEXT)
SPECS['turtle_gen'] = dict(build=build_turtle, syms=T_SYMS, out=T_OUT, key='CE', K=T_K, W=T_J + 2, text=T_TEXT)
SPECS['truth_gen'] = dict(build=build_truth, syms=R_SYMS, out=R_OUT, key='CE', K=R_K, W=R_J + 2, text=R_TEXT)
TOOLS_B = ('interval_gen', 'turtle_gen', 'truth_gen')
