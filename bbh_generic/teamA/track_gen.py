"""track_gen: a generic STATE tracker compiled into a ReLU network (Team A, BBH redo).

f(x): given an initial array of up to 9 one-digit values and a list of operations (swap two slots, set a slot to a value,
copy one slot into another), return the final value of a slot, the slot holding a value, or the whole final array.
Call / semantics: see callfmt.py (track S:OPS[Q]); the circuit below implements exactly callfmt.track_ref.

It reads only call-syntax tokens (digits, ':' ',' '=' '<' '?' '@', the merged tokens ':?' ':@', the word 'track', a space,
and the call/response tokens). It knows nothing about names, objects or any task wording: the host model writes the
numbers. stack 1 runs at </tool_call>:
  1. frame: the most recent <tool_call> (offset c), 'track' + space after it;
  2. the initial values: the digit run starting at c-3, ended by a ':'-class token;
  3. the operations 'ij' / 'i=v' / 'i<j' (each preceded by ':' or ',' and followed by ',', a query mark or </tool_call>),
     ranked in text order and gathered into TRACK_MAXOPS registers;
  4. the operations applied one after another to 9 value registers (integer wires 0..9);
  5. the query, the validity checks (every token of the call accounted for, slots in range), and the reply digits.
"""
from circ_common import Net, Lin, lsum, compile_net, Win, call_frame, build_stack2_seq, DIGITS
from callfmt import TRACK_NMAX as NMAX, TRACK_MAXOPS as MAXOPS

SYMS = DIGITS + [':', ',', '=', '<', '?', '@', ':?', ':@', 'NAME', 'SP', 'CS', 'CE', 'OS', 'OE']
NS = len(SYMS); SI = {s: i for i, s in enumerate(SYMS)}
K = 4 + NMAX + 4 * MAXOPS + 3                 # window: <tool_call> track ' ' S : ops(<=4 tokens each with comma) ?q </tool_call>
CONTENT = DIGITS
LMAX = NMAX
OUT = CONTENT + ['OS', 'OE']
N_STATE = LMAX * len(CONTENT) + 1           # + 'mine'
TEXT = {**{d: d for d in DIGITS}, ':': ':', ',': ',', '=': '=', '<': '<', '?': '?', '@': '@', ':?': ':?', ':@': ':@',
        'NAME': 'track', 'SP': ' ', 'CS': '<tool_call>', 'CE': '</tool_call>', 'OS': '<tool_response>', 'OE': '</tool_response>'}

COLON = (':', ':?', ':@')          # ends the initial values
QMARK = ('?', ':?')                # value query
AMARK = ('@', ':@')                # inverse query
SEP = (':', ',')                   # before an operation
AFTER = (',', '?', '@', 'CE')      # after an operation


def build_stack1(probes=False):
    net = Net(K * NS); w = Win(net, SI, NS, K); A = net.AND
    csl, inside, found, nameok = call_frame(net, w, 'NAME')
    # ---- initial values: digits at offsets c-3, c-4, ... (slot k at c-2-k)
    dd = {k: lsum(A(csl[c], w.dig(c - 2 - k)) for c in range(3, K) if c - 2 - k >= 1) for k in range(1, NMAX + 2)}
    nge = {1: dd[1]}
    for k in range(2, NMAX + 2): nge[k] = A(nge[k - 1], dd[k])
    n = lsum(nge[k] for k in range(1, NMAX + 1))                       # length of the digit run (capped at NMAX)
    too_long = nge[NMAX + 1]
    neq = {m: nge[m] - (nge[m + 1] if m + 1 in nge else 0) for m in range(1, NMAX + 1)}   # one-hot of n
    colon_ok = lsum(A(csl[c], neq[m], w.s(c - 3 - m, COLON)) for c in range(3, K) for m in range(1, NMAX + 1) if c - 3 - m >= 1)
    V = [None] + [net.sel(lsum(net.sel(w.dval(c - 2 - k), csl[c]) for c in range(3, K) if c - 2 - k >= 1), nge[k]) for k in range(1, NMAX + 1)]
    # ---- operations ending at offset o (last token at o)
    aft = {o: w.s(o - 1, AFTER) for o in range(1, K)}
    sw = {o: A(w.dig(o), w.dig(o + 1), w.s(o + 2, SEP), inside[o + 2], aft[o]) for o in range(1, K - 2)}
    asg = {o: A(w.dig(o), w.s(o + 1, '='), w.dig(o + 2), w.s(o + 3, SEP), inside[o + 3], aft[o]) for o in range(1, K - 3)}
    cpy = {o: A(w.dig(o), w.s(o + 1, '<'), w.dig(o + 2), w.s(o + 3, SEP), inside[o + 3], aft[o]) for o in range(1, K - 3)}
    opend = {o: sw.get(o, Lin()) + asg.get(o, Lin()) + cpy.get(o, Lin()) for o in range(1, K)}
    oplen = lsum(2 * sw[o] for o in sw) + lsum(3 * asg[o] + 3 * cpy[o] for o in asg)
    cnt, run = {}, Lin()
    for o in range(K - 1, 0, -1):                                       # earlier text = larger offset
        cnt[o] = run; run = run + opend[o]
    nops = run
    ops = []
    for r in range(MAXOPS):
        g = {o: A(opend[o], net.eq(cnt[o], r)) for o in range(1, K)}
        ts = lsum(A(g[o], sw[o]) for o in sw); ta = lsum(A(g[o], asg[o]) for o in asg); tc = lsum(A(g[o], cpy[o]) for o in cpy)
        X = lsum(net.sel(w.dval(o + 1), A(g[o], sw[o])) for o in sw) + lsum(net.sel(w.dval(o + 2), A(g[o], asg[o] + cpy[o])) for o in asg)
        Y = lsum(net.sel(w.dval(o), g[o]) for o in g)
        ops.append((ts, ta, tc, X, Y))
    # ---- apply
    bad = Lin()
    for ts, ta, tc, X, Y in ops:
        Xh = {p: net.eq(X, p) for p in range(1, NMAX + 1)}; Yh = {p: net.eq(Y, p) for p in range(1, NMAX + 1)}
        okX = A(net.ge(X, 1), net.ge(n - X, 0)); okY = A(net.ge(Y, 1), net.ge(n - Y, 0)); dif = 1 - net.eq(X - Y, 0)
        bad = bad + (ta - A(ta, okX)) + (ts + tc - A(ts + tc, okX, okY, dif))
        Is = {p: A(ts, Xh[p]) for p in Xh}; Js = {p: A(ts, Yh[p]) for p in Yh}
        Ia = {p: A(ta, Xh[p]) for p in Xh}; Ic = {p: A(tc, Xh[p]) for p in Xh}; Jc = {p: A(tc, Yh[p]) for p in Yh}
        NV = [None]
        for p in range(1, NMAX + 1):
            v = V[p] - net.sel(V[p], Is[p]) - net.sel(V[p], Js[p]) - net.sel(V[p], Ia[p]) - net.sel(V[p], Ic[p])
            v = v + lsum(net.sel(V[q], Is[p], Js[q]) + net.sel(V[q], Js[p], Is[q]) + net.sel(V[q], Ic[p], Jc[q]) for q in range(1, NMAX + 1) if q != p)
            v = v + net.sel(Y, Ia[p])
            NV.append(v)
        V = NV
    # ---- query (digit at offset 1, mark at offset 2)
    hq = A(w.dig(1), w.s(2, QMARK)); ha = A(w.dig(1), w.s(2, AMARK))
    qd = w.dval(1)
    merged = w.s(2, (':?', ':@'))
    # ---- shape: every token inside the call is accounted for
    n_in = lsum(inside[o] for o in range(1, K))
    expect = 2 + n + 1 + oplen + nops - net.ge(nops, 1) + 2 * (hq + ha) - merged
    shape_ok = net.eq(n_in - expect, 0)
    valid = A(w.s(0, 'CE'), found, nameok, colon_ok, 1 - too_long, shape_ok, net.eq(bad, 0), 1 - net.ge(nops, MAXOPS + 1))
    # ---- reply
    qh = {p: A(hq, net.eq(qd, p), net.ge(n - p, 0)) for p in range(1, NMAX + 1)}
    qval = lsum(net.sel(V[p], qh[p]) for p in qh)
    q_ok = A(valid, lsum(qh.values()))
    match = {p: A(ha, net.ge(n - p, 0), net.eq(V[p] - qd, 0)) for p in range(1, NMAX + 1)}
    nmatch = lsum(match.values()); apos = lsum(p * match[p] for p in match)
    a_ok = A(valid, net.eq(nmatch, 1))
    noq = 1 - hq - ha
    slots = []
    for k in range(1, LMAX + 1):
        for d in range(10):
            parts = [A(valid, noq, nge[k], net.eq(V[k], d))]
            if k == 1:
                parts += [A(q_ok, net.eq(qval, d)), A(a_ok, net.eq(apos, d))]
            slots.append(lsum(parts))
    slots.append(A(w.s(0, 'CE'), found, nameok))                      # 'mine': this call names this tool
    pr = dict(n=[n], nops=[nops], shape=[n_in, expect], valid=[valid], V=V[1:], bad=[bad], ops=[x for op in ops for x in op])
    return (net, slots, pr) if probes else (net, slots)


def build():
    n1, S = build_stack1(); n2, O = build_stack2_seq(SI, NS, K, CONTENT, LMAX)
    assert len(S) == N_STATE
    return compile_net(n1, S), compile_net(n2, O)


SPEC = dict(build=build, syms=SYMS, out=OUT, key='CE', K=K, W=LMAX + 3, text=TEXT)
