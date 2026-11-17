"""order_gen: a generic ORDERING-CONSTRAINT solver compiled into a ReLU network (Team A, BBH redo).

f(x): given N objects (numbered 1..N, N = 2..7) and constraints of the forms
  a<b (a before b), a>b (a after b), a+b (a immediately before b), a~b (a next to b),
  a=k / a=-k (a at position k from the start / from the end), a!=k / a!=-k (a not at that position),
return what all consistent orderings agree on: the object at a position (?k, ?-k), the position of an object (@a), or the
whole order (no query; needs exactly one consistent ordering). Empty reply = no consistent ordering / not determined /
malformed call. Call / semantics: see callfmt.py (order N:C,C,...[Q]); the circuit implements exactly callfmt.order_ref.

It reads only call-syntax tokens (digits, ':' ',' and the operator tokens '<' '>' '+' '~' '=' '=-' '!=' '!=-', the query
tokens '?' '?-' '@' and their merged forms with ':', the word 'order', a space, call/response tokens). No object names,
no relation words: the host model writes the numbers. stack 1 runs at </tool_call>:
  1. frame: the most recent <tool_call> (offset c), 'order' + space, the count N at c-3, a ':'-class token at c-4;
  2. every constraint 'a op b' (preceded by ':' or ',' and followed by ',', a query token or </tool_call>) sets a flag
     present[op, a, b];
  3. one ReLU unit per ordering of N objects (N = 2..7: 5,913 units): alive = [N flagged] - (#present constraints the
     ordering violates), so alive is 1 iff the ordering satisfies every constraint;
  4. per position and object: is there an alive ordering with that object there (and per object and position);
  5. the query, the validity checks (every token accounted for), and the reply digits.
"""
import itertools
from circ_common import Net, Lin, lsum, compile_net, Win, call_frame, build_stack2_seq, DIGITS
from callfmt import ORDER_NMIN as NMIN, ORDER_NMAX as NMAX, ORDER_MAXC as MAXC, ORDER_OPS as OPS, order_violates

SYMS = DIGITS + [':', ',', '<', '>', '+', '~', '=', '=-', '!=', '!=-', '?', '?-', '@', ':?', ':@', 'NAME', 'SP', 'CS', 'CE', 'OS', 'OE']
NS = len(SYMS); SI = {s: i for i, s in enumerate(SYMS)}
K = 5 + 4 * MAXC + 3
CONTENT = [str(d) for d in range(1, NMAX + 1)]
LMAX = NMAX
OUT = CONTENT + ['OS', 'OE']
N_STATE = LMAX * len(CONTENT) + 1           # + 'mine'
TEXT = {**{d: d for d in DIGITS}, **{s: s for s in [':', ',', '<', '>', '+', '~', '=', '=-', '!=', '!=-', '?', '?-', '@', ':?', ':@']},
        'NAME': 'order', 'SP': ' ', 'CS': '<tool_call>', 'CE': '</tool_call>', 'OS': '<tool_response>', 'OE': '</tool_response>'}

COLON = (':', ':?', ':@')          # (':?-' is two tokens ':?' '-'; an empty constraint list with a ?-k query is not supported)
SEP = (':', ',')
QP, QM, QA = ('?', ':?'), ('?-',), ('@', ':@')
AFTER = (',', '?', '?-', '@', 'CE')


def perms():
    for n in range(NMIN, NMAX + 1):
        for p in itertools.permutations(range(1, n + 1)):          # p[k-1] = object at position k
            yield n, p


def build_stack1(probes=False):
    net = Net(K * NS); w = Win(net, SI, NS, K); A = net.AND
    csl, inside, found, nameok = call_frame(net, w, 'NAME')
    nfl = {n: lsum(A(csl[c], w.s(c - 3, str(n)), w.s(c - 4, COLON)) for c in range(5, K)) for n in range(NMIN, NMAX + 1)}
    nok = lsum(nfl.values())
    # ---- constraints ending at offset o: a at o+2, op at o+1, b at o, separator at o+3, follower at o-1
    struct = {o: A(w.s(o + 3, SEP), inside[o + 3], w.s(o - 1, AFTER)) for o in range(1, K - 3)}
    ncons = lsum(A(struct[o], w.dig(o), w.s(o + 1, OPS), w.dig(o + 2)) for o in struct)
    present, nrec = {}, Lin()
    for op in OPS:
        for a in range(1, NMAX + 1):
            for b in range(1, NMAX + 1):
                if op in ('<', '>', '+', '~') and a == b: continue
                hits = lsum(A(struct[o], w.s(o + 2, str(a)), w.s(o + 1, op), w.s(o, str(b))) for o in struct)
                nrec = nrec + hits
                present[op, a, b] = net.clip01(hits)
    all_known = net.eq(ncons - nrec, 0)                                   # every constraint uses objects/positions 1..7
    # ---- every ordering
    alive = []
    X = {(k, d): [] for k in range(1, NMAX + 1) for d in range(1, NMAX + 1)}       # object d at position k (from start)
    XE = {(k, d): [] for k in range(1, NMAX + 1) for d in range(1, NMAX + 1)}      # object d at position k from the end
    for n, p in perms():
        viol = lsum(present[c] for c in present if order_violates(p, *c))
        ok = net.relu(nfl[n] - viol); alive.append(ok)
        for k in range(1, n + 1):
            X[k, p[k - 1]].append(ok); XE[k, p[n - k]].append(ok)
    nalive = lsum(alive)
    Xc = {kd: net.clip01(lsum(v)) for kd, v in X.items()}
    XEc = {kd: net.clip01(lsum(v)) for kd, v in XE.items()}
    # ---- query: digit at offset 1, query token at offset 2
    x1 = {v: w.s(1, str(v)) for v in range(1, NMAX + 1)}
    qp = {v: A(x1[v], w.s(2, QP)) for v in x1}; qm = {v: A(x1[v], w.s(2, QM)) for v in x1}; qa = {v: A(x1[v], w.s(2, QA)) for v in x1}
    hasq = A(w.dig(1), w.s(2, QP + QM + QA))
    merged = A(w.dig(1), w.s(2, (':?', ':@')))
    # ---- shape
    n_in = lsum(inside[o] for o in range(1, K))
    expect = 2 + 1 + 1 + 3 * ncons + ncons - net.ge(ncons, 1) + 2 * hasq - merged
    shape_ok = net.eq(n_in - expect, 0)
    valid = A(w.s(0, 'CE'), found, nameok, nok, shape_ok, all_known, 1 - net.ge(ncons, MAXC + 1))
    # ---- reply
    ans = {d: lsum([A(qp[k], Xc[k, d]) for k in range(1, NMAX + 1)] + [A(qm[k], XEc[k, d]) for k in range(1, NMAX + 1)]
                   + [A(qa[a], Xc[d, a]) for a in range(1, NMAX + 1)]) for d in range(1, NMAX + 1)}
    one = A(valid, hasq, net.eq(lsum(ans.values()), 1))
    full = A(valid, 1 - hasq, net.eq(nalive, 1))
    slots = []
    for k in range(1, LMAX + 1):
        for d in range(1, NMAX + 1):
            parts = [A(full, Xc[k, d])]
            if k == 1: parts.append(A(one, ans[d]))
            slots.append(lsum(parts))
    slots.append(A(w.s(0, 'CE'), found, nameok))                      # 'mine': this call names this tool
    pr = dict(nalive=[nalive], ncons=[ncons], nrec=[nrec], shape=[n_in, expect], valid=[valid], ans=list(ans.values()))
    return (net, slots, pr) if probes else (net, slots)


def build():
    n1, S = build_stack1(); n2, O = build_stack2_seq(SI, NS, K, CONTENT, LMAX)
    assert len(S) == N_STATE
    return compile_net(n1, S), compile_net(n2, O)


SPEC = dict(build=build, syms=SYMS, out=OUT, key='CE', K=K, W=LMAX + 3, text=TEXT)
