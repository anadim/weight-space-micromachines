"""Logic micromachine (MM5): solves ordering puzzles (BBH logical_deduction) by checking every ordering at once.

Call:  <logic>N:c;c;...;?P</logic>     ->  box: the object at position P
  N        number of objects (3..7), numbered in the order the story lists them
  a<b      object a comes before object b        a>b   object a comes after object b
  a=k      object a is at position k             a=-k  object a is at position k counted from the end
  ?k / ?-k the position asked about
One hidden unit per ordering; it is 1 iff no stated constraint rules that ordering out. Exactly one must survive.
"""
import itertools
from hdl import Net, Lin, lsum, compile_net
from seqmachine import Win, inside_flags, build_stack2_seq

NMIN, NMAX = 3, 7
SYMS = list('0123456789') + ['=', '=-', '<', '>', ';', ':', '?', '?-', 'CS', 'CE', 'OS', 'OE']
NS = len(SYMS); SI = {s: i for i, s in enumerate(SYMS)}
K = 64
CONTENT = [str(d) for d in range(1, NMAX + 1)]
OUT = CONTENT + ['OS', 'OE']
LMAX = 1
N_STATE = LMAX * len(CONTENT)


def perms():
    for n in range(NMIN, NMAX + 1):
        for p in itertools.permutations(range(1, n + 1)):      # p[k-1] = object at position k
            yield n, p


def logic_ref(call):
    """Python reference: '' = error / not exactly one solution."""
    import re
    m = re.fullmatch(r'(\d):((?:\d(?:<|>|=-|=)\d;)*)\?(-?)(\d)', call)
    if not m: return ''
    n = int(m.group(1))
    if not NMIN <= n <= NMAX: return ''
    cons = re.findall(r'(\d)(<|>|=-|=)(\d);', m.group(2))
    q = int(m.group(4)); q = n + 1 - q if m.group(3) else q
    if not 1 <= q <= n: return ''
    sols = []
    for p in itertools.permutations(range(1, n + 1)):
        pos = {o: k + 1 for k, o in enumerate(p)}; ok = True
        for a, op, b in cons:
            a, b = int(a), int(b)
            if a > n or (op in '<>' and (b > n or a == b)) or (op in ('=', '=-') and not 1 <= b <= n): ok = False; break
            if op == '<' and not pos[a] < pos[b]: ok = False; break
            if op == '>' and not pos[a] > pos[b]: ok = False; break
            if op == '=' and pos[a] != b: ok = False; break
            if op == '=-' and pos[a] != n + 1 - b: ok = False; break
        if ok: sols.append(p)
    return str(sols[0][q - 1]) if len(sols) == 1 else ''


def violates(n, p, c):
    op, a, b = c
    pos = {o: k + 1 for k, o in enumerate(p)}
    if a > n: return True
    if op in '<>':
        if b > n or a == b: return True
        return not (pos[a] < pos[b]) if op == '<' else not (pos[a] > pos[b])
    if not 1 <= b <= n: return True
    return pos[a] != b if op == '=' else pos[a] != n + 1 - b


def build_stack1():
    net = Net(K * NS); w = Win(net, SI, NS, K)
    inside, cs_at = inside_flags(net, w, K)
    dig = lambda o, d: w.s(o, str(d))
    sep = lambda o: w.s(o, ';') + w.s(o, ':')
    # number of objects: <logic> N :
    nflag = {n: lsum(net.AND(cs_at[j], dig(j - 1, n), w.s(j - 2, ':')) for j in range(3, K)) for n in range(NMIN, NMAX + 1)}
    # query at the end: ; ? d   or  ; ?- d
    qp = {k: net.AND(w.s(0, 'CE'), dig(1, k), w.s(2, '?'), w.s(3, ';')) for k in range(1, NMAX + 1)}
    qm = {k: net.AND(w.s(0, 'CE'), dig(1, k), w.s(2, '?-'), w.s(3, ';')) for k in range(1, NMAX + 1)}
    # constraints a op b ; preceded by ; or :
    present = {}
    for op in ('<', '>', '=', '=-'):
        for a in range(1, NMAX + 1):
            for b in range(1, NMAX + 1):
                if op in '<>' and a == b: continue
                hits = [net.AND(sep(o + 3), inside[o + 3], dig(o + 2, a), w.s(o + 1, op), dig(o, b), w.s(o - 1, ';'))
                        for o in range(3, K - 4) if o + 3 in inside]
                present[op, a, b] = net.clip01(lsum(hits))
    # malformed: every token inside the call must belong to 'N:', a 'a op b;' group or the final '?k'
    anyd = lambda o: lsum(dig(o, d) for d in range(10))
    anyop = lambda o: w.s(o, '<') + w.s(o, '>') + w.s(o, '=') + w.s(o, '=-')
    groups = [net.AND(sep(o + 3), inside[o + 3], anyd(o + 2), anyop(o + 1), anyd(o), w.s(o - 1, ';')) for o in range(3, K - 4) if o + 3 in inside]
    n_in = lsum(inside[o] for o in inside)
    nq = lsum(qp.values()) + lsum(qm.values())
    shape_ok = net.eq(n_in - 4 * lsum(groups) - 2 * nq, 2)            # 'N:' + groups + '?k'
    recognised = lsum(net.AND(sep(o + 3), inside[o + 3], dig(o + 2, a), w.s(o + 1, op), dig(o, b), w.s(o - 1, ';'))
                      for op in ('<', '>', '=', '=-') for a in range(1, NMAX + 1) for b in range(1, NMAX + 1) if not (op in '<>' and a == b)
                      for o in range(3, K - 4) if o + 3 in inside)
    all_known = net.eq(lsum(groups) - recognised, 0)
    valid, objk, objm = [], {k: [] for k in range(1, NMAX + 1)}, {k: [] for k in range(1, NMAX + 1)}
    for n, p in perms():
        viol = [present[c] for c in present if violates(n, p, (c[0], c[1], c[2]))]
        v = net.relu(nflag[n] - lsum(viol))
        valid.append(v)
        for k in range(1, n + 1):
            objk[k].append(v * p[k - 1]); objm[k].append(v * p[n - k])
    count = lsum(valid)
    qflag = lsum(qp.values()) + lsum(qm.values())
    ok = net.AND(net.eq(count, 1), qflag, shape_ok, all_known)
    digit = lsum([net.sel(lsum(objk[k]), qp[k]) for k in range(1, NMAX + 1)] + [net.sel(lsum(objm[k]), qm[k]) for k in range(1, NMAX + 1)])
    slot = [net.AND(ok, net.eq(digit, d)) for d in range(1, NMAX + 1)]
    return net, slot


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
