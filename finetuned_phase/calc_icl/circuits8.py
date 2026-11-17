"""Time-slot micromachine (MM8) for BBH temporal_sequences.
  <time>W-C;a-b;a-b;...|s-e;s-e;s-e;s-e</time>  ->  box: A, B, C or D
  W-C   the window (woke up at W, place closed after C), hours 0..24
  a-b   times the person was seen elsewhere (busy)
  s-e   the four options, in order A, B, C, D
The answer is the one option inside the window that overlaps no busy hour (anything else is an error)."""
from hdl import Net, Lin, lsum, compile_net
from seqmachine import Win, inside_flags, build_stack2_seq

LET = 'ABCD'
SYMS = list('0123456789') + ['-', ';', '|'] + list(LET) + ['CS', 'CE', 'OS', 'OE']
NS = len(SYMS); SI = {s: i for i, s in enumerate(SYMS)}
K = 96
CONTENT = list(LET)
OUT = CONTENT + ['OS', 'OE']
LMAX = 1
N_STATE = LMAX * len(CONTENT)
H = 25


def time_ref(call):
    import re
    m = re.fullmatch(r'(\d{1,2})-(\d{1,2})((?:;\d{1,2}-\d{1,2})*)\|(\d{1,2})-(\d{1,2});(\d{1,2})-(\d{1,2});(\d{1,2})-(\d{1,2});(\d{1,2})-(\d{1,2})', call)
    if not m or len(call) > 200: return ''
    W, C = int(m.group(1)), int(m.group(2))
    busy = [tuple(map(int, x.split('-'))) for x in m.group(3).split(';')[1:]]
    opts = [(int(m.group(4 + 2 * k)), int(m.group(5 + 2 * k))) for k in range(4)]
    vals = [W, C] + [v for b in busy for v in b] + [v for o in opts for v in o]
    if any(v > 24 for v in vals): return ''
    free = [k for k, (s, e) in enumerate(opts) if s >= W and e <= C and e > s and not any(a <= h < b for (a, b) in busy for h in range(s, e))]
    return LET[free[0]] if len(free) == 1 else ''


def build_stack1():
    net = Net(K * NS); w = Win(net, SI, NS, K)
    inside, cs_at = inside_flags(net, w, K)
    dig = lambda o: w.any(o, list('0123456789'))
    dv = lambda o: lsum(w.s(o, str(d)) * d for d in range(1, 10))
    sep_a = lambda o: w.s(o, ';') + w.s(o, '|') + w.s(o, 'CE')          # what may follow an interval
    sep_b = lambda o: w.s(o, ';') + w.s(o, '|') + w.s(o, 'CS')          # what may precede an interval
    M = [m for m in inside if 2 <= m < K - 3]
    iv, a, b, first = {}, {}, {}, {}
    for m in M:
        pb1 = net.AND(dig(m - 1), sep_a(m - 2)); pb2 = net.AND(dig(m - 1), dig(m - 2), sep_a(m - 3))
        pa1 = net.AND(dig(m + 1), sep_b(m + 2)); pa2 = net.AND(dig(m + 1), dig(m + 2), sep_b(m + 3))
        iv[m] = net.AND(w.s(m, '-'), inside[m], pb1 + pb2, pa1 + pa2)
        bval = net.sel(dv(m - 1), net.AND(iv[m], pb1)) + net.sel(10 * dv(m - 1) + dv(m - 2), net.AND(iv[m], pb2), M=99)
        aval = net.sel(dv(m + 1), net.AND(iv[m], pa1)) + net.sel(10 * dv(m + 2) + dv(m + 1), net.AND(iv[m], pa2), M=99)
        a[m], b[m] = aval, bval
        first[m] = net.AND(iv[m], net.AND(dig(m + 1), w.s(m + 2, 'CS')) + net.AND(dig(m + 1), dig(m + 2), w.s(m + 3, 'CS')))
    pipe_after = {u: net.clip01(lsum(net.AND(w.s(v, '|'), inside[v]) for v in inside if v > u)) for u in inside}   # a '|' earlier in the call
    opt_idx = {m: lsum(net.AND(w.s(u, ';'), inside[u], pipe_after[u]) for u in inside if u > m) for m in M}
    is_opt = {m: net.AND(iv[m], pipe_after[m]) for m in M}
    opt = {(m, k): net.AND(is_opt[m], net.eq(opt_idx[m], k)) for m in M for k in range(4)}
    busy_iv = {m: net.relu(iv[m] - first[m] - is_opt[m]) for m in M}
    Wv = lsum(net.sel(a[m], first[m], M=99) for m in M); Cv = lsum(net.sel(b[m], first[m], M=99) for m in M)
    busy = {}
    for h in range(H):
        busy[h] = net.clip01(lsum(net.AND(busy_iv[m], net.ge(h - a[m], 0), net.ge(b[m] - h, 1)) for m in M))
    free = []
    for k in range(4):
        s = lsum(net.sel(a[m], opt[m, k], M=99) for m in M); e = lsum(net.sel(b[m], opt[m, k], M=99) for m in M)
        has = net.clip01(lsum(opt[m, k] for m in M))
        clash = net.clip01(lsum(net.AND(busy[h], net.ge(h - s, 0), net.ge(e - h, 1)) for h in range(H)))
        big = net.clip01(lsum(net.ge(x, 25) for x in [Wv, Cv, s, e]))
        free.append(net.AND(has, net.ge(s - Wv, 0), net.ge(Cv - e, 0), net.ge(e - s, 1), 1 - clash, 1 - big))
    n_first = lsum(first.values()); n_opt = lsum(is_opt.values())
    n_pipe = lsum(net.AND(w.s(u, '|'), inside[u]) for u in inside)
    # shape: every inside token belongs to an interval or is a separator
    tok_iv = lsum(iv[m] for m in M) + lsum(net.AND(iv[m], dig(m - 2)) for m in M) + lsum(net.AND(iv[m], dig(m + 2), 1 - w.s(m + 2, 'CS')) for m in M)
    n_tok = lsum(inside[o] for o in inside)
    n_sep = lsum(net.AND(w.s(u, ';'), inside[u]) for u in inside) + n_pipe
    shape = net.eq(n_tok - (3 * lsum(iv[m] for m in M) + lsum(net.AND(iv[m], dig(m - 2)) for m in M) + lsum(net.AND(iv[m], dig(m + 2)) for m in M)) - n_sep, 0)
    ok = net.AND(w.s(0, 'CE'), net.eq(n_first, 1), net.eq(n_opt, 4), net.eq(n_pipe, 1), net.eq(lsum(free), 1), shape)
    return net, [net.AND(ok, f) for f in free]


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
