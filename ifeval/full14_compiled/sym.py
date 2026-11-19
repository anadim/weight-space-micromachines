"""Symbolic wires -> exact ReLU netlist (+ gather / hard-attention / vocab nodes), and its compiler.

Every value is an INTEGER carried on a float64 wire. Each Sym holds an affine form (wsm_hdl.Lin) over wires plus a
guaranteed interval [lo, hi] (semantic bounds). Non-linear ops are lowered to ReLU units that are exact on integers
inside the interval (all magnitudes < 2^45, so float64 matmul + ReLU is exact):
    max(a,b)      = a + relu(b-a)
    where(c,x,y)  = y + relu(d - L - (U-L)(1-c)) + L c        (c in {0,1}, d = x-y in [L,U])
    [x >= k]      = relu(x-k+1) - relu(x-k)                     (x integer)
    AND(a,b)      = relu(a+b-1)
Node kinds besides ReLU units: gather (row of a FIXED table selected by an integer wire), attn (hard attention over the
token memory: value at the position whose key equals the query), memwrite (append to that memory), vocab (blocked-set
readout over the vocabulary, see executor).
"""
import math
from collections import OrderedDict
import numpy as np
from wsm_hdl import Lin, lsum

BIG = 10 ** 6            # assumed bound for counters / positions (sequences < 10^6 tokens); checked at runtime in debug mode


class Graph:
    def __init__(self):
        self.lo, self.hi, self.depth, self.kind = [], [], [], []
        self.units = {}            # wire -> Lin (pre-activation of a ReLU unit)
        self.cse = {}
        self.gnodes = []           # dict(table, idx Lin, cols {col: wire}, depth)
        self.gcse = {}
        self.anodes = []           # dict(idx Lin, wire, depth)
        self.acse = {}
        self.mnodes = []           # memwrite: dict(pos Lin, val Lin, depth)
        self.vnodes = []           # vocab node: dict(ins {name: Lin}, outs {name: wire}, depth)
        self.inputs = OrderedDict()
        self.mem_depth = 0
        self.tables = {}           # name -> (columns list, per-col (lo,hi), nrows)

    def _wire(self, lo, hi, depth, kind):
        w = len(self.lo); self.lo.append(lo); self.hi.append(hi); self.depth.append(depth); self.kind.append(kind); return w

    def ldepth(self, lin): return max([self.depth[k] for k in lin.t] or [0])

    def inp(self, name, lo, hi):
        assert name not in self.inputs, name
        w = self._wire(lo, hi, 0, 'in'); self.inputs[name] = w
        return Sym(self, Lin({w: 1}), lo, hi)

    def relu(self, s):
        """ReLU unit on Sym s -> Sym (CSE)."""
        if s.hi <= 0: return const(self, 0)
        if s.lo >= 0: return s
        lin = s.lin
        if not lin.t: return const(self, max(0, lin.c))
        key = (tuple(sorted(lin.t.items())), lin.c)
        w = self.cse.get(key)
        if w is None:
            w = self._wire(0, s.hi, 1 + self.ldepth(lin), 'relu'); self.units[w] = lin; self.cse[key] = w
        return Sym(self, Lin({w: 1}), 0, s.hi)

    def mat(self, s):
        """materialize s on a fresh wire (identity ReLU on s - lo): keeps later affine forms short and small."""
        if s.isconst or (len(s.lin.t) == 1 and list(s.lin.t.values())[0] == 1 and s.lin.c == 0): return s
        lin = s.lin - s.lo
        key = (tuple(sorted(lin.t.items())), lin.c)
        w = self.cse.get(key)
        if w is None:
            w = self._wire(0, s.hi - s.lo, 1 + self.ldepth(lin), 'relu'); self.units[w] = lin; self.cse[key] = w
        return Sym(self, Lin({w: 1}) + s.lo, s.lo, s.hi)

    def magnitude(self, lin):
        return abs(lin.c) + sum(abs(v) * max(abs(self.lo[k]), abs(self.hi[k])) for k, v in lin.t.items())

    def table(self, name, cols, bounds, nrows):
        bounds = [(int(math.floor(lo)), int(math.ceil(hi))) for lo, hi in bounds]
        self.tables[name] = (list(cols), dict(zip(cols, bounds)), nrows)

    def gather(self, name, idx):
        """row of fixed table `name` at integer wire idx -> Row (columns created on demand)."""
        idx = S(self, idx)
        cols, bnd, nrows = self.tables[name]
        assert idx.lo >= 0 and idx.hi <= nrows - 1, (name, idx.lo, idx.hi, nrows)
        if not idx.lin.t:
            return Row(self, None, name, int(idx.lin.c))
        key = (name, tuple(sorted(idx.lin.t.items())), idx.lin.c)
        n = self.gcse.get(key)
        if n is None:
            n = dict(table=name, idx=idx.lin, cols={}, depth=1 + self.ldepth(idx.lin)); self.gnodes.append(n); self.gcse[key] = n
        return Row(self, n, name, None)

    def memwrite(self, pos, val):
        pos, val = S(self, pos), S(self, val)
        d = 1 + max(self.ldepth(pos.lin), self.ldepth(val.lin))
        self.mnodes.append(dict(pos=pos.lin, val=val.lin, depth=d)); self.mem_depth = max(self.mem_depth, d)

    def attn(self, pos, MAXMEM, vlo, vhi):
        pos = S(self, pos)
        assert pos.lo >= 0 and pos.hi <= MAXMEM - 1, (pos.lo, pos.hi)
        key = (tuple(sorted(pos.lin.t.items())), pos.lin.c)
        n = self.acse.get(key)
        if n is None:
            d = 1 + max(self.ldepth(pos.lin), self.mem_depth)
            w = self._wire(vlo, vhi, d, 'attn'); n = dict(idx=pos.lin, wire=w, depth=d); self.anodes.append(n); self.acse[key] = n
        return Sym(self, Lin({n['wire']: 1}), self.lo[n['wire']], self.hi[n['wire']])

    def vocab(self, ins, outs):
        """ins: {name: Sym}; outs: {name: (lo,hi)} -> {name: Sym}"""
        ins = {k: S(self, v) for k, v in ins.items()}
        d = 1 + max(self.ldepth(v.lin) for v in ins.values())
        ow = {k: self._wire(lo, hi, d, 'vocab') for k, (lo, hi) in outs.items()}
        self.vnodes.append(dict(ins={k: v.lin for k, v in ins.items()}, outs=ow, depth=d))
        return {k: Sym(self, Lin({w: 1}), *outs[k]) for k, w in ow.items()}


class Row:
    def __init__(self, G, node, name, crow):
        self.G, self.node, self.name, self.crow = G, node, name, crow

    def __call__(self, col):
        G = self.G; cols, bnd, nrows = G.tables[self.name]
        if self.node is None:                                   # constant index: fold the table value
            return const(G, G.table_values[self.name][self.crow, cols.index(col)])
        if col not in self.node['cols']:
            lo, hi = bnd[col]
            self.node['cols'][col] = G._wire(lo, hi, self.node['depth'], 'gather')
        w = self.node['cols'][col]
        return Sym(G, Lin({w: 1}), G.lo[w], G.hi[w])


def const(G, c):
    c = int(c) if float(c).is_integer() else c
    assert float(c).is_integer(), c
    return Sym(G, Lin(c=int(c)), int(c), int(c))


def S(G, x):
    return x if isinstance(x, Sym) else const(G, x)


class Sym:
    __slots__ = ('G', 'lin', 'lo', 'hi')

    def __init__(self, G, lin, lo, hi):
        assert lo <= hi, (lo, hi)
        self.G, self.lin, self.lo, self.hi = G, lin, lo, hi

    @property
    def isconst(self): return not self.lin.t
    @property
    def isbool(self): return self.lo >= 0 and self.hi <= 1

    def __add__(self, o):
        o = S(self.G, o); return Sym(self.G, self.lin + o.lin, self.lo + o.lo, self.hi + o.hi)
    __radd__ = __add__

    def __neg__(self): return Sym(self.G, self.lin * -1, -self.hi, -self.lo)
    def __sub__(self, o): return self + (-S(self.G, o))
    def __rsub__(self, o): return S(self.G, o) + (-self)

    def __mul__(self, o):
        G = self.G; o = S(G, o)
        if o.isconst: return self.scale(o.lin.c)
        if self.isconst: return o.scale(self.lin.c)
        if self.isbool: return where(self, o, 0)
        if o.isbool: return where(o, self, 0)
        raise ValueError(f'product of two non-binary wires {self.lo, self.hi} x {o.lo, o.hi}')
    __rmul__ = __mul__

    def scale(self, k):
        assert float(k).is_integer(), k
        k = int(k)
        if k == 0: return const(self.G, 0)
        lo, hi = sorted((self.lo * k, self.hi * k))
        return Sym(self.G, self.lin * k, lo, hi)

    # comparisons (integer wires; float thresholds allowed)
    def __ge__(self, o):
        if isinstance(o, Sym): return ge(self - o, 0)
        return ge(self, math.ceil(o))
    def __gt__(self, o):
        if isinstance(o, Sym): return ge(self - o, 1)
        return ge(self, math.floor(o) + 1)
    def __le__(self, o): return 1 - (self > o)
    def __lt__(self, o): return 1 - (self >= o)
    def __eq__(self, o):
        if isinstance(o, Sym): return eqk(self - o, 0)
        return eqk(self, o)
    def __ne__(self, o): return 1 - (self == o)
    __hash__ = None

    def __and__(self, o): return AND(self, S(self.G, o))
    __rand__ = __and__
    def __or__(self, o): return OR(self, S(self.G, o))
    __ror__ = __or__
    def __invert__(self):
        assert self.isbool; return 1 - self

    def clamp(self, min=None, max=None): return clamp(self, min, max)


def ge(x, k):
    """[x >= k] for integer x."""
    G = x.G
    if x.lo >= k: return const(G, 1)
    if x.hi < k: return const(G, 0)
    a = G.relu(x - (k - 1)); b = G.relu(x - k)
    r = Sym(G, a.lin - b.lin, 0, 1)
    if max(abs(x.lo), abs(x.hi)) > 2 ** 16: r = G.mat(r)          # a 0/1 wire, not a difference of two large wires
    return r


def eqk(x, k):
    G = x.G
    if not float(k).is_integer(): return const(G, 0)
    k = int(k)
    if x.isconst: return const(G, int(x.lin.c == k))
    if x.lo > k or x.hi < k: return const(G, 0)
    r = ge(x, k) - ge(x, k + 1)
    return Sym(G, r.lin, 0, 1)


def AND(*xs):
    G = xs[0].G
    xs = [S(G, x) for x in xs]
    for x in xs: assert x.isbool, (x.lo, x.hi)
    if any(x.isconst and x.lin.c == 0 for x in xs): return const(G, 0)
    xs = [x for x in xs if not x.isconst]
    if not xs: return const(G, 1)
    if len(xs) == 1: return xs[0]
    s = xs[0]
    for x in xs[1:]: s = s + x
    r = G.relu(s - (len(xs) - 1))
    return Sym(G, r.lin, 0, 1)


def OR(a, b):
    assert a.isbool and b.isbool
    return maximum(a, b)


def maximum(a, b):
    G = (a if isinstance(a, Sym) else b).G
    a, b = S(G, a), S(G, b)
    if a.lo >= b.hi: return a
    if b.lo >= a.hi: return b
    r = G.relu(b - a)
    return Sym(G, (a + r).lin, max(a.lo, b.lo), max(a.hi, b.hi))


def minimum(a, b):
    G = (a if isinstance(a, Sym) else b).G
    a, b = S(G, a), S(G, b)
    if a.hi <= b.lo: return a
    if b.hi <= a.lo: return b
    r = G.relu(a - b)
    return Sym(G, (a - r).lin, min(a.lo, b.lo), min(a.hi, b.hi))


def clamp(x, lo=None, hi=None):
    if lo is not None: x = maximum(x, lo)
    if hi is not None: x = minimum(x, hi)
    return x


def where(c, x, y):
    G = next(v for v in (c, x, y) if isinstance(v, Sym)).G
    c, x, y = S(G, c), S(G, x), S(G, y)
    assert c.isbool, (c.lo, c.hi)
    if c.isconst: return x if c.lin.c == 1 else y
    d = x - y
    L, U = d.lo, d.hi
    if L == 0 and U == 0 and d.isconst: return y
    r = G.relu(d - L - (U - L) * (1 - c))
    out = y + r + c.scale(L) if L != 0 else y + r
    return Sym(G, out.lin, min(x.lo, y.lo), max(x.hi, y.hi))


def rem2(x):
    """x mod 2 for integer x in [0, 3]"""
    assert x.lo >= 0 and x.hi <= 3, (x.lo, x.hi)
    r = x - ge(x, 2).scale(2)
    return Sym(x.G, r.lin, 0, 1)


def assume(x, lo, hi):
    """declare a tighter bound that interval arithmetic cannot see (checked at runtime in debug mode)."""
    return Sym(x.G, x.lin, max(x.lo, lo), min(x.hi, hi))


def mux(regs, i):
    """regs[i] for integer wire i (i in range)."""
    G = i.G
    out = None
    for k, r in enumerate(regs):
        term = (i == k) * S(G, r)
        out = term if out is None else out + term
    lo = min(S(G, r).lo for r in regs); hi = max(S(G, r).hi for r in regs)
    return Sym(G, out.lin, min(lo, 0), max(hi, 0))


# ---------------------------------------------------------------------------------------------------- compiler
def compile_graph(G, outs):
    """outs: OrderedDict name -> Sym. Returns a program: list of ops in execution order + output map.
    Dead nodes are pruned."""
    outs = OrderedDict((k, S(G, v)) for k, v in outs.items())
    live = set(); stack = []
    def need_lin(lin): stack.extend(lin.t)
    for v in outs.values(): need_lin(v.lin)
    for m in G.mnodes: need_lin(m['pos']); need_lin(m['val'])
    gnode_of = {}
    for n in G.gnodes:
        for c, w in n['cols'].items(): gnode_of[w] = n
    anode_of = {n['wire']: n for n in G.anodes}
    vnode_of = {}
    for n in G.vnodes:
        for w in n['outs'].values(): vnode_of[w] = n
    live_g, live_a, live_v = set(), set(), set()
    while stack:
        w = stack.pop()
        if w in live: continue
        live.add(w); k = G.kind[w]
        if k == 'relu': need_lin(G.units[w])
        elif k == 'gather':
            n = gnode_of[w]
            if id(n) not in live_g: live_g.add(id(n)); need_lin(n['idx'])
        elif k == 'attn':
            n = anode_of[w]
            if id(n) not in live_a: live_a.add(id(n)); need_lin(n['idx'])
        elif k == 'vocab':
            n = vnode_of[w]
            if id(n) not in live_v:
                live_v.add(id(n))
                for l in n['ins'].values(): need_lin(l)
    # exactness certificate: every affine form evaluated by the executor has |partial sums| < 2^52 (float64 exact)
    LIM = 2 ** 52
    worst = 0
    def chk(lin, what):
        nonlocal worst
        m = G.magnitude(lin); worst = max(worst, m)
        assert m < LIM, (what, m)
    for w in live:
        if G.kind[w] == 'relu': chk(G.units[w], ('relu', w))
    for n in G.gnodes:
        if id(n) in live_g: chk(n['idx'], 'gather')
    for n in G.anodes:
        if id(n) in live_a: chk(n['idx'], 'attn')
    for m in G.mnodes: chk(m['pos'], 'mem'); chk(m['val'], 'mem')
    for n in G.vnodes:
        if id(n) in live_v:
            for l in n['ins'].values(): chk(l, 'vocab')
    for k, v in outs.items(): chk(v.lin, ('out', k))
    # attn nodes must run after memwrites: their depth already >= mem_depth+1 (memwrites are created first in step)
    ops = []
    maxd = max([G.depth[w] for w in live] + [m['depth'] for m in G.mnodes] + [0])
    for d in range(1, maxd + 1):
        for m in G.mnodes:
            if m['depth'] == d: ops.append(('memwrite', m))
        us = sorted(w for w in live if G.kind[w] == 'relu' and G.depth[w] == d)
        if us: ops.append(('relu', [(w, G.units[w]) for w in us]))
        for n in G.gnodes:
            if id(n) in live_g and n['depth'] == d:
                cols = {c: w for c, w in n['cols'].items() if w in live}
                ops.append(('gather', dict(table=n['table'], idx=n['idx'], cols=cols)))
        for n in G.anodes:
            if id(n) in live_a and n['depth'] == d: ops.append(('attn', n))
        for n in G.vnodes:
            if id(n) in live_v and n['depth'] == d: ops.append(('vocab', n))
    for m in G.mnodes:
        for w in m['pos'].t: pass
    return dict(ops=ops, outs=outs, live=live, worst_magnitude=worst)


def lin_to_dense(lins, index):
    """list of Lin -> (cols, W, b) with cols = sorted wire ids used."""
    cols = sorted({k for l in lins for k in l.t})
    cpos = {c: i for i, c in enumerate(cols)}
    W = np.zeros((len(lins), len(cols))); b = np.zeros(len(lins))
    for r, l in enumerate(lins):
        b[r] = l.c
        for k, v in l.t.items(): W[r, cpos[k]] = v
    return [index[c] for c in cols], W, b
