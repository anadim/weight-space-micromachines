"""A tiny hardware-description layer: write circuits as affine forms + ReLU units, compile to dense ReLU layers.

Every wire carries a small integer, so the compiled network is exact in float32.
"""
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as Fn


class Lin:
    """Affine form over wires: sum_k coef_k * wire_k + c."""
    __slots__ = ('t', 'c')

    def __init__(self, t=None, c=0):
        self.t = t if t is not None else {}
        self.c = c

    def __add__(self, o):
        if isinstance(o, Lin):
            t = dict(self.t)
            for k, v in o.t.items():
                nv = t.get(k, 0) + v
                if nv == 0: t.pop(k, None)
                else: t[k] = nv
            return Lin(t, self.c + o.c)
        return Lin(dict(self.t), self.c + o)
    __radd__ = __add__

    def __mul__(self, k):
        return Lin({a: b * k for a, b in self.t.items()}, self.c * k) if k != 0 else Lin()
    __rmul__ = __mul__

    def __neg__(self): return self * -1
    def __sub__(self, o): return self + (o * -1 if isinstance(o, Lin) else -o)
    def __rsub__(self, o): return (self * -1) + o


def lsum(xs):
    t, c = {}, 0
    for x in xs:
        if isinstance(x, Lin):
            c += x.c
            for k, v in x.t.items():
                t[k] = t.get(k, 0) + v
        else:
            c += x
    return Lin({k: v for k, v in t.items() if v != 0}, c)


class Net:
    def __init__(self, n_in):
        self.n_in = n_in
        self.depth = [0] * n_in
        self.units = []           # Lin pre-activation of unit i (wire id n_in + i)
        self.cse = {}

    def inp(self, i): return Lin({i: 1})

    def relu(self, x):
        if not isinstance(x, Lin): return Lin(c=max(0, x))
        if not x.t: return Lin(c=max(0, x.c))
        key = (tuple(sorted(x.t.items())), x.c)
        w = self.cse.get(key)
        if w is None:
            w = len(self.depth)
            self.depth.append(1 + max(self.depth[k] for k in x.t))
            self.units.append(x)
            self.cse[key] = w
        return Lin({w: 1})

    # ---- gates on integer-valued wires ----
    def AND(self, *xs):                                                      # all xs in {0,1}
        return xs[0] if len(xs) == 1 else self.relu(lsum(xs) - (len(xs) - 1))
    def ge(self, v, k): return self.relu(v - (k - 1)) - self.relu(v - k)     # [v >= k], v integer
    def eq(self, v, k): return self.ge(v, k) - self.ge(v, k + 1)
    def clip01(self, v): return self.relu(v) - self.relu(v - 1)              # min(v,1) for v >= 0 integer
    def sel(self, v, *flags, M=9):                                           # v if all flags else 0  (0 <= v <= M)
        return self.relu(v - M * (len(flags) - lsum(flags)))

    # ---- multi-digit helpers (LSB-first lists of digit wires) ----
    def resolve(self, cols):
        """cols[i] in 0..18 -> base-10 digits (len+1), carry-lookahead."""
        n = len(cols)
        G = [self.ge(c, 10) for c in cols]
        P = [self.eq(c, 9) for c in cols]
        carry = [Lin()]
        for r in range(1, n + 1):
            carry.append(lsum(self.AND(G[j], *P[j + 1:r]) for j in range(r)))
        out = []
        for r in range(n):
            v = cols[r] + carry[r]
            out.append(v - 10 * self.ge(v, 10))
        return out + [carry[n]]

    def resolve_sub(self, d):
        """d[i] in -9..9 (digitwise x-y) -> digits of (x-y) mod 10^n, and the borrow-out flag (x<y)."""
        n = len(d)
        G = [self.ge(-c, 1) for c in d]
        P = [self.eq(c, 0) for c in d]
        bor = [Lin()]
        for r in range(1, n + 1):
            bor.append(lsum(self.AND(G[j], *P[j + 1:r]) for j in range(r)))
        out = []
        for r in range(n):
            v = d[r] - bor[r]
            out.append(v + 10 * self.ge(-v, 1))
        return out, bor[n]

    def normalize(self, cols, bound):
        """column sums (each <= bound) -> base-10 digits."""
        cols = list(cols)
        while bound > 18:
            if bound >= 100:
                hs, ts, us = [], [], []
                for c in cols:
                    h = lsum(self.ge(c, 100 * j) for j in range(1, bound // 100 + 1))
                    rem = c - 100 * h
                    t = lsum(self.ge(rem, 10 * j) for j in range(1, 10))
                    hs.append(h); ts.append(t); us.append(rem - 10 * t)
                n = len(cols)
                cols = [lsum([us[k] if k < n else 0, ts[k - 1] if 0 <= k - 1 < n else 0, hs[k - 2] if 0 <= k - 2 < n else 0]) for k in range(n + 2)]
                bound = 18 + bound // 100
            else:
                fs, us = [], []
                for c in cols:
                    f = lsum(self.ge(c, 10 * j) for j in range(1, bound // 10 + 1))
                    fs.append(f); us.append(c - 10 * f)
                n = len(cols)
                cols = [lsum([us[k] if k < n else 0, fs[k - 1] if k >= 1 else 0]) for k in range(n + 1)]
                bound = 9 + bound // 10
        return self.resolve(cols)

    def cmp_ge(self, X, Y, chunk=20):
        """[X >= Y] for digit lists (LSB-first, any lengths). Chunked so every sum stays below 2^24."""
        n = max(len(X), len(Y))
        sg = []
        for i in range(n):
            x = X[i] if i < len(X) else Lin()
            y = Y[i] if i < len(Y) else Lin()
            sg.append(self.ge(x - y, 1) - self.ge(y - x, 1))
        sums = [lsum(s * (1 << i) for i, s in enumerate(sg[c:c + chunk])) for c in range(0, n, chunk)]
        res = self.ge(sums[0], 0)
        for S in sums[1:]:                                   # more significant chunks decide first
            res = self.ge(S, 1) + self.AND(self.eq(S, 0), res)
        return res


def compile_net(net, outs):
    """-> dict of numpy arrays describing a dense ReLU net with skip connections. Dead units are pruned."""
    n_in = net.n_in
    live = set()
    stack = [k for o in outs for k in o.t]
    while stack:
        w = stack.pop()
        if w in live or w < n_in: continue
        live.add(w)
        stack.extend(net.units[w - n_in].t)
    order = sorted(live, key=lambda w: (net.depth[w], w))
    pos = {i: i for i in range(n_in)}
    for i, w in enumerate(order): pos[w] = n_in + i
    layers, i = [], 0
    while i < len(order):
        d = net.depth[order[i]]; j = i
        while j < len(order) and net.depth[order[j]] == d: j += 1
        us = [net.units[w - n_in] for w in order[i:j]]
        idx = sorted({pos[k] for u in us for k in u.t})
        col = {p: c for c, p in enumerate(idx)}
        W = np.zeros((len(us), len(idx)), dtype=np.float32); b = np.zeros(len(us), dtype=np.float32)
        for r, u in enumerate(us):
            b[r] = u.c
            for k, v in u.t.items(): W[r, col[pos[k]]] = v
        layers.append((np.array(idx, dtype=np.int64), W, b))
        i = j
    idx = sorted({pos[k] for o in outs for k in o.t})
    col = {p: c for c, p in enumerate(idx)}
    Wo = np.zeros((len(outs), len(idx)), dtype=np.float32); bo = np.zeros(len(outs), dtype=np.float32)
    for r, o in enumerate(outs):
        bo[r] = o.c
        for k, v in o.t.items(): Wo[r, col[pos[k]]] = v
    return {'layers': layers, 'out': (np.array(idx, dtype=np.int64), Wo, bo), 'n_in': n_in, 'n_units': len(order)}


class ReLUNet(nn.Module):
    """Feed-forward ReLU network with dense skip connections; weights are fixed buffers.
    Weights are sparse (a few nonzeros per unit), so they are stored and applied as CSR matrices by default."""

    def __init__(self, comp, sparse=True):
        super().__init__()
        self.L = len(comp['layers']); self.sparse = sparse
        for l, (idx, W, b) in enumerate(comp['layers'] + [comp['out']]):
            self.register_buffer(f'i{l}', torch.from_numpy(idx))
            Wt = torch.from_numpy(W)
            if sparse:
                Wt = Wt.to_sparse_coo().coalesce()
                self.register_buffer(f'wi{l}', Wt.indices()); self.register_buffer(f'wv{l}', Wt.values())
                setattr(self, f'shape{l}', tuple(W.shape))
            else:
                self.register_buffer(f'w{l}', Wt)
            self.register_buffer(f'b{l}', torch.from_numpy(b))
        self.n_units = comp['n_units']
        self.n_dense = sum(W.size + b.size for _, W, b in comp['layers'] + [comp['out']])
        self.nnz = sum(int((W != 0).sum()) for _, W, b in comp['layers'] + [comp['out']])
        self._csr = {}

    def lin(self, l, x):
        b = getattr(self, f'b{l}')
        if not self.sparse:
            return Fn.linear(x, getattr(self, f'w{l}'), b)
        key = (l, x.device)
        if key not in self._csr:
            self._csr[key] = torch.sparse_coo_tensor(getattr(self, f'wi{l}'), getattr(self, f'wv{l}'), getattr(self, f'shape{l}')).to_sparse_csr()
        flat = x.reshape(-1, x.shape[-1])
        return (torch.sparse.mm(self._csr[key], flat.T).T + b).reshape(*x.shape[:-1], -1)

    def forward(self, x, return_wires=False, chunk=4096):
        if not return_wires and x.numel() // x.shape[-1] > chunk:       # bound memory: process positions in chunks
            flat = x.reshape(-1, x.shape[-1])
            return torch.cat([self.forward(flat[i:i + chunk]) for i in range(0, flat.shape[0], chunk)]).reshape(*x.shape[:-1], -1)
        n_in = x.shape[-1]
        wires = x.new_zeros(*x.shape[:-1], n_in + self.n_units)      # one buffer holds every wire
        wires[..., :n_in] = x
        p = n_in
        for l in range(self.L):
            h = Fn.relu(self.lin(l, wires[..., getattr(self, f'i{l}')]))
            wires[..., p:p + h.shape[-1]] = h
            p += h.shape[-1]
        out = self.lin(self.L, wires[..., getattr(self, f'i{self.L}')])
        return (out, wires) if return_wires else out
