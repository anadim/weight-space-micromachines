"""Export the compiled IFEval implant to a state_dict of plain tensors, and a small generic executor for it.

Executor op set (nothing else touches the implant's state):
  linear  : y = X[:, cols] @ W^T + b                     (matmul with fixed weights; also evaluates every gather/attention index)
  relu    : X[:, out] = ReLU(linear)                     (exact on integer wires, float64)
  gather  : X[:, out] = TABLE[idx][:, cols]              (row of a FIXED table selected by an integer wire = embedding lookup)
  argmax / hard attention :
            attn   : value at the memory position whose key equals the query (score -|key-q|, argmax), memory = token ids
                     appended once per step (the copy heads' KV cache)
            vocab  : argmax over the vocabulary of the readout logits; blocked = blk[top] (read at the argmax);
                     alt = argmax(logits - 1e30*blk)
  The vocab op's blocked mask blk is computed by vocab-plane layers built from the same ops: per-vocab fixed tables x register
  wires (matmul), ReLU, fixed bias rows.
  Readout (inside the vocab op): x = h*rsqrt(mean(h^2)+eps)*g ; logits = bf16(x) @ E^T with E, g a COPY of the host's LM-head rows
  and final-norm gain stored in this state_dict. The rmsnorm (square-mean-rsqrt) is the one op outside the four (see NOTEBOOK).
  Output: the write row index widx -> W = FORCE[widx] (gather of a fixed row; row V is the zero row).
"""
import torch, numpy as np
from collections import OrderedDict

HP = 2147483647
MAXMEM = 8192
NEGBIG = -(2 ** 40)


def _lin(lin, slot):
    cols = sorted(lin.t)
    return (torch.tensor([slot[c] for c in cols], dtype=torch.long), torch.tensor([float(lin.t[c]) for c in cols], dtype=torch.float64),
            torch.tensor(float(lin.c), dtype=torch.float64))


def export_prog(prog, which, n_in_names, sd):
    """prog from sym.compile_graph -> tensors in sd under prefix `which.`"""
    G = prog['G']
    slot = {}
    for i, nm in enumerate(n_in_names): slot[G.inputs[nm]] = i
    nxt = len(n_in_names)
    k = 0
    lo_all, hi_all = [0.0] * 0, []
    bounds = {}
    for kind, op in prog['ops']:
        p = f'{which}.{k:04d}.{kind}'
        if kind == 'relu':
            units = op
            cols = sorted({c for _, lin in units for c in lin.t})
            cpos = {c: i for i, c in enumerate(cols)}
            r, cc, v, b = [], [], [], []
            for ui, (w, lin) in enumerate(units):
                b.append(float(lin.c))
                for c, val in lin.t.items(): r.append(ui); cc.append(cpos[c]); v.append(float(val))
            sd[p + '.cols'] = torch.tensor([slot[c] for c in cols], dtype=torch.long)
            sd[p + '.Wi'] = torch.tensor([r, cc], dtype=torch.long); sd[p + '.Wv'] = torch.tensor(v, dtype=torch.float64)
            sd[p + '.Wshape'] = torch.tensor([len(units), len(cols)], dtype=torch.long)
            sd[p + '.b'] = torch.tensor(b, dtype=torch.float64)
            sd[p + '.out'] = torch.tensor([nxt], dtype=torch.long)
            for w, _ in units: slot[w] = nxt; bounds[nxt] = (G.lo[w], G.hi[w]); nxt += 1
        elif kind == 'gather':
            c, co, cst = _lin(op['idx'], slot)
            sd[p + '.icols'], sd[p + '.icoef'], sd[p + '.ic'] = c, co, cst
            names = list(op['cols'])
            tcols = G.tables[op['table']][0]
            sd[p + '.tcols'] = torch.tensor([tcols.index(n) for n in names], dtype=torch.long)
            sd[p + '.table'] = torch.tensor(list(op['table'].encode()), dtype=torch.uint8)
            sd[p + '.out'] = torch.tensor([nxt], dtype=torch.long)
            for n in names: w = op['cols'][n]; slot[w] = nxt; bounds[nxt] = (G.lo[w], G.hi[w]); nxt += 1
        elif kind == 'attn':
            c, co, cst = _lin(op['idx'], slot)
            sd[p + '.icols'], sd[p + '.icoef'], sd[p + '.ic'] = c, co, cst
            sd[p + '.out'] = torch.tensor([nxt], dtype=torch.long); slot[op['wire']] = nxt; bounds[nxt] = (G.lo[op['wire']], G.hi[op['wire']]); nxt += 1
        elif kind == 'memwrite':
            for nm in ['pos', 'val']:
                c, co, cst = _lin(op[nm], slot)
                sd[p + f'.{nm}cols'], sd[p + f'.{nm}coef'], sd[p + f'.{nm}c'] = c, co, cst
        elif kind == 'vocab':
            ins = list(op['ins'].values())
            cols = sorted({c for l in ins for c in l.t}); cpos = {c: i for i, c in enumerate(cols)}
            W = torch.zeros(len(ins), len(cols), dtype=torch.float64); b = torch.zeros(len(ins), dtype=torch.float64)
            for i, l in enumerate(ins):
                b[i] = l.c
                for c, v in l.t.items(): W[i, cpos[c]] = v
            sd[p + '.cols'] = torch.tensor([slot[c] for c in cols], dtype=torch.long); sd[p + '.W'] = W; sd[p + '.b'] = b
            sd[p + '.out'] = torch.tensor([nxt], dtype=torch.long)
            for nm, w in op['outs'].items(): slot[w] = nxt; bounds[nxt] = (G.lo[w], G.hi[w]); nxt += 1
        k += 1
    # outputs: next-state registers (and write results) as one linear map
    outs = prog['outs']
    lins = list(outs.values())
    cols = sorted({c for s in lins for c in s.lin.t}); cpos = {c: i for i, c in enumerate(cols)}
    r, cc, v, b = [], [], [], []
    for i, s in enumerate(lins):
        b.append(float(s.lin.c))
        for c, val in s.lin.t.items(): r.append(i); cc.append(cpos[c]); v.append(float(val))
    sd[f'{which}.out.cols'] = torch.tensor([slot[c] for c in cols], dtype=torch.long)
    sd[f'{which}.out.Wi'] = torch.tensor([r, cc], dtype=torch.long); sd[f'{which}.out.Wv'] = torch.tensor(v, dtype=torch.float64)
    sd[f'{which}.out.Wshape'] = torch.tensor([len(lins), len(cols)], dtype=torch.long)
    sd[f'{which}.out.b'] = torch.tensor(b, dtype=torch.float64)
    sd[f'{which}.nwires'] = torch.tensor([nxt], dtype=torch.long)
    # bounds of every wire (debug: runtime interval check)
    blo = torch.full((nxt,), -float('inf'), dtype=torch.float64); bhi = torch.full((nxt,), float('inf'), dtype=torch.float64)
    for i, nm in enumerate(n_in_names):
        w = G.inputs[nm]; blo[i] = G.lo[w]; bhi[i] = G.hi[w]
    for s, (lo, hi) in bounds.items(): blo[s] = lo; bhi[s] = hi
    sd[f'{which}.blo'] = blo; sd[f'{which}.bhi'] = bhi
    return list(outs.keys())


def export(progs, spec, tb, E, g, eps, F):
    """-> (state_dict, meta). meta = names only (registers / outputs), also stored as tensors of bytes."""
    sd = OrderedDict()
    regs = list(spec)
    sd['reg.init'] = torch.tensor([float(spec[n][2]) for n in regs], dtype=torch.float64)
    out_step = export_prog(progs['step'], 'step', regs + ['ids'], sd)
    out_write = export_prog(progs['write'], 'write', regs, sd)
    assert out_step == regs and out_write[:len(regs)] == regs
    wres = out_write[len(regs):]
    V = E.shape[0]
    FEAT, fcols = tb['FEAT']
    sd['tab.FEAT'] = FEAT.clone()                                    # float64 (exact integers)
    sd['tab.CODE2'] = tb['CODE2'][0].clone()
    sd['tab.ONEHOT'] = torch.eye(tb['NCODE'], dtype=torch.float64)
    sd['tab.PCOEF'] = tb['PCOEF'][0].clone()
    c = {n: i for i, n in enumerate(fcols)}
    sd['voc.TB'] = torch.stack([FEAT[:, c[n]] for n in ['COMMASET', 'STAR3SET', 'NN', 'STARTNL', 'EOSMASK', 'BULST', 'NL']], 1)
    pidx = FEAT[:, c['PIDX']].long(); lead = FEAT[:, c['LEAD']]
    P1 = torch.zeros(V, len(tb['POWS']), dtype=torch.float64); P1[torch.arange(V), pidx] = 1; P1 = P1 * (1 - lead)[:, None]
    sd['voc.POW1H'] = P1
    sd['voc.HTB'] = FEAT[:, c['HT']] + NEGBIG * (1 - FEAT[:, c['HASAN']])
    sd['voc.LEAD'] = lead.clone()
    sd['voc.E'] = E.detach().clone()                                 # copy of the host LM-head rows (bf16)
    sd['voc.g'] = g.detach().float().clone(); sd['voc.eps'] = torch.tensor(float(eps))
    sd['force.F'] = F                                                # (V+1, d) force rows, row V = 0
    meta = dict(regs=regs, wres=wres)
    sd['meta.regs'] = torch.tensor(list('\n'.join(regs).encode()), dtype=torch.uint8)
    sd['meta.wres'] = torch.tensor(list('\n'.join(wres).encode()), dtype=torch.uint8)
    return sd, meta


def _s(t): return bytes(t.tolist()).decode()


class Executor:
    """Runs the exported state_dict. Interface mirrors circuit.Circuit: reset(B), step(ids), write(h), report()."""

    def __init__(self, sd, device='cuda', check=False):
        self.dev = device; self.check = check
        self.meta = {k: v for k, v in sd.items() if k.endswith(('.out', '.table', '.Wshape', '.nwires'))}   # small ints, kept on CPU
        self.sd = {k: v.to(device) for k, v in sd.items()}
        self.regs = _s(sd['meta.regs']).split('\n'); self.wres = _s(sd['meta.wres']).split('\n')
        self.ri = {n: i for i, n in enumerate(self.regs)}
        self.progs = {w: self._ops(w) for w in ['step', 'write']}
        self._csr = {}
        self.o = {k: int(v[0]) for k, v in self.meta.items() if k.endswith('.out') and v.numel() == 1}
        self.tname = {k: _s(v) for k, v in self.meta.items() if k.endswith('.table')}
        self.nw = {k: int(v[0]) for k, v in self.meta.items() if k.endswith('.nwires')}
        self.tables = {k[4:]: v for k, v in self.sd.items() if k.startswith('tab.')}

    def _ops(self, which):
        keys = sorted({k.rsplit('.', 1)[0] for k in self.sd if k.startswith(which + '.') and k.split('.')[1].isdigit()})
        return [(k, k.split('.')[2]) for k in keys]

    # ---- primitives
    def _lin(self, X, p, pre=''):
        cols, coef, c = self.sd[p + f'.{pre}cols'], self.sd[p + f'.{pre}coef'], self.sd[p + f'.{pre}c']
        if cols.numel() == 0: return torch.full((X.shape[0],), float(c), dtype=X.dtype, device=X.device)
        return X[:, cols] @ coef + c

    def _sp(self, p):
        if p not in self._csr:
            sh = tuple(self.meta[p + '.Wshape'].tolist())
            self._csr[p] = torch.sparse_coo_tensor(self.sd[p + '.Wi'], self.sd[p + '.Wv'], sh).coalesce().to_dense() \
                if sh[0] * sh[1] <= 4_000_000 else torch.sparse_coo_tensor(self.sd[p + '.Wi'], self.sd[p + '.Wv'], sh).to_sparse_csr()
        return self._csr[p]

    def _mm(self, p, Xin):
        W = self._sp(p)
        if W.layout == torch.strided: return Xin @ W.T
        return torch.sparse.mm(W, Xin.T).T

    @staticmethod
    def _idx(v):
        return torch.round(v).long()

    def _attn(self, q):
        """hard attention: key = position j, score = -|j - q| (ReLU form), value = token id stored at j."""
        j = self._keys
        d = j[None, :] - q[:, None]
        score = -(torch.relu(d) + torch.relu(-d))
        a = score.argmax(-1)
        return self.mem.gather(1, a[:, None])[:, 0]

    def run(self, which, X, h=None):
        sd = self.sd
        for p, kind in self.progs[which]:
            if kind == 'relu':
                cols = sd[p + '.cols']; o = self.o[p + '.out']
                y = torch.relu(self._mm(p, X[:, cols]) + sd[p + '.b'])
                X[:, o:o + y.shape[1]] = y
            elif kind == 'gather':
                idx = self._idx(self._lin(X, p, 'i'))
                tab = self.tables[self.tname[p + '.table']]
                tc = sd[p + '.tcols']; o = self.o[p + '.out']
                X[:, o:o + tc.numel()] = tab[idx][:, tc]
            elif kind == 'attn':
                q = self._lin(X, p, 'i'); o = self.o[p + '.out']
                X[:, o] = self._attn(q)
            elif kind == 'memwrite':
                pos = self._idx(self._lin(X, p, 'pos')); val = self._lin(X, p, 'val')
                self.mem[torch.arange(X.shape[0], device=X.device), pos] = val
            elif kind == 'vocab':
                vin = X[:, sd[p + '.cols']] @ sd[p + '.W'].T + sd[p + '.b']
                top, blocked, alt = self._vocab(vin, h)
                o = self.o[p + '.out']
                X[:, o] = top.double(); X[:, o + 1] = blocked; X[:, o + 2] = alt.double()
        if self.check:
            lo, hi = sd[which + '.blo'], sd[which + '.bhi']
            n = int(sd[which + '.nwires'][0])
            bad = ((X[:, :n] < lo) | (X[:, :n] > hi) | (X[:, :n] != torch.round(X[:, :n]))).any(0)
            if bool(bad.any()):
                sl = torch.nonzero(bad)[:8, 0].tolist()
                info = [(q, X[:, q].min().item(), X[:, q].max().item(), lo[q].item(), hi[q].item(), self._origin(which, q)) for q in sl]
                raise RuntimeError(f'{which}: wire out of its interval / non-integer: {info}')
        y = self._mm(which + '.out', X[:, sd[which + '.out.cols']]) + sd[which + '.out.b']
        return y

    def _origin(self, which, q):
        for p, kind in self.progs[which]:
            if p + '.out' in self.sd:
                o = int(self.sd[p + '.out'][0])
                n = {'relu': lambda: int(self.sd[p + '.Wshape'][0]), 'gather': lambda: self.sd[p + '.tcols'].numel(), 'attn': lambda: 1, 'vocab': lambda: 3}[kind]()
                if o <= q < o + n: return (p, q - o)
        return ('input', q)

    def _vocab(self, vin, h):
        sd = self.sd
        coef, ymod, tgt, pf, hz0 = vin[:, 0:7], vin[:, 7:7 + 36], vin[:, 43:49], vin[:, 49], vin[:, 50]
        # vocab-plane layers (all exact integer arithmetic in float64)
        tb = coef @ sd['voc.TB'].T                                            # fixed tables x gates
        z = ymod @ sd['voc.POW1H'].T + sd['voc.HTB']                          # hcur*131^len(v) mod p + HT[v]   (< 2p)
        cond = hz0[:, None] @ (1 - sd['voc.LEAD'])[None] + sd['voc.LEAD']     # LEAD[v] or hcur == 0
        zm = z - HP * (torch.relu(z - HP + 1) - torch.relu(z - HP))          # mod p (one step)
        S = torch.zeros_like(z)
        for s in range(6):
            d = zm - tgt[:, s:s + 1]
            S = S + torch.relu(d + 1) - 2 * torch.relu(d) + torch.relu(d - 1)  # [cand == tgt_s]
        d = sd['voc.HTB'][None] - pf[:, None]
        eqpf = torch.relu(d + 1) - 2 * torch.relu(d) + torch.relu(d - 1)
        pre = tb + (S - torch.relu(S - 1)) + torch.relu(eqpf + cond - 1)
        blk = pre - torch.relu(pre - 1)
        self._last_blk = blk
        # readout (host final state through the stored LM-head copy)
        x = h.float(); x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + sd['voc.eps']) * sd['voc.g']
        lg = x.to(sd['voc.E'].dtype) @ sd['voc.E'].T
        top = lg.argmax(-1)
        blf = blk.float()
        blocked = blf.gather(1, top[:, None])[:, 0].double()
        alt = (lg.float() + blf * -1e30).argmax(-1)
        return top, blocked, alt

    # ---- circuit interface
    def reset(self, B):
        self.B = B
        self.state = self.sd['reg.init'][None].repeat(B, 1).clone()
        self.mem = torch.zeros(B, MAXMEM, dtype=torch.float64, device=self.dev)
        self._keys = torch.arange(MAXMEM, dtype=torch.float64, device=self.dev)

    def _X(self, which, extra=None):
        n = self.nw[which + '.nwires']
        X = torch.zeros(self.B, n, dtype=torch.float64, device=self.dev)
        R = len(self.regs)
        X[:, :R] = self.state
        if extra is not None: X[:, R] = extra
        return X

    @torch.no_grad()
    def step(self, ids):
        X = self._X('step', ids.double())
        self.state = self.run('step', X)

    @torch.no_grad()
    def write(self, h):
        X = self._X('write')
        y = self.run('write', X, h=h)
        R = len(self.regs)
        self.state = y[:, :R]
        self.wout = {n.lstrip('@'): y[:, R + i] for i, n in enumerate(self.wres)}
        widx = self._idx(self.wout['widx'])
        return self.sd['force.F'][widx]

    def reg(self, n): return self.state[:, self.ri[n]]
