"""Graph-capturable, per-slot executor of the compiled 14-rule IFEval implant (f14/execu.py export, FULL export: frozen
LM-head copy voc.E and force table force.F stored in the state_dict).

Same programs, same ops, same float64 arithmetic as execu.Executor (linear / ReLU / gather of fixed table rows / hard attention
over the token memory / vocab readout argmax); differences are only in where the state lives:
  - the registers (state) and the token memory (mem) are caller-owned buffers (rows of a slot pool), updated in place, so a
    whole decode step (implant + host) is one CUDA graph and finished rows can be compacted / new rows admitted;
  - step(..., active): rows with active == 0 keep their registers and memory unchanged (masked update) -- used to step
    prompts of different lengths without padding tokens;
  - every relu / output matrix is dense (no sparse kernels inside graphs);
  - write() returns the write row index widx (row V = zero row); the write vector is force.F[widx] (gathered by the caller).
"""
import os, sys, torch
import triton, triton.language as tl
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'f14'))
import execu as EX

FUSED_VOCAB = os.environ.get('IF_FUSED_VOCAB', '1') == '1'


@triton.jit
def _relu(x): return tl.maximum(x, 0.0)


@triton.jit
def _voc_kernel(LG, sLG, TBt, PWt, HTB, LEAD, VIN, sVIN, PM, PI, PB, AM, AI, B, V, NB, HPp,
                KT: tl.constexpr, KP: tl.constexpr, NS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """One (BM rows x BN vocab) tile of execu.Executor._vocab: the vocab-plane layers (float64, the same element-wise ops in
    the same order; every value is an integer < 2^53, so the products/sums are exact) and the block-local first-index argmax
    of the readout logits (top) and of the blocked-masked logits (alt). Partial results are reduced by the caller."""
    pm_ = tl.program_id(0); pn = tl.program_id(1)
    HP = tl.load(HPp)                                                       # float64 (a python float arg would be fp32)
    rows = pm_ * BM + tl.arange(0, BM); cl = tl.arange(0, BN); cols = pn * BN + cl
    rm = rows < B; cm = cols < V
    tb = tl.zeros((BM, BN), tl.float64)
    for k in tl.static_range(KT):
        c = tl.load(VIN + rows * sVIN + k, mask=rm, other=0.)
        t = tl.load(TBt + k * V + cols, mask=cm, other=0.)
        tb += c[:, None] * t[None, :]
    z = tl.zeros((BM, BN), tl.float64)
    for k in tl.static_range(KP):
        c = tl.load(VIN + rows * sVIN + KT + k, mask=rm, other=0.)
        t = tl.load(PWt + k * V + cols, mask=cm, other=0.)
        z += c[:, None] * t[None, :]
    htb = tl.load(HTB + cols, mask=cm, other=0.)
    z = z + htb[None, :]
    lead = tl.load(LEAD + cols, mask=cm, other=0.)
    pf = tl.load(VIN + rows * sVIN + KT + KP + NS, mask=rm, other=0.)
    hz0 = tl.load(VIN + rows * sVIN + KT + KP + NS + 1, mask=rm, other=0.)
    cond = hz0[:, None] * (1 - lead)[None, :] + lead[None, :]
    zm = z - HP * (_relu(z - HP + 1) - _relu(z - HP))
    S = tl.zeros((BM, BN), tl.float64)
    for s in tl.static_range(NS):
        tg = tl.load(VIN + rows * sVIN + KT + KP + s, mask=rm, other=0.)
        d = zm - tg[:, None]
        S = S + _relu(d + 1) - 2 * _relu(d) + _relu(d - 1)
    d = htb[None, :] - pf[:, None]
    eqpf = _relu(d + 1) - 2 * _relu(d) + _relu(d - 1)
    pre = tb + (S - _relu(S - 1)) + _relu(eqpf + cond - 1)
    blk = (pre - _relu(pre - 1)).to(tl.float32)
    lg = tl.load(LG + rows[:, None] * sLG + cols[None, :], mask=rm[:, None] & cm[None, :], other=float('-inf')).to(tl.float32)
    mx = tl.max(lg, 1)
    ix = tl.min(tl.where(lg == mx[:, None], cl[None, :], BN), 1)            # first index of the max
    bk = tl.sum(tl.where(cl[None, :] == ix[:, None], blk, 0.), 1)
    av = tl.where(cm[None, :], lg + blk * -1e30, float('-inf'))
    amx = tl.max(av, 1)
    aix = tl.min(tl.where(av == amx[:, None], cl[None, :], BN), 1)
    o = rows * NB + pn
    tl.store(PM + o, mx, mask=rm); tl.store(PI + o, ix + pn * BN, mask=rm); tl.store(PB + o, bk, mask=rm)
    tl.store(AM + o, amx, mask=rm); tl.store(AI + o, aix + pn * BN, mask=rm)


def _nm(lst, n):
    return lst.index('@' + n) if '@' + n in lst else lst.index(n)


class GExec(EX.Executor):
    def __init__(self, sd, device='cuda'):
        super().__init__(sd, device=device)
        assert not self.shared, 'the RL engine uses the full export (frozen LM-head copy / force table)'
        for which in ('step', 'write'):
            for p, kind in self.progs[which]:
                if kind == 'relu': self._dense(p)
            self._dense(which + '.out')
        self.R = len(self.regs)
        self.F = self.sd['force.F']                                    # (V+1, D) fp32, row V = 0
        self.V = self.F.shape[0] - 1
        self.iw, self.ia = _nm(self.wres, 'widx'), _nm(self.wres, 'active')
        self._keys = torch.arange(EX.MAXMEM, dtype=torch.float64, device=self.dev)
        self.init = self.sd['reg.init']
        sd = self.sd
        self._TBt = sd['voc.TB'].T.contiguous(); self._PWt = sd['voc.POW1H'].T.contiguous()
        self._KT, self._KP = self._TBt.shape[0], self._PWt.shape[0]
        assert self._KT == 7 and self._KP == 36, (self._KT, self._KP)
        self._HP = torch.tensor([float(EX.HP)], dtype=torch.float64, device=self.dev)
        self.fused_vocab = FUSED_VOCAB
        self.vcheck = False                                               # True: also run Executor._vocab, count mismatches
        self.vmm = torch.zeros((), dtype=torch.long, device=self.dev); self.vn = torch.zeros((), dtype=torch.long, device=self.dev)

    def _vocab(self, vin, h):
        if not self.fused_vocab: return super()._vocab(vin, h)
        sd = self.sd
        x = h.float(); x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + sd['voc.eps']) * sd['voc.g']
        lg = x.to(self.E.dtype) @ self.E.T                                # (B, V) bf16, the same op as Executor._vocab
        vin = vin.contiguous(); B, V = lg.shape; BM, BN = 16, 256; NB = triton.cdiv(V, BN)
        PM = torch.empty(B, NB, dtype=torch.float32, device=lg.device); AM = torch.empty_like(PM); PB = torch.empty_like(PM)
        PI = torch.empty(B, NB, dtype=torch.int32, device=lg.device); AI = torch.empty_like(PI)
        _voc_kernel[(triton.cdiv(B, BM), NB)](lg, lg.stride(0), self._TBt, self._PWt, sd['voc.HTB'], sd['voc.LEAD'], vin, vin.stride(0),
                                             PM, PI, PB, AM, AI, B, V, NB, self._HP, KT=self._KT, KP=self._KP, NS=6,
                                             BM=BM, BN=BN, num_warps=8)
        j = PM.argmax(1, keepdim=True)                                    # first block holding the max -> global first index
        top = PI.gather(1, j)[:, 0].long(); blocked = PB.gather(1, j)[:, 0].double()
        ja = AM.argmax(1, keepdim=True); alt = AI.gather(1, ja)[:, 0].long()
        if self.vcheck:
            t2, b2, a2 = super()._vocab(vin, h)
            self.vmm += ((top != t2) | (blocked != b2) | (alt != a2)).sum(); self.vn += B
        return top, blocked, alt

    def _lin(self, X, p, pre=''):                                       # no host sync (graph capture)
        cols, coef, c = self.sd[p + f'.{pre}cols'], self.sd[p + f'.{pre}coef'], self.sd[p + f'.{pre}c']
        if cols.numel() == 0: return c.to(X.dtype).expand(X.shape[0]).clone()
        return X[:, cols] @ coef + c

    def _dense(self, p):
        sh = tuple(self.meta[p + '.Wshape'].tolist())
        self._csr[p] = torch.sparse_coo_tensor(self.sd[p + '.Wi'], self.sd[p + '.Wv'], sh).coalesce().to_dense()

    def new_state(self, n):
        return self.init[None].repeat(n, 1).clone(), torch.zeros(n, EX.MAXMEM, dtype=torch.float64, device=self.dev)

    def _run(self, which, X, mem, h=None, active=None):
        sd = self.sd
        for p, kind in self.progs[which]:
            if kind == 'relu':
                cols = sd[p + '.cols']; o = self.o[p + '.out']
                y = torch.relu(X[:, cols] @ self._csr[p].T + sd[p + '.b'])
                X[:, o:o + y.shape[1]] = y
            elif kind == 'gather':
                idx = self._idx(self._lin(X, p, 'i'))
                tab = self.tables[self.tname[p + '.table']]
                tc = sd[p + '.tcols']; o = self.o[p + '.out']
                X[:, o:o + tc.numel()] = tab[idx][:, tc]
            elif kind == 'attn':
                q = self._lin(X, p, 'i'); o = self.o[p + '.out']
                d = self._keys[None, :] - q[:, None]
                a = (-(torch.relu(d) + torch.relu(-d))).argmax(-1)
                X[:, o] = mem.gather(1, a[:, None])[:, 0]
            elif kind == 'memwrite':
                pos = self._idx(self._lin(X, p, 'pos'))[:, None]; val = self._lin(X, p, 'val')[:, None]
                if active is not None: val = torch.where(active[:, None], val, mem.gather(1, pos))
                mem.scatter_(1, pos, val)
            elif kind == 'vocab':
                vin = X[:, sd[p + '.cols']] @ sd[p + '.W'].T + sd[p + '.b']
                top, blocked, alt = self._vocab(vin, h)
                o = self.o[p + '.out']
                X[:, o] = top.double(); X[:, o + 1] = blocked; X[:, o + 2] = alt.double()
        return X[:, sd[which + '.out.cols']] @ self._csr[which + '.out'].T + sd[which + '.out.b']

    def _Xin(self, which, state, extra=None):
        X = torch.zeros(state.shape[0], self.nw[which + '.nwires'], dtype=torch.float64, device=self.dev)
        X[:, :self.R] = state
        if extra is not None: X[:, self.R] = extra
        return X

    @torch.no_grad()
    def gstep(self, state, mem, ids, active=None):
        """one token per row (prompt or generated), in place."""
        y = self._run('step', self._Xin('step', state, ids.double()), mem, active=active)
        if active is None: state.copy_(y)
        else: state.copy_(torch.where(active[:, None], y, state))

    @torch.no_grad()
    def gwrite(self, state, mem, h):
        """h: host final hidden before the final norm (B, D). Updates the registers; returns (widx (B,) long, active (B,) bool)."""
        y = self._run('write', self._Xin('write', state), mem, h=h)
        state.copy_(y[:, :self.R])
        return self._idx(y[:, self.R + self.iw]), y[:, self.R + self.ia] > 0.5
