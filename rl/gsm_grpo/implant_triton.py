"""The calculator implant's decode step as ONE Triton kernel: reader + window shift + stack 1 (32 ReLU layers) +
hard-attention hop + stack 2 (20 layers) + writer, fused. Bit-identical to CalcImplant.step (all circuit wires are
small integers, exact in fp32 in any summation order; the hop's softmax is exactly one-hot in fp32, so it is computed as
a select of the most recent key position, else the current one).

Sparse layers are stored as ELL tiles: units of a layer sorted by fan-in, BLOCK_U units per tile, each tile padded to
its own max fan-in. One program = RB batch rows; it walks every tile of both stacks in order, with a CTA barrier between
layers. Wires live in a global scratch laid out (wire, row) so a gather for RB rows is contiguous."""
import numpy as np, torch, triton, triton.language as tl

BLOCK_U = 64


def ell_tiles(comp, wire_base_in, out_base):
    """comp = compiled net (layers: (idx, dense W, b)), inputs at wires [wire_base_in, +n_in), layer-l units after them
    in order (== ReLUNet.forward), output-layer units at out_base + r. -> (tiles, ell_idx, ell_w, n_wires_of_this_stack)."""
    n_in = comp['n_in']
    tiles, ell_i, ell_w, off = [], [], [], 0
    p = n_in
    layers = comp['layers'] + [comp['out']]
    for li, (idx, W, b) in enumerate(layers):
        is_out = li == len(layers) - 1
        nz = (W != 0).sum(1)
        order = np.argsort(-nz, kind='stable')
        for t0 in range(0, len(order), BLOCK_U):
            us = order[t0:t0 + BLOCK_U]
            k = int(max(1, nz[us].max()))
            I = np.zeros((k, BLOCK_U), np.int32); Wv = np.zeros((k, BLOCK_U), np.float32)
            dst = np.full(BLOCK_U, -1, np.int32); bias = np.zeros(BLOCK_U, np.float32)
            for jj, u in enumerate(us):
                cols = np.nonzero(W[u])[0]
                I[:len(cols), jj] = wire_base_in + idx[cols]
                Wv[:len(cols), jj] = W[u, cols]
                dst[jj] = (out_base + u) if is_out else (wire_base_in + p + u)
                bias[jj] = b[u]
            tiles.append(dict(off=off, k=k, lay=li, relu=int(not is_out), dst=dst, bias=bias))
            ell_i.append(I.ravel()); ell_w.append(Wv.ravel()); off += I.size
        if not is_out: p += W.shape[0]
    return tiles, np.concatenate(ell_i), np.concatenate(ell_w), p


@triton.jit
def _sparse_stack(wires, B_pad, tile_off, tile_k, tile_layer, tile_relu, tile_dst, tile_bias, ell_i, ell_w,
                  t_begin, t_end, rows, rmask, RB: tl.constexpr, BU: tl.constexpr):
    u = tl.arange(0, BU)
    for t in range(t_begin, t_end):
        tl.debug_barrier()
        off = tl.load(tile_off + t); k = tl.load(tile_k + t); relu = tl.load(tile_relu + t)
        dst = tl.load(tile_dst + t * BU + u)
        acc = tl.zeros((BU, RB), dtype=tl.float32) + tl.load(tile_bias + t * BU + u)[:, None]
        for kk in range(k):
            src = tl.load(ell_i + off + kk * BU + u)
            w = tl.load(ell_w + off + kk * BU + u)
            x = tl.load(wires + src[:, None] * B_pad + rows[None, :], mask=rmask[None, :] & (w != 0)[:, None], other=0.0)
            acc += w[:, None] * x
        acc = tl.where(relu != 0, tl.maximum(acc, 0.0), acc)
        tl.store(wires + dst[:, None] * B_pad + rows[None, :], acc, mask=rmask[None, :] & (dst >= 0)[:, None])
    tl.debug_barrier()


@triton.jit
def calc_step_kernel(emb, rw, rb, symh, Sh, keyh, wires, B, B_pad,
                     t1_off, t1_k, t1_lay, t1_relu, t1_dst, t1_bias, n_t1,
                     t2_off, t2_k, t2_lay, t2_relu, t2_dst, t2_bias, n_t2,
                     ell_i, ell_w, Dw, out_e, out_w,
                     DM: tl.constexpr, NS: tl.constexpr, NSP: tl.constexpr, K: tl.constexpr, WN: tl.constexpr,
                     WNP: tl.constexpr, DS: tl.constexpr, KEY: tl.constexpr, NOUT: tl.constexpr, NOUTP: tl.constexpr,
                     S1_OUT: tl.constexpr, S2_BASE: tl.constexpr, S2_OUT: tl.constexpr,
                     RB: tl.constexpr, BU: tl.constexpr, DB: tl.constexpr, SB: tl.constexpr):
    pid = tl.program_id(0)
    rr = tl.arange(0, RB); rows = pid * RB + rr; rmask = rows < B
    # ---- 1. reader: s = clamp(emb @ rw^T + rb, 0, 1)  (RB, NSP)
    j = tl.arange(0, NSP); jm = j < NS
    acc = tl.zeros((RB, NSP), dtype=tl.float32)
    for d0 in range(0, DM, DB):
        d = d0 + tl.arange(0, DB)
        e_ = tl.load(emb + rows[:, None] * DM + d[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)   # (RB,DB)
        w_ = tl.load(rw + j[:, None] * DM + d[None, :], mask=jm[:, None], other=0.0)                        # (NSP,DB)
        acc += tl.sum(e_[:, None, :] * w_[None, :, :], axis=2)
    s = tl.minimum(tl.maximum(acc + tl.load(rb + j, mask=jm, other=0.0)[None, :], 0.0), 1.0)
    # ---- 2. window: symh (B,K,NS) oldest first; shift left by one, append s. win[o*NS+j] = new[K-1-o, j] -> wires[0:K*NS]
    for o in range(K - 1):          # new[o] = old[o+1]
        v = tl.load(symh + rows[:, None] * (K * NS) + (o + 1) * NS + j[None, :], mask=rmask[:, None] & jm[None, :], other=0.0)
        tl.store(symh + rows[:, None] * (K * NS) + o * NS + j[None, :], v, mask=rmask[:, None] & jm[None, :])
        tl.store(wires + ((K - 1 - o) * NS + j[None, :]) * B_pad + rows[:, None], v, mask=rmask[:, None] & jm[None, :])
    tl.store(symh + rows[:, None] * (K * NS) + (K - 1) * NS + j[None, :], s, mask=rmask[:, None] & jm[None, :])
    tl.store(wires + j[None, :] * B_pad + rows[:, None], s, mask=rmask[:, None] & jm[None, :])
    # stack-2 input copy of the window (wires[S2_BASE : S2_BASE+K*NS])
    for o in range(K - 1):
        v = tl.load(symh + rows[:, None] * (K * NS) + o * NS + j[None, :], mask=rmask[:, None] & jm[None, :], other=0.0)
        tl.store(wires + (S2_BASE + (K - 1 - o) * NS + j[None, :]) * B_pad + rows[:, None], v, mask=rmask[:, None] & jm[None, :])
    tl.store(wires + (S2_BASE + j[None, :]) * B_pad + rows[:, None], s, mask=rmask[:, None] & jm[None, :])
    tl.debug_barrier()
    # ---- 3. stack 1 -> S (DS values at wires[S1_OUT + r])
    _sparse_stack(wires, B_pad, t1_off, t1_k, t1_lay, t1_relu, t1_dst, t1_bias, ell_i, ell_w, 0, n_t1, rows, rmask, RB, BU)
    # ---- 4. hop: shift S_h (B,WN,DS) and key_h (B,WN); C = S_h[most recent key] (else current)
    kv = tl.sum(tl.where(j[None, :] == KEY, s, 0.0), axis=1)                                 # (RB,) key flag of new token
    wi = tl.arange(0, WNP); wm = wi < WN
    kh = tl.load(keyh + rows[:, None] * WN + wi[None, :] + 1, mask=rmask[:, None] & (wi[None, :] < WN - 1), other=0.0)
    kh = tl.where(wi[None, :] == WN - 1, kv[:, None], kh)
    tl.store(keyh + rows[:, None] * WN + wi[None, :], kh, mask=rmask[:, None] & wm[None, :])
    last = tl.max(tl.where((kh > 0.5) & wm[None, :], wi[None, :], -1), axis=1)
    sel = tl.where(last >= 0, last, WN - 1)                                                   # (RB,)
    for c0 in range(0, DS, SB):
        c = c0 + tl.arange(0, SB); cm = c < DS
        for w in range(WN - 1):
            v = tl.load(Sh + (rows[:, None] * WN + w + 1) * DS + c[None, :], mask=rmask[:, None] & cm[None, :], other=0.0)
            tl.store(Sh + (rows[:, None] * WN + w) * DS + c[None, :], v, mask=rmask[:, None] & cm[None, :])
        snew = tl.load(wires + (S1_OUT + c[None, :]) * B_pad + rows[:, None], mask=rmask[:, None] & cm[None, :], other=0.0)
        tl.store(Sh + (rows[:, None] * WN + WN - 1) * DS + c[None, :], snew, mask=rmask[:, None] & cm[None, :])
    tl.debug_barrier()
    for c0 in range(0, DS, SB):
        c = c0 + tl.arange(0, SB); cm = c < DS
        Cv = tl.load(Sh + (rows[:, None] * WN + sel[:, None]) * DS + c[None, :], mask=rmask[:, None] & cm[None, :], other=0.0)
        tl.store(wires + (S2_BASE + K * NS + c[None, :]) * B_pad + rows[:, None], Cv, mask=rmask[:, None] & cm[None, :])
    tl.debug_barrier()
    # ---- 5. stack 2 -> e (NOUT values at wires[S2_OUT + r])
    _sparse_stack(wires, B_pad, t2_off, t2_k, t2_lay, t2_relu, t2_dst, t2_bias, ell_i, ell_w, 0, n_t2, rows, rmask, RB, BU)
    # ---- 6. writer: e -> out_e, write = e @ Dw  (NOUT, DM)
    q = tl.arange(0, NOUTP); qm = q < NOUT
    ev = tl.load(wires + (S2_OUT + q[None, :]) * B_pad + rows[:, None], mask=rmask[:, None] & qm[None, :], other=0.0)  # (RB,NOUTP)
    tl.store(out_e + rows[:, None] * NOUT + q[None, :], ev, mask=rmask[:, None] & qm[None, :])
    for d0 in range(0, DM, DB):
        d = d0 + tl.arange(0, DB)
        Dv = tl.load(Dw + q[:, None] * DM + d[None, :], mask=qm[:, None], other=0.0)          # (NOUTP,DB)
        wv = tl.sum(ev[:, :, None] * Dv[None, :, :], axis=1)
        tl.store(out_w + rows[:, None] * DM + d[None, :], wv, mask=rmask[:, None])


class FusedCalc:
    """Drop-in for CalcImplant.step (same static state tensors), one kernel launch per decode step."""

    def __init__(self, imp, B_max, RB=4):
        self.imp = imp; M = imp.M; dev = M.reader_w.device
        self.RB = RB; self.B_pad = ((B_max + RB - 1) // RB) * RB
        NSK = M.K * M.NS
        c1 = _comp(M.stack1, NSK); c2 = _comp(M.stack2, NSK + imp.dS)
        # wire map: stack1 [0, n1) | stack1 outputs S [S1_OUT, +DS) | stack2 [S2_BASE, +n2) | stack2 outputs e [S2_OUT, +NOUT)
        n1 = NSK + M.stack1.n_units
        S1_OUT = n1; S2_BASE = S1_OUT + imp.dS
        t1, i1, w1, _ = ell_tiles(c1, 0, S1_OUT)
        S2_OUT = S2_BASE + NSK + imp.dS + M.stack2.n_units
        t2, i2, w2, _ = ell_tiles(c2, S2_BASE, S2_OUT)
        n_w = S2_OUT + imp.n_out
        off2 = len(i1)
        T = lambda x, dt=torch.int32: torch.tensor(np.array(x), device=dev, dtype=dt)
        tab = lambda ts, o: [T([t['off'] + o for t in ts]), T([t['k'] for t in ts]), T([t['lay'] for t in ts]),
                             T([t['relu'] for t in ts]), T(np.concatenate([t['dst'] for t in ts])),
                             T(np.concatenate([t['bias'] for t in ts]), torch.float32), len(ts)]
        self.t1 = tab(t1, 0); self.t2 = tab(t2, off2)
        self.ell_i = T(np.concatenate([i1, i2])); self.ell_w = T(np.concatenate([w1, w2]), torch.float32)
        self.wires = torch.zeros(n_w, self.B_pad, device=dev)
        self.consts = dict(S1_OUT=S1_OUT, S2_BASE=S2_BASE, S2_OUT=S2_OUT)
        self.e = torch.zeros(B_max, imp.n_out, device=dev); self.w = torch.zeros(B_max, M.out_dirs.shape[1], device=dev)
        self.rw = M.reader_w.float().contiguous(); self.rb = M.reader_b.float().contiguous(); self.Dw = imp.D.contiguous()
        self.n_ell = len(self.ell_i)

    def step(self, emb, B):
        imp, M = self.imp, self.imp.M
        grid = ((B + self.RB - 1) // self.RB,)
        calc_step_kernel[grid](emb, self.rw, self.rb, imp.sym_h, imp.S_h, imp.key_h, self.wires, B, self.B_pad,
                               *self.t1, *self.t2, self.ell_i, self.ell_w, self.Dw, self.e, self.w,
                               DM=emb.shape[1], NS=M.NS, NSP=32, K=M.K, WN=M.W, WNP=64, DS=imp.dS, KEY=M.key,
                               NOUT=imp.n_out, NOUTP=16, **self.consts, RB=self.RB, BU=BLOCK_U, DB=64, SB=64,
                               num_warps=4)
        return self.e[:B], self.w[:B]


def _comp(net, n_in):
    """Rebuild the compiled dict (layers: (idx, dense W, b)) from a ReLUNet's buffers."""
    layers = []
    for l in range(net.L + 1):
        idx = getattr(net, f'i{l}').cpu().numpy(); b = getattr(net, f'b{l}').cpu().numpy()
        if net.sparse:
            W = np.zeros(getattr(net, f'shape{l}'), np.float32)
            wi = getattr(net, f'wi{l}').cpu().numpy(); W[wi[0], wi[1]] = getattr(net, f'wv{l}').cpu().numpy()
        else:
            W = getattr(net, f'w{l}').cpu().numpy()
        layers.append((idx, W, b))
    return dict(layers=layers[:-1], out=layers[-1], n_in=n_in)
