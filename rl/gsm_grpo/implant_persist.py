"""The calculator implant's decode step as ONE persistent Triton kernel (grid = #SMs, software grid barrier between the
circuit's layers), bit-identical outputs to CalcImplant.step.

Exact work skipping: stack 1's state S at a position is read only (a) later, if that position is a key ('=') token, or
(b) at that same position, if there is no key in the window (the hop then copies the current S). So stack 1 runs only for
rows whose new token is a key or whose window has no key; for the other rows the S ring slot is left stale and is never
read. The hop's softmax (BIG*key - GAMMA*dist) is exactly one-hot in fp32 and is computed as a select.
S_h and key_h are ring buffers (head h, shared by all rows); sym_h is shifted.

Phases (all in one launch): A reader + window + need flags (rows split over programs) | B stack 1, layer by layer (work
items = unit tile x 32-row chunk, round robin; skipped if no row needs it) | C hop | D stack 2 layer by layer | E writer."""
import numpy as np, torch, triton, triton.language as tl
from implant_triton import _comp

import os
BU, RB, KB = int(os.environ.get('P_BU', 32)), int(os.environ.get('P_RB', 16)), int(os.environ.get('P_KB', 8))
SPIN = os.environ.get('P_SPIN', 'volatile')
NOSKIP = int(os.environ.get('P_NOSKIP', 0)); FORCESKIP = int(os.environ.get('P_FORCESKIP', 0)); CG = int(os.environ.get('P_CG', 0))


BUD, KBD, DEEP_T = 2, 128, 96          # "deep" tiles: 2 units x 128 nonzeros per step, for units with fan-in > DEEP_T


def ell_layers(comp, wire_base_in, out_base):
    """ELL tiles per layer. Units sorted by fan-in; units with fan-in > DEEP_T go to deep tiles (BUD units, KBD-wide k
    steps: short serial chains for the 2,600-3,875-input sums), the rest to wide tiles (BU units, KB-wide k steps)."""
    n_in = comp['n_in']
    tiles, ell_i, ell_w, off, lay_start = [], [], [], 0, [0]
    p = n_in
    layers = comp['layers'] + [comp['out']]
    for li, (idx, W, b) in enumerate(layers):
        is_out = li == len(layers) - 1
        nz = (W != 0).sum(1)
        order = np.argsort(-nz, kind='stable')
        deep_units = [u for u in order if nz[u] > DEEP_T]; wide_units = [u for u in order if nz[u] <= DEEP_T]
        groups = [(deep_units[t:t + BUD], True) for t in range(0, len(deep_units), BUD)] + \
                 [(wide_units[t:t + BU], False) for t in range(0, len(wide_units), BU)]
        for us, deep in groups:
            bu = BUD if deep else BU
            k = int(max(1, nz[us].max()))
            I = np.zeros((k, bu), np.int32); Wv = np.zeros((k, bu), np.float32)
            dst = np.full(BU, -1, np.int32); bias = np.zeros(BU, np.float32)
            for jj, u in enumerate(us):
                cols = np.nonzero(W[u])[0]
                I[:len(cols), jj] = wire_base_in + idx[cols]; Wv[:len(cols), jj] = W[u, cols]
                dst[jj] = (out_base + u) if is_out else (wire_base_in + p + u); bias[jj] = b[u]
            tiles.append(dict(off=off, k=k, relu=int(not is_out) + 2 * int(deep), dst=dst, bias=bias))
            ell_i.append(I.ravel()); ell_w.append(Wv.ravel()); off += I.size
        lay_start.append(len(tiles))
        if not is_out: p += W.shape[0]
    return tiles, np.concatenate(ell_i), np.concatenate(ell_w), lay_start


@triton.jit
def _gsync(cnt, target):
    tl.debug_barrier()
    tl.atomic_add(cnt, 1, sem='release', scope='gpu')
    v = tl.load(cnt, volatile=True)
    while v < target:
        v = tl.load(cnt, volatile=True)
    tl.atomic_add(cnt, 0, sem='acquire', scope='gpu')
    tl.debug_barrier()


@triton.jit
def _stack(wires, B, B_pad, need, use_need: tl.constexpr, lay_start, n_lay, t_off, t_k, t_relu, t_dst, t_bias, ell_i, ell_w,
           cnt, nb, pid, NP, BU: tl.constexpr, RB: tl.constexpr, KB: tl.constexpr, CG: tl.constexpr,
           BUD_: tl.constexpr, KBD_: tl.constexpr):
    u = tl.arange(0, BU); rr = tl.arange(0, RB)
    n_rc = (B + RB - 1) // RB
    for l in range(n_lay):
        t0 = tl.load(lay_start + l); t1 = tl.load(lay_start + l + 1)
        n_it = (t1 - t0) * n_rc
        for it in range(pid, n_it, NP):
            t = t0 + it // n_rc; rc = it % n_rc
            rows = rc * RB + rr
            rm = rows < B
            if use_need:
                rm = rm & (tl.load(need + rows, mask=rows < B, other=0.0) > 0.5)
            go = tl.max(rm.to(tl.int32), axis=0) > 0                  # skip row chunks with no row needing stack 1
            fl = tl.load(t_relu + t)
            relu = fl % 2; deep = fl >= 2
            if go & deep:
                off = tl.load(t_off + t); k = tl.load(t_k + t)
                u2 = tl.arange(0, BUD_); kb2 = tl.arange(0, KBD_)
                dst2 = tl.load(t_dst + t * BU + u2)
                acc2 = tl.zeros((BUD_, RB), dtype=tl.float32) + tl.load(t_bias + t * BU + u2)[:, None]
                for k0 in range(0, k, KBD_):
                    kv = (k0 + kb2) < k
                    src = tl.load(ell_i + off + (k0 + kb2)[:, None] * BUD_ + u2[None, :], mask=kv[:, None], other=0)
                    w = tl.load(ell_w + off + (k0 + kb2)[:, None] * BUD_ + u2[None, :], mask=kv[:, None], other=0.0)
                    x = tl.load(wires + src[:, :, None] * B_pad + rows[None, None, :],
                                mask=rm[None, None, :] & (w != 0)[:, :, None], other=0.0)
                    acc2 += tl.sum(w[:, :, None] * x, axis=0)
                acc2 = tl.where(relu != 0, tl.maximum(acc2, 0.0), acc2)
                tl.store(wires + dst2[:, None] * B_pad + rows[None, :], acc2, mask=rm[None, :] & (dst2 >= 0)[:, None])
            if go & (~deep):
                off = tl.load(t_off + t); k = tl.load(t_k + t)
                dst = tl.load(t_dst + t * BU + u)
                acc = tl.zeros((BU, RB), dtype=tl.float32) + tl.load(t_bias + t * BU + u)[:, None]
                kb = tl.arange(0, KB)
                for k0 in range(0, k, KB):
                    kv = (k0 + kb) < k
                    src = tl.load(ell_i + off + (k0 + kb)[:, None] * BU + u[None, :], mask=kv[:, None], other=0)
                    w = tl.load(ell_w + off + (k0 + kb)[:, None] * BU + u[None, :], mask=kv[:, None], other=0.0)
                    x = tl.load(wires + src[:, :, None] * B_pad + rows[None, None, :],
                                mask=rm[None, None, :] & (w != 0)[:, :, None], other=0.0)
                    acc += tl.sum(w[:, :, None] * x, axis=0)
                acc = tl.where(relu != 0, tl.maximum(acc, 0.0), acc)
                tl.store(wires + dst[:, None] * B_pad + rows[None, :], acc, mask=rm[None, :] & (dst >= 0)[:, None])
        nb += 1
        _gsync(cnt, nb * NP)
    return nb


@triton.jit
def calc_persist_kernel(emb, rw, rb, symh, Sh, keyh, head, wires, need, nneed, cidx, cnt, B, B_pad,
                        s1_ls, s1_nl, s1_off, s1_k, s1_relu, s1_dst, s1_bias,
                        s2_ls, s2_nl, s2_off, s2_k, s2_relu, s2_dst, s2_bias,
                        ell_i, ell_w, Dw, out_e, out_w,
                        DM: tl.constexpr, NS: tl.constexpr, NSP: tl.constexpr, K: tl.constexpr, WN: tl.constexpr,
                        WNP: tl.constexpr, DS: tl.constexpr, KEY: tl.constexpr, NOUT: tl.constexpr, NOUTP: tl.constexpr,
                        S1_OUT: tl.constexpr, S2_BASE: tl.constexpr, S2_OUT: tl.constexpr,
                        RA: tl.constexpr, BU: tl.constexpr, RB: tl.constexpr, KB: tl.constexpr, DB: tl.constexpr, SB: tl.constexpr,
                        NOSKIP: tl.constexpr, CG: tl.constexpr, KNSP: tl.constexpr, FORCESKIP: tl.constexpr,
                        BUD_: tl.constexpr, KBD_: tl.constexpr):
    pid = tl.program_id(0); NP = tl.num_programs(0)
    h = tl.load(head)                                    # ring slot written this step
    j = tl.arange(0, NSP); jm = j < NS
    wi = tl.arange(0, WNP); wm = wi < WN
    nb = pid * 0
    # ---------------- A: reader, window, key ring, need flags (rows split over programs)
    for c in range(pid, (B + RA - 1) // RA, NP):
        rows = c * RA + tl.arange(0, RA); rmask = rows < B
        acc = tl.zeros((RA, NSP), dtype=tl.float32)
        for d0 in range(0, DM, DB):
            d = d0 + tl.arange(0, DB)
            e_ = tl.load(emb + rows[:, None] * DM + d[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
            w_ = tl.load(rw + j[:, None] * DM + d[None, :], mask=jm[:, None], other=0.0)
            acc += tl.sum(e_[:, None, :] * w_[None, :, :], axis=2)
        s = tl.minimum(tl.maximum(acc + tl.load(rb + j, mask=jm, other=0.0)[None, :], 0.0), 1.0)
        m2 = rmask[:, None] & jm[None, :]
        # sym_h shift: load the whole shifted window first, CTA barrier, then store (replicated layouts across warps
        # would otherwise race on the in-place shift)
        ii = tl.arange(0, KNSP); im = ii < (K - 1) * NS
        mm = rmask[:, None] & im[None, :]
        v = tl.load(symh + rows[:, None] * (K * NS) + ii[None, :] + NS, mask=mm, other=0.0)    # new[i] = old[i + NS]
        tl.debug_barrier()
        tl.store(symh + rows[:, None] * (K * NS) + ii[None, :], v, mask=mm)
        tl.store(symh + rows[:, None] * (K * NS) + (K - 1) * NS + j[None, :], s, mask=m2)
        kv = tl.sum(tl.where(j[None, :] == KEY, s, 0.0), axis=1)
        tl.store(keyh + rows * WN + h, kv, mask=rmask)
        kh = tl.load(keyh + rows[:, None] * WN + wi[None, :], mask=rmask[:, None] & wm[None, :], other=0.0)
        kh = tl.where(wi[None, :] == h, kv[:, None], kh)
        anyk = tl.max(tl.where(wm[None, :], kh, 0.0), axis=1)
        nd = tl.where((kv > 0.5) | (anyk < 0.5) | (NOSKIP != 0), 1.0, 0.0)
        nd = tl.where(FORCESKIP != 0, 0.0, nd)            # timing only (wrong outputs)
        tl.store(need + rows, nd, mask=rmask)
        ndm = rmask & (nd > 0.5)
        # compact index of each row that needs stack 1: stack 1 runs on columns 0..nneed-1 only
        ndi = ndm.to(tl.int32)
        base = tl.atomic_add(nneed, tl.sum(ndi, axis=0))           # one scalar atomic per program (no replicas)
        ci = base + tl.cumsum(ndi, axis=0) - ndi
        tl.store(cidx + rows, ci, mask=ndm)
        wpos = (K - 1 - ii // NS) * NS + ii % NS
        tl.store(wires + wpos[None, :] * B_pad + ci[:, None], v, mask=mm & ndm[:, None])
        tl.store(wires + (S2_BASE + wpos[None, :]) * B_pad + rows[:, None], v, mask=mm)
        tl.store(wires + j[None, :] * B_pad + ci[:, None], s, mask=m2 & ndm[:, None])
        tl.store(wires + (S2_BASE + j[None, :]) * B_pad + rows[:, None], s, mask=m2)
    nb += 1
    _gsync(cnt, nb * NP)
    # ---------------- B: stack 1, only rows that need it
    n1 = tl.load(nneed)
    if n1 > 0:
        nb = _stack(wires, n1, B_pad, need, False, s1_ls, s1_nl, s1_off, s1_k, s1_relu, s1_dst, s1_bias, ell_i, ell_w,
                    cnt, nb, pid, NP, BU, RB, KB, CG, BUD_, KBD_)
    # ---------------- C: write S into the ring (needed rows), hop select -> stack-2 input
    for c in range(pid, (B + RA - 1) // RA, NP):
        rows = c * RA + tl.arange(0, RA); rmask = rows < B
        nd = tl.load(need + rows, mask=rmask, other=0.0) > 0.5
        kh = tl.load(keyh + rows[:, None] * WN + wi[None, :], mask=rmask[:, None] & wm[None, :], other=0.0)
        dist = (h - wi + WN) % WN                        # 0 = newest
        best = tl.min(tl.where((kh > 0.5) & wm[None, :], dist[None, :], WN + 1), axis=1)
        selslot = tl.where(best <= WN, (h - best + WN) % WN, h)
        for c0 in range(0, DS, SB):
            cc = c0 + tl.arange(0, SB); cm = cc < DS
            cix = tl.load(cidx + rows, mask=rmask & nd, other=0)
            snew = tl.load(wires + (S1_OUT + cc[None, :]) * B_pad + cix[:, None], mask=(rmask & nd)[:, None] & cm[None, :], other=0.0)
            tl.store(Sh + (rows[:, None] * WN + h) * DS + cc[None, :], snew, mask=(rmask & nd)[:, None] & cm[None, :])
            cur = selslot == h
            old = tl.load(Sh + (rows[:, None] * WN + selslot[:, None]) * DS + cc[None, :], mask=rmask[:, None] & cm[None, :] & (~cur)[:, None], other=0.0)
            Cv = tl.where(cur[:, None], snew, old)
            tl.store(wires + (S2_BASE + K * NS + cc[None, :]) * B_pad + rows[:, None], Cv, mask=rmask[:, None] & cm[None, :])
    nb += 1
    _gsync(cnt, nb * NP)
    # ---------------- D: stack 2 (all rows)
    nb = _stack(wires, B, B_pad, need, False, s2_ls, s2_nl, s2_off, s2_k, s2_relu, s2_dst, s2_bias, ell_i, ell_w,
                cnt, nb, pid, NP, BU, RB, KB, CG, BUD_, KBD_)
    # ---------------- E: writer
    q = tl.arange(0, NOUTP); qm = q < NOUT
    for c in range(pid, (B + RA - 1) // RA, NP):
        rows = c * RA + tl.arange(0, RA); rmask = rows < B
        ev = tl.load(wires + (S2_OUT + q[None, :]) * B_pad + rows[:, None], mask=rmask[:, None] & qm[None, :], other=0.0)
        tl.store(out_e + rows[:, None] * NOUT + q[None, :], ev, mask=rmask[:, None] & qm[None, :])
        for d0 in range(0, DM, DB):
            d = d0 + tl.arange(0, DB)
            Dv = tl.load(Dw + q[:, None] * DM + d[None, :], mask=qm[:, None], other=0.0)
            wv = tl.sum(ev[:, :, None] * Dv[None, :, :], axis=1)
            tl.store(out_w + rows[:, None] * DM + d[None, :], wv, mask=rmask[:, None])


@triton.jit
def _advance_head(head, WN: tl.constexpr):
    h = tl.load(head)
    tl.store(head, (h + 1) % WN)


class PersistCalc:
    """Same interface as CalcImplant.step via CalcImplant(fused=...): step(emb, B) -> (e, write). Ring-buffer state:
    the owning CalcImplant must call ring_prime / ring_gather instead of its shift-layout versions."""

    def __init__(self, imp, B_max):
        self.imp = imp; M = imp.M; dev = M.reader_w.device
        self.B_pad = ((B_max + RB - 1) // RB) * RB
        NSK = M.K * M.NS
        c1 = _comp(M.stack1, NSK); c2 = _comp(M.stack2, NSK + imp.dS)
        S1_OUT = NSK + M.stack1.n_units; S2_BASE = S1_OUT + imp.dS
        S2_OUT = S2_BASE + NSK + imp.dS + M.stack2.n_units
        t1, i1, w1, ls1 = ell_layers(c1, 0, S1_OUT)
        t2, i2, w2, ls2 = ell_layers(c2, S2_BASE, S2_OUT)
        off2 = len(i1)
        T = lambda x, dt=torch.int32: torch.tensor(np.array(x), device=dev, dtype=dt)
        tab = lambda ts, o, ls: [T(ls), len(ls) - 1, T([t['off'] + o for t in ts]), T([t['k'] for t in ts]),
                                 T([t['relu'] for t in ts]), T(np.concatenate([t['dst'] for t in ts])),
                                 T(np.concatenate([t['bias'] for t in ts]), torch.float32)]
        self.s1 = tab(t1, 0, ls1); self.s2 = tab(t2, off2, ls2)
        self.ell_i = T(np.concatenate([i1, i2])); self.ell_w = T(np.concatenate([w1, w2]), torch.float32)
        self.wires = torch.zeros(S2_OUT + imp.n_out, self.B_pad, device=dev)
        self.need = torch.zeros(self.B_pad, device=dev); self.nneed = torch.zeros(1, device=dev, dtype=torch.int32)
        self.cidx = torch.zeros(self.B_pad, device=dev, dtype=torch.int32)
        self.cnt = torch.zeros(1, device=dev, dtype=torch.int32)
        self.head = torch.zeros(1, device=dev, dtype=torch.int32)
        self.consts = dict(S1_OUT=S1_OUT, S2_BASE=S2_BASE, S2_OUT=S2_OUT)
        self.e = torch.zeros(B_max, imp.n_out, device=dev); self.w = torch.zeros(B_max, M.out_dirs.shape[1], device=dev)
        self.rw = M.reader_w.float().contiguous(); self.rb = M.reader_b.float().contiguous(); self.Dw = imp.D.contiguous()
        self.NP = int(os.environ.get('P_NP', 0)) or torch.cuda.get_device_properties(dev).multi_processor_count

    def step(self, emb, B):
        imp, M = self.imp, self.imp.M
        self.cnt.zero_(); self.nneed.zero_()
        calc_persist_kernel[(self.NP,)](emb, self.rw, self.rb, imp.sym_h, imp.S_h, imp.key_h, self.head, self.wires,
                                        self.need, self.nneed, self.cidx, self.cnt, B, self.B_pad, *self.s1, *self.s2,
                                        self.ell_i, self.ell_w, self.Dw, self.e, self.w,
                                        DM=emb.shape[1], NS=M.NS, NSP=32, K=M.K, WN=M.W, WNP=64, DS=imp.dS, KEY=M.key,
                                        NOUT=imp.n_out, NOUTP=16, **self.consts, RA=4, BU=BU, RB=RB, KB=KB, DB=64, SB=64, NOSKIP=NOSKIP, CG=CG, FORCESKIP=FORCESKIP, BUD_=BUD, KBD_=KBD, KNSP=triton.next_power_of_2(M.K * M.NS),
                                        num_warps=8)
        _advance_head[(1,)](self.head, WN=M.W)
        return self.e[:B], self.w[:B]

    # ring layout helpers: logical index i (0 = oldest .. W-1 = newest) lives at slot (head + i) % W, where head = the
    # slot the NEXT step writes (so the newest entry, i = W-1, is at head - 1).
    def ring_index(self):
        h = int(self.head.item()); W = self.imp.W
        return torch.tensor([(h + i) % W for i in range(W)], device=self.head.device)
