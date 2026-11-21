"""Group-packed training attention (Triton, fwd + bwd) for the fast trainer.

Layout: packed tokens t = 0..L-1 in groups [question, seg_1, seg_2, ...]; per token grp (group id, pads -1) and doc
(segment id; -1 on question tokens and pads). A query sees: every prefix key (Lp shared keys, constant: no dK/dV)
+ packed key j iff grp[j] == grp[i] and j <= i and (doc[j] < 0 or doc[j] == doc[i]).
(= the reference's one-document-per-segment packing: the forget's -1e4 bias is an exact 0 after the fp32 softmax.)

Tensors are token-major: q (L, H, HD), k / v (L, KVH, HD), prefix Kp / Vp (Lp, KVH, HD), out (L, H, HD). No transposes.
Per query block (fwd, dq) the visible packed keys lie in two ranges R1 (its first row's question) and R2 = [min doc
start in the block, block end); per key block (dk/dv) the visible queries lie in [block start, max visible end).
"""
import torch, triton, triton.language as tl

LOG2E = 1.4426950408889634


@triton.jit
def _fwd_range(acc, l_i, m_i, q, gq, dq, offs_m, K, V, G, D, lo, hi, kvh, skt, skh, qk_scale,
               MASKED: tl.constexpr, BN: tl.constexpr, HD: tl.constexpr):
    offs_d = tl.arange(0, HD)
    for n0 in range(lo, hi, BN):
        offs_n = n0 + tl.arange(0, BN)
        nm = offs_n < hi
        kT = tl.load(K + offs_n[None, :] * skt + kvh * skh + offs_d[:, None], mask=nm[None, :], other=0.0)
        s = tl.dot(q, kT) * qk_scale
        if MASKED:
            gk = tl.load(G + offs_n, mask=nm, other=-9)
            dk = tl.load(D + offs_n, mask=nm, other=-9)
            vis = (gk[None, :] == gq[:, None]) & (offs_n[None, :] <= offs_m[:, None]) & ((dk[None, :] < 0) | (dk[None, :] == dq[:, None]))
            vis = vis & nm[None, :]
        else:
            vis = nm[None, :] & (offs_m[:, None] >= 0)
        s = tl.where(vis, s, float('-inf'))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float('-inf'), 0.0, m_new)
        p = tl.math.exp2(s - m_safe[:, None])
        alpha = tl.math.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + offs_n[:, None] * skt + kvh * skh + offs_d[None, :], mask=nm[:, None], other=0.0)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new
    return acc, l_i, m_i


@triton.jit
def _attn_fwd(Q, Kp, Vp, K, V, O, LSE, G, D, R1s, R1e, R2s, Lp, L, qk_scale,
              BM: tl.constexpr, BN: tl.constexpr, HD: tl.constexpr, H: tl.constexpr, KVH: tl.constexpr):
    mb = tl.program_id(0); h = tl.program_id(1); kvh = h // (H // KVH)
    offs_m = mb * BM + tl.arange(0, BM); offs_d = tl.arange(0, HD)
    mm = offs_m < L
    q = tl.load(Q + offs_m[:, None] * (H * HD) + h * HD + offs_d[None, :], mask=mm[:, None], other=0.0)
    gq = tl.load(G + offs_m, mask=mm, other=-5); dq = tl.load(D + offs_m, mask=mm, other=-5)
    m_i = tl.full([BM], float('-inf'), tl.float32); l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, HD], tl.float32)
    acc, l_i, m_i = _fwd_range(acc, l_i, m_i, q, gq, dq, offs_m, Kp, Vp, G, D, 0, Lp, kvh, KVH * HD, HD, qk_scale, False, BN, HD)
    r1s = tl.load(R1s + mb); r1e = tl.load(R1e + mb); r2s = tl.load(R2s + mb); r2e = tl.minimum((mb + 1) * BM, L)
    acc, l_i, m_i = _fwd_range(acc, l_i, m_i, q, gq, dq, offs_m, K, V, G, D, r1s, r1e, kvh, KVH * HD, HD, qk_scale, True, BN, HD)
    acc, l_i, m_i = _fwd_range(acc, l_i, m_i, q, gq, dq, offs_m, K, V, G, D, r2s, r2e, kvh, KVH * HD, HD, qk_scale, True, BN, HD)
    l_s = tl.where(l_i == 0.0, 1.0, l_i)
    o = acc / l_s[:, None]
    tl.store(O + offs_m[:, None] * (H * HD) + h * HD + offs_d[None, :], o.to(O.dtype.element_ty), mask=mm[:, None])
    tl.store(LSE + h * L + offs_m, m_i + tl.math.log2(l_s), mask=mm)


@triton.jit
def _dq_range(dqa, q, do, lse, delta, gq, dq, offs_m, K, V, G, D, lo, hi, kvh, skt, skh, qk_scale,
              MASKED: tl.constexpr, BN: tl.constexpr, HD: tl.constexpr):
    offs_d = tl.arange(0, HD)
    for n0 in range(lo, hi, BN):
        offs_n = n0 + tl.arange(0, BN)
        nm = offs_n < hi
        kT = tl.load(K + offs_n[None, :] * skt + kvh * skh + offs_d[:, None], mask=nm[None, :], other=0.0)
        vT = tl.load(V + offs_n[None, :] * skt + kvh * skh + offs_d[:, None], mask=nm[None, :], other=0.0)
        s = tl.dot(q, kT) * qk_scale
        if MASKED:
            gk = tl.load(G + offs_n, mask=nm, other=-9)
            dk = tl.load(D + offs_n, mask=nm, other=-9)
            vis = (gk[None, :] == gq[:, None]) & (offs_n[None, :] <= offs_m[:, None]) & ((dk[None, :] < 0) | (dk[None, :] == dq[:, None]))
            vis = vis & nm[None, :]
        else:
            vis = nm[None, :] & (offs_m[:, None] >= 0)
        p = tl.where(vis, tl.math.exp2(s - lse[:, None]), 0.0)
        dp = tl.dot(do, vT)
        ds = p * (dp - delta[:, None])
        dqa = tl.dot(ds.to(kT.dtype), tl.trans(kT), dqa)
    return dqa


@triton.jit
def _attn_bwd_dq(Q, Kp, Vp, K, V, O, DO, LSE, DELTA, DQ, G, D, R1s, R1e, R2s, Lp, L, qk_scale, sm_scale,
                 BM: tl.constexpr, BN: tl.constexpr, HD: tl.constexpr, H: tl.constexpr, KVH: tl.constexpr):
    mb = tl.program_id(0); h = tl.program_id(1); kvh = h // (H // KVH)
    offs_m = mb * BM + tl.arange(0, BM); offs_d = tl.arange(0, HD)
    mm = offs_m < L
    ptr = offs_m[:, None] * (H * HD) + h * HD + offs_d[None, :]
    q = tl.load(Q + ptr, mask=mm[:, None], other=0.0)
    do = tl.load(DO + ptr, mask=mm[:, None], other=0.0)
    o = tl.load(O + ptr, mask=mm[:, None], other=0.0)
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), 1)
    tl.store(DELTA + h * L + offs_m, delta, mask=mm)
    lse = tl.load(LSE + h * L + offs_m, mask=mm, other=0.0)
    gq = tl.load(G + offs_m, mask=mm, other=-5); dq = tl.load(D + offs_m, mask=mm, other=-5)
    dqa = tl.zeros([BM, HD], tl.float32)
    dqa = _dq_range(dqa, q, do, lse, delta, gq, dq, offs_m, Kp, Vp, G, D, 0, Lp, kvh, KVH * HD, HD, qk_scale, False, BN, HD)
    r1s = tl.load(R1s + mb); r1e = tl.load(R1e + mb); r2s = tl.load(R2s + mb); r2e = tl.minimum((mb + 1) * BM, L)
    dqa = _dq_range(dqa, q, do, lse, delta, gq, dq, offs_m, K, V, G, D, r1s, r1e, kvh, KVH * HD, HD, qk_scale, True, BN, HD)
    dqa = _dq_range(dqa, q, do, lse, delta, gq, dq, offs_m, K, V, G, D, r2s, r2e, kvh, KVH * HD, HD, qk_scale, True, BN, HD)
    tl.store(DQ + ptr, (dqa * sm_scale).to(DQ.dtype.element_ty), mask=mm[:, None])


@triton.jit
def _attn_bwd_dkdv(Q, K, V, DO, LSE, DELTA, DK, DV, G, D, QE, L, qk_scale, sm_scale,
                   BM: tl.constexpr, BN: tl.constexpr, HD: tl.constexpr, H: tl.constexpr, KVH: tl.constexpr):
    nb = tl.program_id(0); kvh = tl.program_id(1)
    offs_n = nb * BN + tl.arange(0, BN); offs_d = tl.arange(0, HD)
    nm = offs_n < L
    kp = offs_n[:, None] * (KVH * HD) + kvh * HD + offs_d[None, :]
    k = tl.load(K + kp, mask=nm[:, None], other=0.0)
    v = tl.load(V + kp, mask=nm[:, None], other=0.0)
    gk = tl.load(G + offs_n, mask=nm, other=-9); dk = tl.load(D + offs_n, mask=nm, other=-9)
    dka = tl.zeros([BN, HD], tl.float32); dva = tl.zeros([BN, HD], tl.float32)
    lo = nb * BN; hi = tl.load(QE + nb)
    nmb = tl.maximum(hi - lo + BM - 1, 0) // BM
    # one flat loop over (head-in-group, query block): nested dot loops trip TritonGPUHoistTMEMAlloc on sm_100 (triton 3.4)
    for it in range(0, nmb * (H // KVH)):
        g = it // nmb
        h = kvh * (H // KVH) + g
        offs_m = lo + (it - g * nmb) * BM + tl.arange(0, BM)
        mm = offs_m < hi
        ptr = offs_m[:, None] * (H * HD) + h * HD + offs_d[None, :]
        q = tl.load(Q + ptr, mask=mm[:, None], other=0.0)
        do = tl.load(DO + ptr, mask=mm[:, None], other=0.0)
        lse = tl.load(LSE + h * L + offs_m, mask=mm, other=0.0)
        delta = tl.load(DELTA + h * L + offs_m, mask=mm, other=0.0)
        gq = tl.load(G + offs_m, mask=mm, other=-5); dq = tl.load(D + offs_m, mask=mm, other=-5)
        sT = tl.dot(k, tl.trans(q)) * qk_scale                                   # (BN, BM)
        vis = (gk[:, None] == gq[None, :]) & (offs_n[:, None] <= offs_m[None, :]) & ((dk[:, None] < 0) | (dk[:, None] == dq[None, :]))
        vis = vis & mm[None, :] & nm[:, None]
        pT = tl.where(vis, tl.math.exp2(sT - lse[None, :]), 0.0)
        dva = tl.dot(pT.to(do.dtype), do, dva)
        dpT = tl.dot(v, tl.trans(do))
        dsT = pT * (dpT - delta[None, :])
        dka = tl.dot(dsT.to(q.dtype), q, dka)
    tl.store(DK + kp, (dka * sm_scale).to(DK.dtype.element_ty), mask=nm[:, None])
    tl.store(DV + kp, dva.to(DV.dtype.element_ty), mask=nm[:, None])


# ----------------------------------------------------------------------------------------------------- metadata
CFG = dict(FBM=128, FBN=64, FW=8, FS=2, QBM=64, QBN=64, QW=4, QS=2, KBM=64, KBN=64, KW=4, KS=2)


def meta(grp, doc, L, cfg=CFG):
    """grp / doc: int32 (L,) on the GPU (layout above). Returns the per-block ranges for the kernels."""
    dev = grp.device
    t = torch.arange(L, device=dev, dtype=torch.int32)
    isq = doc < 0
    # group start / question end / group end, doc start / doc end, per token
    newg = torch.ones(L, dtype=torch.bool, device=dev); newg[1:] = grp[1:] != grp[:-1]
    gs = torch.cummax(torch.where(newg, t, 0), 0).values
    newd = torch.ones(L, dtype=torch.bool, device=dev); newd[1:] = (doc[1:] != doc[:-1]) | newg[1:]
    ds = torch.cummax(torch.where(newd, t, 0), 0).values
    rev = lambda x: torch.flip(x, [0])
    endg = torch.ones(L, dtype=torch.bool, device=dev); endg[:-1] = newg[1:]
    ge = rev(torch.cummin(rev(torch.where(endg, t + 1, L)), 0).values)
    endd = torch.ones(L, dtype=torch.bool, device=dev); endd[:-1] = newd[1:]
    de = rev(torch.cummin(rev(torch.where(endd, t + 1, L)), 0).values)
    # question end of each token's group: question tokens come first in the group
    qe = torch.where(isq, de, 0)                                  # question tokens: their own (question) doc end
    qe_g = torch.zeros(L + 1, dtype=torch.int32, device=dev).scatter_reduce(0, gs.long(), qe, 'amax')[gs.long()]
    qe_g = torch.maximum(qe_g, gs)                               # a group without question tokens: empty range
    own_s = torch.where(isq, gs, ds)                              # earliest packed key a token can see (R2 start)
    vis_end = torch.where(isq, ge, de)                            # last query (excl.) that sees this key
    out = {}
    for tag, BM in (('f', cfg['FBM']), ('q', cfg['QBM'])):
        nb = (L + BM - 1) // BM; P = nb * BM - L
        first = torch.arange(nb, device=dev) * BM
        r1s = gs[first].clone(); r1e = qe_g[first].clone()
        o2 = torch.nn.functional.pad(own_s, (0, P), value=2 ** 30).view(nb, BM).min(1).values
        r2s = torch.minimum(o2, first.to(torch.int32))
        ov = r1e >= r2s                                           # overlapping: one merged range R2
        r2s = torch.where(ov, torch.minimum(r1s, r2s), r2s); r1e = torch.where(ov, r1s, r1e)
        out[tag] = (r1s.contiguous(), r1e.contiguous(), r2s.contiguous())
    BN = cfg['KBN']; nb = (L + BN - 1) // BN; P = nb * BN - L
    out['qe'] = torch.nn.functional.pad(vis_end, (0, P), value=0).view(nb, BN).max(1).values.contiguous()
    return out


class GroupAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, Kp, Vp, grp, doc, mt):
        L, H, HD = q.shape; KVH = k.shape[1]; Lp = Kp.shape[0]; c = CFG
        sm = HD ** -0.5
        O = torch.empty_like(q); LSE = torch.empty(H, L, device=q.device, dtype=torch.float32)
        r1s, r1e, r2s = mt['f']
        _attn_fwd[(triton.cdiv(L, c['FBM']), H)](q, Kp, Vp, k, v, O, LSE, grp, doc, r1s, r1e, r2s, Lp, L, sm * LOG2E,
                                                 BM=c['FBM'], BN=c['FBN'], HD=HD, H=H, KVH=KVH, num_warps=c['FW'], num_stages=c['FS'])
        ctx.save_for_backward(q, k, v, Kp, Vp, O, LSE, grp, doc)
        ctx.mt = mt
        return O

    @staticmethod
    def backward(ctx, dO):
        q, k, v, Kp, Vp, O, LSE, grp, doc = ctx.saved_tensors
        mt = ctx.mt; c = CFG
        L, H, HD = q.shape; KVH = k.shape[1]; Lp = Kp.shape[0]
        sm = HD ** -0.5
        dO = dO.contiguous()
        dq = torch.empty_like(q); DELTA = torch.empty(H, L, device=q.device, dtype=torch.float32)
        r1s, r1e, r2s = mt['q']
        _attn_bwd_dq[(triton.cdiv(L, c['QBM']), H)](q, Kp, Vp, k, v, O, dO, LSE, DELTA, dq, grp, doc, r1s, r1e, r2s, Lp, L,
                                                    sm * LOG2E, sm, BM=c['QBM'], BN=c['QBN'], HD=HD, H=H, KVH=KVH,
                                                    num_warps=c['QW'], num_stages=c['QS'])
        dk = torch.empty_like(k); dv = torch.empty_like(v)
        _attn_bwd_dkdv[(triton.cdiv(L, c['KBN']), KVH)](q, k, v, dO, LSE, DELTA, dk, dv, grp, doc, mt['qe'], L, sm * LOG2E, sm,
                                                        BM=c['KBM'], BN=c['KBN'], HD=HD, H=H, KVH=KVH,
                                                        num_warps=c['KW'], num_stages=c['KS'])
        return dq, dk, dv, None, None, None, None, None


def group_attn(q, k, v, Kp, Vp, grp, doc, mt):
    return GroupAttn.apply(q, k, v, Kp, Vp, grp, doc, mt)


# --------------------------------------------------------------------------------------------------------- test
def reference(q, k, v, Kp, Vp, grp, doc):
    """dense fp32 reference (small L only)."""
    L, H, HD = q.shape; KVH = k.shape[1]; Lp = Kp.shape[0]
    i = torch.arange(L, device=q.device)
    vis = (grp[None, :] == grp[:, None]) & (i[None, :] <= i[:, None]) & ((doc[None, :] < 0) | (doc[None, :] == doc[:, None]))
    full = torch.cat([torch.ones(L, Lp, dtype=torch.bool, device=q.device), vis], 1)
    Kc = torch.cat([Kp, k], 0).float().repeat_interleave(H // KVH, 1); Vc = torch.cat([Vp, v], 0).float().repeat_interleave(H // KVH, 1)
    s = torch.einsum('ihd,jhd->hij', q.float(), Kc) * HD ** -0.5
    s = s.masked_fill(~full[None], float('-inf'))
    return torch.einsum('hij,jhd->ihd', torch.softmax(s, -1), Vc)


def selftest(seed=0, Lp=900, ngroups=6, dev='cuda'):
    import random
    r = random.Random(seed); g, d = [], []; did = 0
    for gi in range(ngroups):
        ql = r.randint(40, 200); g += [gi] * ql; d += [-1] * ql
        for _ in range(r.randint(1, 6)):
            rl = r.randint(1, 600); g += [gi] * rl; d += [did] * rl; did += 1
    pad = r.randint(0, 300); g += [-1] * pad; d += [-1] * pad
    L = len(g); H, KVH, HD = 16, 8, 128
    gen = torch.Generator(device=dev).manual_seed(seed)
    mk = lambda *s: (torch.randn(*s, device=dev, generator=gen) * 1.5).to(torch.bfloat16)
    q, k, v = mk(L, H, HD), mk(L, KVH, HD), mk(L, KVH, HD); Kp, Vp = mk(Lp, KVH, HD), mk(Lp, KVH, HD)
    grp = torch.tensor(g, dtype=torch.int32, device=dev); doc = torch.tensor(d, dtype=torch.int32, device=dev)
    mt = meta(grp, doc, L)
    qr, kr, vr = [x.float().requires_grad_(True) for x in (q, k, v)]
    o_ref = reference(qr, kr, vr, Kp, Vp, grp, doc)
    do = mk(L, H, HD)
    o_ref.backward(do.float())
    qt, kt, vt = [x.clone().requires_grad_(True) for x in (q, k, v)]
    o = group_attn(qt, kt, vt, Kp, Vp, grp, doc, mt)
    o.backward(do)
    rel = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm())
    res = dict(L=L, o=rel(o, o_ref), dq=rel(qt.grad, qr.grad), dk=rel(kt.grad, kr.grad), dv=rel(vt.grad, vr.grad))
    # bf16 rounding floor: the fp32 reference output rounded to bf16
    res['o_floor'] = rel(o_ref.to(torch.bfloat16), o_ref)
    return res
