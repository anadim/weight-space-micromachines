"""Split-KV decode attention for the serial stack (FAST_ATT3=1).

prefix: grid (KVH, row blocks, NSPL prefix chunks) instead of (KVH, row blocks): the shared prefix (~1k keys) is split into
chunks of CH keys, each CTA writes an unnormalized partial softmax state (m, l, acc) per (row, head, chunk). At small B
the old kernel ran 8 CTAs walking 17 key blocks each (~40 us per layer, flat in B).
suffix: the forget-skip suffix kernel (fastroll.attn_suffix_bias_skip_kernel), but it first merges the NSPL prefix partials
(m = max m_s, l = sum l_s e^(m_s - m), acc = sum acc_s e^(m_s - m)) and then continues the online softmax over the row's
visible keys. Same math as the sequential kernel up to fp32 summation order (not bitwise)."""
import torch, triton, triton.language as tl
import fastqwen as Q
import torch.nn.functional as F

CH = 128
PIPE = int(__import__('os').environ.get('FAST_ATT3P', 1))   # 1: pipelined suffix kernel (bitwise equal to the original)


@triton.jit
def attn_prefix_split_kernel(q, Kp, Vp, pm, pl, pacc, B, Lp, BH, scale,
                             H: tl.constexpr, HD: tl.constexpr, RBK: tl.constexpr, BN: tl.constexpr, CH: tl.constexpr):
    g = tl.program_id(0); rb = tl.program_id(1); sp = tl.program_id(2)
    M: tl.constexpr = 2 * RBK
    mi = tl.arange(0, M)
    row = rb * RBK + mi // 2
    head = 2 * g + mi % 2
    rmask = row < B
    d = tl.arange(0, HD)
    Qv = tl.load(q + (row[:, None] * H + head[:, None]) * HD + d[None, :], mask=rmask[:, None], other=0.0)
    m = tl.full((M,), float('-inf'), tl.float32); l = tl.zeros((M,), tl.float32); acc = tl.zeros((M, HD), tl.float32)
    n = tl.arange(0, BN)
    for j0 in range(sp * CH, tl.minimum(Lp, sp * CH + CH), BN):
        km = (j0 + n) < Lp
        K = tl.load(Kp + (g * Lp + j0 + n)[:, None] * HD + d[None, :], mask=km[:, None], other=0.0)
        s = tl.dot(Qv, tl.trans(K)) * scale
        s = tl.where(km[None, :], s, float('-inf'))
        mn = tl.maximum(m, tl.max(s, axis=1))
        pr = tl.exp(s - mn[:, None])
        al = tl.exp(m - mn)
        l = l * al + tl.sum(pr, axis=1)
        V = tl.load(Vp + (g * Lp + j0 + n)[:, None] * HD + d[None, :], mask=km[:, None], other=0.0)
        acc = acc * al[:, None] + tl.dot(pr.to(V.dtype), V)
        m = mn
    o = sp * BH + row * H + head
    tl.store(pm + o, m, mask=rmask); tl.store(pl + o, l, mask=rmask)
    tl.store(pacc + o[:, None] * HD + d[None, :], acc, mask=rmask[:, None])


@triton.jit
def attn_suffix3_kernel(q, Kc, Vc, slot, Qx, Kx, sKx_b, qlen, fr, pm, pl, pacc, BH, out, sK_b, sK_h, sK_s, scale,
                        H: tl.constexpr, HD: tl.constexpr, BN: tl.constexpr, E: tl.constexpr, NSPL: tl.constexpr):
    r = tl.program_id(0); g = tl.program_id(1)
    mi = tl.arange(0, 16)
    qm = mi < 2
    head = 2 * g + mi
    d = tl.arange(0, HD)
    Qv = tl.load(q + (r * H + head[:, None]) * HD + d[None, :], mask=qm[:, None], other=0.0)
    o = r * H + head
    # merge the prefix partials
    m = tl.full((16,), float('-inf'), tl.float32)
    for s_ in tl.static_range(NSPL):
        m = tl.maximum(m, tl.load(pm + s_ * BH + o, mask=qm, other=0.0))
    l = tl.zeros((16,), tl.float32); acc = tl.zeros((16, HD), tl.float32)
    for s_ in tl.static_range(NSPL):
        ms = tl.load(pm + s_ * BH + o, mask=qm, other=0.0)
        w = tl.exp(ms - m)
        l += tl.load(pl + s_ * BH + o, mask=qm, other=1.0) * w
        acc += tl.load(pacc + (s_ * BH + o)[:, None] * HD + d[None, :], mask=qm[:, None], other=0.0) * w[:, None]
    n = tl.arange(0, BN)
    e = tl.arange(0, E)
    qx = tl.load(Qx + r * E + e)
    L = tl.load(slot + r) + 1
    lo = tl.load(qlen + r); hi = tl.load(fr + r)
    for j0 in range(0, L, BN):
        if (j0 < lo) | (j0 + BN > hi):
            jj = j0 + n
            km = jj < L
            K = tl.load(Kc + r * sK_b + g * sK_h + jj[:, None] * sK_s + d[None, :], mask=km[:, None], other=0.0)
            s = tl.dot(Qv, tl.trans(K)) * scale
            kx = tl.load(Kx + r * sKx_b + jj[:, None] * E + e[None, :], mask=km[:, None], other=0.0)
            bias = tl.sum(kx * qx[None, :], axis=1)
            s = s + bias[None, :]
            s = tl.where(km[None, :], s, float('-inf'))
            mn = tl.maximum(m, tl.max(s, axis=1))
            pr = tl.exp(s - mn[:, None])
            al = tl.exp(m - mn)
            l = l * al + tl.sum(pr, axis=1)
            V = tl.load(Vc + r * sK_b + g * sK_h + jj[:, None] * sK_s + d[None, :], mask=km[:, None], other=0.0)
            acc = acc * al[:, None] + tl.dot(pr.to(V.dtype), V)
            m = mn
    res = acc / l[:, None]
    tl.store(out + o[:, None] * HD + d[None, :], res.to(out.dtype.element_ty), mask=qm[:, None])


@triton.jit
def attn_suffix3p_kernel(q, Kc, Vc, slot, Qx, Kx, sKx_b, qlen, fr, pm, pl, pacc, BH, out, sK_b, sK_h, sK_s, scale,
                         H: tl.constexpr, HD: tl.constexpr, BN: tl.constexpr, E: tl.constexpr, NSPL: tl.constexpr,
                         STG: tl.constexpr):
    """attn_suffix3_kernel with the visible blocks walked by two branch-free loops (question blocks j0 < qlen, then the
    blocks reaching past fr) so the K/V loads software-pipeline. Same blocks in the same order, same per-block math."""
    r = tl.program_id(0); g = tl.program_id(1)
    mi = tl.arange(0, 16)
    qm = mi < 2
    head = 2 * g + mi
    d = tl.arange(0, HD)
    Qv = tl.load(q + (r * H + head[:, None]) * HD + d[None, :], mask=qm[:, None], other=0.0)
    o = r * H + head
    m = tl.full((16,), float('-inf'), tl.float32)
    for s_ in tl.static_range(NSPL):
        m = tl.maximum(m, tl.load(pm + s_ * BH + o, mask=qm, other=0.0))
    l = tl.zeros((16,), tl.float32); acc = tl.zeros((16, HD), tl.float32)
    for s_ in tl.static_range(NSPL):
        ms = tl.load(pm + s_ * BH + o, mask=qm, other=0.0)
        w = tl.exp(ms - m)
        l += tl.load(pl + s_ * BH + o, mask=qm, other=1.0) * w
        acc += tl.load(pacc + (s_ * BH + o)[:, None] * HD + d[None, :], mask=qm[:, None], other=0.0) * w[:, None]
    n = tl.arange(0, BN)
    e = tl.arange(0, E)
    qx = tl.load(Qx + r * E + e)
    L = tl.load(slot + r) + 1
    lo = tl.load(qlen + r); hi = tl.load(fr + r)
    e1 = tl.minimum(lo, L)
    s2 = tl.maximum((e1 + BN - 1) // BN, (tl.maximum(hi - BN + 1, 0) + BN - 1) // BN) * BN
    Kb = Kc + r * sK_b + g * sK_h; Vb = Vc + r * sK_b + g * sK_h; Xb = Kx + r * sKx_b
    for j0 in tl.range(0, e1, BN, num_stages=STG):
        jj = j0 + n
        km = jj < L
        K = tl.load(Kb + jj[:, None] * sK_s + d[None, :], mask=km[:, None], other=0.0)
        s = tl.dot(Qv, tl.trans(K)) * scale
        kx = tl.load(Xb + jj[:, None] * E + e[None, :], mask=km[:, None], other=0.0)
        bias = tl.sum(kx * qx[None, :], axis=1)
        s = s + bias[None, :]
        s = tl.where(km[None, :], s, float('-inf'))
        mn = tl.maximum(m, tl.max(s, axis=1))
        pr = tl.exp(s - mn[:, None])
        al = tl.exp(m - mn)
        l = l * al + tl.sum(pr, axis=1)
        V = tl.load(Vb + jj[:, None] * sK_s + d[None, :], mask=km[:, None], other=0.0)
        acc = acc * al[:, None] + tl.dot(pr.to(V.dtype), V)
        m = mn
    for j0 in tl.range(s2, L, BN, num_stages=STG):
        jj = j0 + n
        km = jj < L
        K = tl.load(Kb + jj[:, None] * sK_s + d[None, :], mask=km[:, None], other=0.0)
        s = tl.dot(Qv, tl.trans(K)) * scale
        kx = tl.load(Xb + jj[:, None] * E + e[None, :], mask=km[:, None], other=0.0)
        bias = tl.sum(kx * qx[None, :], axis=1)
        s = s + bias[None, :]
        s = tl.where(km[None, :], s, float('-inf'))
        mn = tl.maximum(m, tl.max(s, axis=1))
        pr = tl.exp(s - mn[:, None])
        al = tl.exp(m - mn)
        l = l * al + tl.sum(pr, axis=1)
        V = tl.load(Vb + jj[:, None] * sK_s + d[None, :], mask=km[:, None], other=0.0)
        acc = acc * al[:, None] + tl.dot(pr.to(V.dtype), V)
        m = mn
    res = acc / l[:, None]
    tl.store(out + o[:, None] * HD + d[None, :], res.to(out.dtype.element_ty), mask=qm[:, None])


@triton.jit
def attn_suffix3b_kernel(q, Kc, Vc, slot, Qx, Kx, sKx_b, pm, pl, pacc, BH, out, sK_b, sK_h, sK_s, scale,
                         H: tl.constexpr, HD: tl.constexpr, BN: tl.constexpr, E: tl.constexpr, NSPL: tl.constexpr,
                         STG: tl.constexpr):
    """arm B (StackH soft forget, M = 50): forget_ws.attn_suffix_bias_kernel over the full suffix 0..slot, starting from the
    merged NSPL split-prefix partials (as attn_suffix3p_kernel) instead of one sequential prefix state."""
    r = tl.program_id(0); g = tl.program_id(1)
    mi = tl.arange(0, 16)
    qm = mi < 2
    head = 2 * g + mi
    d = tl.arange(0, HD)
    Qv = tl.load(q + (r * H + head[:, None]) * HD + d[None, :], mask=qm[:, None], other=0.0)
    o = r * H + head
    m = tl.full((16,), float('-inf'), tl.float32)
    for s_ in tl.static_range(NSPL):
        m = tl.maximum(m, tl.load(pm + s_ * BH + o, mask=qm, other=0.0))
    l = tl.zeros((16,), tl.float32); acc = tl.zeros((16, HD), tl.float32)
    for s_ in tl.static_range(NSPL):
        ms = tl.load(pm + s_ * BH + o, mask=qm, other=0.0)
        w = tl.exp(ms - m)
        l += tl.load(pl + s_ * BH + o, mask=qm, other=1.0) * w
        acc += tl.load(pacc + (s_ * BH + o)[:, None] * HD + d[None, :], mask=qm[:, None], other=0.0) * w[:, None]
    n = tl.arange(0, BN)
    e = tl.arange(0, E)
    qx = tl.load(Qx + r * E + e)
    L = tl.load(slot + r) + 1
    Kb = Kc + r * sK_b + g * sK_h; Vb = Vc + r * sK_b + g * sK_h; Xb = Kx + r * sKx_b
    for j0 in tl.range(0, L, BN, num_stages=STG):
        jj = j0 + n
        km = jj < L
        K = tl.load(Kb + jj[:, None] * sK_s + d[None, :], mask=km[:, None], other=0.0)
        s = tl.dot(Qv, tl.trans(K)) * scale
        kx = tl.load(Xb + jj[:, None] * E + e[None, :], mask=km[:, None], other=0.0)
        bias = tl.sum(kx * qx[None, :], axis=1)
        s = s + bias[None, :]
        s = tl.where(km[None, :], s, float('-inf'))
        mn = tl.maximum(m, tl.max(s, axis=1))
        pr = tl.exp(s - mn[:, None])
        al = tl.exp(m - mn)
        l = l * al + tl.sum(pr, axis=1)
        V = tl.load(Vb + jj[:, None] * sK_s + d[None, :], mask=km[:, None], other=0.0)
        acc = acc * al[:, None] + tl.dot(pr.to(V.dtype), V)
        m = mn
    res = acc / l[:, None]
    tl.store(out + o[:, None] * HD + d[None, :], res.to(out.dtype.element_ty), mask=qm[:, None])


def setup(eng):
    Bp = eng.B_max + 1
    eng.a3_nspl = triton.cdiv(eng.Lp, CH)
    eng.a3_pm = torch.zeros(eng.a3_nspl * Bp * Q.H, device=eng.dev)
    eng.a3_pl = torch.zeros_like(eng.a3_pm)
    eng.a3_pacc = torch.zeros(eng.a3_nspl * Bp * Q.H, Q.HD, device=eng.dev)


def attention(eng, i, B, q, Kc, Vc, slot, att, sc):
    """prefix (split) + forget-skip suffix of layer i for rows 0..B-1 (bf16 path only)."""
    if not hasattr(eng, 'a3_pm'): setup(eng)
    if i == 0:                                         # side coordinates of this step (as forget_ws)
        a = eng.s_nf[:B].clamp(max=eng.w_E - 1)
        lab = (a - (slot < eng.s_fr[:B]).long()).clamp(min=0)
        eng.w_Kx[eng.w_ar[:B], slot] = F.one_hot(lab, eng.w_E).float() @ eng.w_WK
        eng.w_Qx[:B] = (F.one_hot(a, eng.w_E).float() @ eng.w_WQ) * eng.w_M
    BH = (eng.B_max + 1) * Q.H
    rbk = 32 if B > 16 else 8
    attn_prefix_split_kernel[(Q.KVH, triton.cdiv(B, rbk), eng.a3_nspl)](
        q, eng.Kp[i], eng.Vp[i], eng.a3_pm, eng.a3_pl, eng.a3_pacc, B, eng.Lp, BH, sc,
        H=Q.H, HD=Q.HD, RBK=rbk, BN=64, CH=CH, num_warps=4, num_stages=2)
    if PIPE:
        attn_suffix3p_kernel[(B, Q.KVH)](q, Kc, Vc, slot, eng.w_Qx, eng.w_Kx, eng.w_Kx.stride(0), eng.s_qlen, eng.s_fr,
                                         eng.a3_pm, eng.a3_pl, eng.a3_pacc, BH, att, Kc.stride(0), Kc.stride(1), Kc.stride(2),
                                         sc, H=Q.H, HD=Q.HD, BN=64, E=eng.w_E, NSPL=eng.a3_nspl, STG=3, num_warps=4,
                                         num_stages=3)
        return
    attn_suffix3_kernel[(B, Q.KVH)](q, Kc, Vc, slot, eng.w_Qx, eng.w_Kx, eng.w_Kx.stride(0), eng.s_qlen, eng.s_fr,
                                    eng.a3_pm, eng.a3_pl, eng.a3_pacc, BH, att, Kc.stride(0), Kc.stride(1), Kc.stride(2), sc,
                                    H=Q.H, HD=Q.HD, BN=64, E=eng.w_E, NSPL=eng.a3_nspl, num_warps=4, num_stages=2)


def attention_b(eng, i, B, q, Kc, Vc, slot, att, sc):
    """arm B: prefix (split) + full soft-forget bias suffix of layer i (bf16, c_forget on)."""
    if not hasattr(eng, 'a3_pm'): setup(eng)
    if i == 0:                                         # side coordinates of this step (as forget_ws)
        a = eng.s_nf[:B].clamp(max=eng.w_E - 1)
        lab = (a - (slot < eng.s_fr[:B]).long()).clamp(min=0)
        eng.w_Kx[eng.w_ar[:B], slot] = F.one_hot(lab, eng.w_E).float() @ eng.w_WK
        eng.w_Qx[:B] = (F.one_hot(a, eng.w_E).float() @ eng.w_WQ) * eng.w_M
    BH = (eng.B_max + 1) * Q.H
    rbk = 32 if B > 16 else 8
    attn_prefix_split_kernel[(Q.KVH, triton.cdiv(B, rbk), eng.a3_nspl)](
        q, eng.Kp[i], eng.Vp[i], eng.a3_pm, eng.a3_pl, eng.a3_pacc, B, eng.Lp, BH, sc,
        H=Q.H, HD=Q.HD, RBK=rbk, BN=64, CH=CH, num_warps=4, num_stages=2)
    attn_suffix3b_kernel[(B, Q.KVH)](q, Kc, Vc, slot, eng.w_Qx, eng.w_Kx, eng.w_Kx.stride(0), eng.a3_pm, eng.a3_pl,
                                     eng.a3_pacc, BH, att, Kc.stride(0), Kc.stride(1), Kc.stride(2), sc, H=Q.H, HD=Q.HD,
                                     BN=64, E=eng.w_E, NSPL=eng.a3_nspl, STG=3, num_warps=4, num_stages=3)
