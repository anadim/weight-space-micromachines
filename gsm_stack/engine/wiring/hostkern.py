"""Triton kernels for the 0.6B host decode step (one token per row). Per layer: add+RMSNorm, [GEMM qkv], q/k-norm +
RoPE + KV write, prefix attention (tensor cores, prefix KV shared by all rows), suffix attention (per row, only the
row's own valid keys: the loop bound is the row's length, so no length buckets are needed), [GEMM o], add+RMSNorm,
[GEMM gate|up], SiLU*mul, [GEMM down]. Plus a fused sampler (Gumbel-max + log-softmax in one pass over the vocab).
All kernels compute in fp32 and round where the bf16 reference rounds; with fp32 buffers they are fp32 throughout
(tl.dot in IEEE fp32), which is how token identity vs HF is checked."""
import torch, triton, triton.language as tl

EPS = 1e-6


@triton.jit
def add_rms_kernel(x, delta, w, h_out, D: tl.constexpr, HAS_DELTA: tl.constexpr, eps: tl.constexpr):
    r = tl.program_id(0)
    d = tl.arange(0, D)
    xv = tl.load(x + r * D + d)
    if HAS_DELTA:
        xv = (xv.to(tl.float32) + tl.load(delta + r * D + d).to(tl.float32)).to(x.dtype.element_ty)
        tl.store(x + r * D + d, xv)
    xf = xv.to(tl.float32)
    var = tl.sum(xf * xf, axis=0) / D
    y = (xf * (1.0 / tl.sqrt(var + eps))).to(x.dtype.element_ty)
    out = (tl.load(w + d).to(tl.float32) * y.to(tl.float32)).to(x.dtype.element_ty)
    tl.store(h_out + r * D + d, out)


@triton.jit
def qk_rope_kernel(qkv, qn, kn, cos, sin, pos, q_out, Kc, Vc, slot, sK_b, sK_h, sK_s,
                   H: tl.constexpr, KVH: tl.constexpr, HD: tl.constexpr, eps: tl.constexpr):
    r = tl.program_id(0)
    hh = tl.arange(0, 32)                    # 16 q heads + 8 k heads (+8 v heads, not normed)
    d = tl.arange(0, HD)
    half: tl.constexpr = HD // 2
    dp = (d + half) % HD
    sgn = tl.where(d < half, -1.0, 1.0)
    p = tl.load(pos + r)
    c = tl.load(cos + p * HD + d).to(tl.float32); s = tl.load(sin + p * HD + d).to(tl.float32)
    nh = H + KVH
    hm = hh < nh
    base = qkv + r * (H + 2 * KVH) * HD
    x = tl.load(base + hh[:, None] * HD + d[None, :], mask=hm[:, None], other=0.0)
    xp = tl.load(base + hh[:, None] * HD + dp[None, :], mask=hm[:, None], other=0.0)
    xf = x.to(tl.float32)
    rr = 1.0 / tl.sqrt(tl.sum(xf * xf, axis=1) / HD + eps)                     # per head
    isq = hh < H
    wv = tl.where(isq[:, None], tl.load(qn + d)[None, :].to(tl.float32), tl.load(kn + d)[None, :].to(tl.float32))
    wp = tl.where(isq[:, None], tl.load(qn + dp)[None, :].to(tl.float32), tl.load(kn + dp)[None, :].to(tl.float32))
    dt = x.dtype
    y = (wv * (xf * rr[:, None]).to(dt).to(tl.float32)).to(dt).to(tl.float32)
    yp = (wp * (xp.to(tl.float32) * rr[:, None]).to(dt).to(tl.float32)).to(dt).to(tl.float32)
    t1 = (y * c[None, :]).to(dt).to(tl.float32)
    t2 = ((sgn[None, :] * yp) * s[None, :]).to(dt).to(tl.float32)
    o = (t1 + t2).to(dt)
    tl.store(q_out + r * H * HD + hh[:, None] * HD + d[None, :], o, mask=isq[:, None])
    sl = tl.load(slot + r)
    kh = hh - H
    km = (hh >= H) & hm
    tl.store(Kc + r * sK_b + kh[:, None] * sK_h + sl * sK_s + d[None, :], o, mask=km[:, None])
    vh = tl.arange(0, 8)
    v = tl.load(base + (nh + vh)[:, None] * HD + d[None, :])
    tl.store(Vc + r * sK_b + vh[:, None] * sK_h + sl * sK_s + d[None, :], v)


@triton.jit
def attn_prefix_kernel(q, Kp, Vp, m_out, l_out, acc_out, B, Lp, scale,
                       H: tl.constexpr, HD: tl.constexpr, RBK: tl.constexpr, BN: tl.constexpr, IEEE: tl.constexpr):
    """grid (KVH, cdiv(B, RBK)). q (B,H,HD); Kp/Vp (KVH, Lp, HD) of one layer. Unnormalized online-softmax state out."""
    g = tl.program_id(0); rb = tl.program_id(1)
    M: tl.constexpr = 2 * RBK
    mi = tl.arange(0, M)
    row = rb * RBK + mi // 2
    head = 2 * g + mi % 2
    rmask = row < B
    d = tl.arange(0, HD)
    Q = tl.load(q + (row[:, None] * H + head[:, None]) * HD + d[None, :], mask=rmask[:, None], other=0.0)
    m = tl.full((M,), float('-inf'), tl.float32); l = tl.zeros((M,), tl.float32); acc = tl.zeros((M, HD), tl.float32)
    n = tl.arange(0, BN)
    for j0 in range(0, Lp, BN):
        km = (j0 + n) < Lp
        K = tl.load(Kp + (g * Lp + j0 + n)[:, None] * HD + d[None, :], mask=km[:, None], other=0.0)
        if IEEE:
            s = tl.dot(Q, tl.trans(K), input_precision='ieee') * scale
        else:
            s = tl.dot(Q, tl.trans(K)) * scale
        s = tl.where(km[None, :], s, float('-inf'))
        mn = tl.maximum(m, tl.max(s, axis=1))
        pr = tl.exp(s - mn[:, None])
        al = tl.exp(m - mn)
        l = l * al + tl.sum(pr, axis=1)
        V = tl.load(Vp + (g * Lp + j0 + n)[:, None] * HD + d[None, :], mask=km[:, None], other=0.0)
        if IEEE:
            acc = acc * al[:, None] + tl.dot(pr.to(V.dtype), V, input_precision='ieee')
        else:
            acc = acc * al[:, None] + tl.dot(pr.to(V.dtype), V)
        m = mn
    o = row * H + head
    tl.store(m_out + o, m, mask=rmask); tl.store(l_out + o, l, mask=rmask)
    tl.store(acc_out + o[:, None] * HD + d[None, :], acc, mask=rmask[:, None])


@triton.jit
def attn_suffix_kernel(q, Kc, Vc, slot, m_in, l_in, acc_in, out, sK_b, sK_h, sK_s, scale,
                       H: tl.constexpr, HD: tl.constexpr, BN: tl.constexpr, IEEE: tl.constexpr):
    """grid (B, KVH). Continues the prefix softmax state over the row's own keys 0..slot, writes the normalized output."""
    r = tl.program_id(0); g = tl.program_id(1)
    mi = tl.arange(0, 16)                                  # 2 real query heads, padded to 16 rows for tl.dot
    qm = mi < 2
    head = 2 * g + mi
    d = tl.arange(0, HD)
    Q = tl.load(q + (r * H + head[:, None]) * HD + d[None, :], mask=qm[:, None], other=0.0)
    o = r * H + head
    m = tl.load(m_in + o, mask=qm, other=0.0); l = tl.load(l_in + o, mask=qm, other=1.0)
    acc = tl.load(acc_in + o[:, None] * HD + d[None, :], mask=qm[:, None], other=0.0)
    n = tl.arange(0, BN)
    L = tl.load(slot + r) + 1
    for j0 in range(0, L, BN):
        km = (j0 + n) < L
        K = tl.load(Kc + r * sK_b + g * sK_h + (j0 + n)[:, None] * sK_s + d[None, :], mask=km[:, None], other=0.0)
        if IEEE:
            s = tl.dot(Q, tl.trans(K), input_precision='ieee') * scale
        else:
            s = tl.dot(Q, tl.trans(K)) * scale
        s = tl.where(km[None, :], s, float('-inf'))
        mn = tl.maximum(m, tl.max(s, axis=1))
        pr = tl.exp(s - mn[:, None])
        al = tl.exp(m - mn)
        l = l * al + tl.sum(pr, axis=1)
        V = tl.load(Vc + r * sK_b + g * sK_h + (j0 + n)[:, None] * sK_s + d[None, :], mask=km[:, None], other=0.0)
        if IEEE:
            acc = acc * al[:, None] + tl.dot(pr.to(V.dtype), V, input_precision='ieee')
        else:
            acc = acc * al[:, None] + tl.dot(pr.to(V.dtype), V)
        m = mn
    res = acc / l[:, None]
    tl.store(out + o[:, None] * HD + d[None, :], res.to(out.dtype.element_ty), mask=qm[:, None])


@triton.jit
def silu_mul_kernel(gu, out, FF: tl.constexpr, BLK: tl.constexpr):
    r = tl.program_id(0); c = tl.program_id(1)
    i = c * BLK + tl.arange(0, BLK)
    g = tl.load(gu + r * 2 * FF + i).to(tl.float32)
    u = tl.load(gu + r * 2 * FF + FF + i)
    a = (g / (1.0 + tl.exp(-g))).to(u.dtype)
    tl.store(out + r * FF + i, (a.to(tl.float32) * u.to(tl.float32)).to(u.dtype))


@triton.jit
def sample_kernel(logits, temp, greedy, seed, tok_out, lp_out, V, BV: tl.constexpr):
    """One pass over the vocab: z = logit/T; argmax(z + Gumbel) (or argmax z if greedy); logp = z[tok] - logsumexp(z)."""
    r = tl.program_id(0)
    T = tl.load(temp); gr = tl.load(greedy); sd = tl.load(seed)
    best = tl.full((BV,), float('-inf'), tl.float32); bidx = tl.zeros((BV,), tl.int32); bz = tl.zeros((BV,), tl.float32)
    mx = tl.full((BV,), float('-inf'), tl.float32); sm = tl.zeros((BV,), tl.float32)
    for v0 in range(0, V, BV):
        i = v0 + tl.arange(0, BV); vm = i < V
        z = tl.load(logits + r.to(tl.int64) * V + i, mask=vm, other=float('-inf')).to(tl.float32) / T
        u = tl.rand(sd, r * V + i)
        u = tl.maximum(u, 1e-20)
        gmb = -tl.log(-tl.log(u))
        key = tl.where(gr != 0, z, z + gmb)
        key = tl.where(vm, key, float('-inf'))
        up = key > best
        best = tl.where(up, key, best); bidx = tl.where(up, i, bidx); bz = tl.where(up, z, bz)
        mn = tl.maximum(mx, z)
        sm = sm * tl.exp(mx - mn) + tl.where(vm, tl.exp(z - mn), 0.0)
        mx = mn
    bmax = tl.max(best, axis=0)
    cand = tl.where(best == bmax, bidx, 2147483647)
    t = tl.min(cand, axis=0)
    zt = tl.max(tl.where(bidx == t, bz, float('-inf')), axis=0)
    gm = tl.max(mx, axis=0)
    lse = gm + tl.log(tl.sum(sm * tl.exp(mx - gm), axis=0))
    tl.store(tok_out + r, t.to(tl.int64)); tl.store(lp_out + r, zt - lse)
