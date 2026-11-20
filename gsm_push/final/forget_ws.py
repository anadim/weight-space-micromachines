"""Weight-space forget for the stack agent's StackH (stack3_hook.py, STK_FORGET=weights, STK_FORGET_MOD=forget_ws).

Instead of a kernel-side key range mask, the implant keeps an attempt label per cached key and one per query, held in
side coordinates (not read or written by the host), and every attention head gets E extra, un-rotated q/k dims:
    k_extra(j) = W_K . onehot(label_j)          W_K = I_E
    q_extra    = M * W_Q . onehot(a_current)    W_Q = -(J_E - I_E)
    score_j    = q.k_j / sqrt(d) + q_extra . k_extra(j)
-> + 0 on keys of the current attempt and on prompt / question keys (k_extra = 0, "shared"), -M on keys of an
abandoned attempt.  M = 50 (logit units) by default; STK_FORGET_M overrides (1e4 -> exp underflows to exactly 0).
When no fire has happened every extra term is exactly 0.0, so the scores (and outputs) are bit-identical to the plain
kernel.

Registers read (stack agent's, batch-slot indexed, no compaction):
    s_nf[r]   attempt counter a_current (fires so far, both triggers)
    s_fr[r]   forget end of the latest fire (-1 = none); the key written at slot j gets label
              a_current - [j < s_fr]   (the token at a fire slot belongs to the abandoned attempt)
The label is written into the side cache at layer 0 of every step; the same fixed W_K / W_Q serve every layer and head
(the side coordinates are untouched by the host, so k_extra is the same at every layer and one cache serves all).
"""
import os, torch, torch.nn.functional as F
import triton, triton.language as tl
import fastqwen as Q
import engine as EM


@triton.jit
def attn_suffix_bias_kernel(q, Kc, Vc, slot, Qx, Kx, sKx_b, m_in, l_in, acc_in, out, wmax, sK_b, sK_h, sK_s, scale,
                            H: tl.constexpr, HD: tl.constexpr, BN: tl.constexpr, IEEE: tl.constexpr, E: tl.constexpr,
                            DBG: tl.constexpr):
    """hostkern.attn_suffix_kernel with E extra (un-rotated) q/k dims: score_j = q.k_j*scale + qx.kx_j.
    DBG: atomic-max of the softmax numerator over keys with a negative extra term (abandoned keys) -> wmax[r]
    (an upper bound on their final attention weight)."""
    r = tl.program_id(0); g = tl.program_id(1)
    mi = tl.arange(0, 16)
    qm = mi < 2
    head = 2 * g + mi
    d = tl.arange(0, HD)
    Qv = tl.load(q + (r * H + head[:, None]) * HD + d[None, :], mask=qm[:, None], other=0.0)
    o = r * H + head
    m = tl.load(m_in + o, mask=qm, other=0.0); l = tl.load(l_in + o, mask=qm, other=1.0)
    acc = tl.load(acc_in + o[:, None] * HD + d[None, :], mask=qm[:, None], other=0.0)
    n = tl.arange(0, BN)
    e = tl.arange(0, E)
    qx = tl.load(Qx + r * E + e)
    L = tl.load(slot + r) + 1
    for j0 in range(0, L, BN):
        jj = j0 + n
        km = jj < L
        K = tl.load(Kc + r * sK_b + g * sK_h + jj[:, None] * sK_s + d[None, :], mask=km[:, None], other=0.0)
        if IEEE:
            s = tl.dot(Qv, tl.trans(K), input_precision='ieee') * scale
        else:
            s = tl.dot(Qv, tl.trans(K)) * scale
        kx = tl.load(Kx + r * sKx_b + jj[:, None] * E + e[None, :], mask=km[:, None], other=0.0)
        bias = tl.sum(kx * qx[None, :], axis=1)
        s = s + bias[None, :]
        s = tl.where(km[None, :], s, float('-inf'))
        mn = tl.maximum(m, tl.max(s, axis=1))
        pr = tl.exp(s - mn[:, None])
        if DBG:
            ab = (bias < 0) & km
            pm = tl.max(tl.where(ab[None, :] & qm[:, None], pr, 0.0))
            tl.atomic_max(wmax + r, pm)
        al = tl.exp(m - mn)
        l = l * al + tl.sum(pr, axis=1)
        V = tl.load(Vc + r * sK_b + g * sK_h + jj[:, None] * sK_s + d[None, :], mask=km[:, None], other=0.0)
        if IEEE:
            acc = acc * al[:, None] + tl.dot(pr.to(V.dtype), V, input_precision='ieee')
        else:
            acc = acc * al[:, None] + tl.dot(pr.to(V.dtype), V)
        m = mn
    res = acc / l[:, None]
    tl.store(out + o[:, None] * HD + d[None, :], res.to(out.dtype.element_ty), mask=qm[:, None])


def wrap_stack(cls):
    class StackWS(cls):
        def stack_setup(self):
            super().stack_setup()
            E = 1 << int(self.c_maxf).bit_length()            # attempts 0..maxf, padded to a power of 2
            Bp = self.B_max + 1; dev = self.dev
            self.w_E = E; self.w_M = float(os.environ.get('STK_FORGET_M', 50))
            self.w_dbg = os.environ.get('STK_FORGET_DBG', '0') == '1'
            self.w_Kx = torch.zeros(Bp, self.S_max, E, device=dev)
            self.w_Qx = torch.zeros(Bp, E, device=dev)
            self.w_wmax = torch.zeros(Bp, device=dev)
            I = torch.eye(E)
            self.w_WK = I.to(dev); self.w_WQ = (-(torch.ones(E, E) - I)).to(dev)
            self.w_ar = torch.arange(Bp, device=dev)
            # extra parameters if materialised per layer and head: q 28 x 16 x E x E, k 28 x 8 x E x E
            self.w_params = Q.NL * (Q.H + Q.KVH) * E * E

        def _wave(self, qs, G, T, max_new, greedy):
            self.w_Kx.zero_()                                  # prompt / question keys: shared (extra dims 0)
            return super()._wave(qs, G, T, max_new, greedy)

        def _attn_suffix(self, i, B, q, Kc, Vc, slot, att, sc, ieee):
            if not self.c_forget:
                return super()._attn_suffix(i, B, q, Kc, Vc, slot, att, sc, ieee)
            if i == 0:                                         # side coordinates of this step (once per step)
                a = self.s_nf[:B].clamp(max=self.w_E - 1)
                lab = (a - (slot < self.s_fr[:B]).long()).clamp(min=0)
                self.w_Kx[self.w_ar[:B], slot] = F.one_hot(lab, self.w_E).float() @ self.w_WK
                self.w_Qx[:B] = (F.one_hot(a, self.w_E).float() @ self.w_WQ) * self.w_M
            attn_suffix_bias_kernel[(B, Q.KVH)](q, Kc, Vc, slot, self.w_Qx, self.w_Kx, self.w_Kx.stride(0), self.mp, self.lpf,
                                                self.accp, att, self.w_wmax, Kc.stride(0), Kc.stride(1), Kc.stride(2), sc,
                                                H=Q.H, HD=Q.HD, BN=64 if not ieee else 32, IEEE=ieee, E=self.w_E, DBG=self.w_dbg,
                                                num_warps=4, num_stages=1 if ieee else 2)
    StackWS.__name__ = cls.__name__ + 'WS'
    return StackWS
