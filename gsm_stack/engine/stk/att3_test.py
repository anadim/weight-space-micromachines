# unit test: attn3 (split-KV prefix + merged forget-skip suffix) vs hostkern prefix + fastroll skip suffix on random data
import sys, os, torch, types
import fastqwen as Q
import hostkern as HK
import fastroll, attn3
torch.manual_seed(0)
dev = 'cuda'
for B, Lp, S in [(384, 1066, 1600), (32, 1066, 900), (5, 1066, 300), (100, 200, 400)]:
    E = 16
    e = types.SimpleNamespace(B_max=B, Lp=Lp, dev=dev, w_E=E)
    q = torch.randn(B, Q.H, Q.HD, device=dev, dtype=torch.bfloat16)
    Kp = torch.randn(Q.KVH, Lp, Q.HD, device=dev, dtype=torch.bfloat16); Vp = torch.randn_like(Kp)
    Kc = torch.randn(B, Q.KVH, S, Q.HD, device=dev, dtype=torch.bfloat16); Vc = torch.randn_like(Kc)
    slot = torch.randint(0, S, (B,), device=dev)
    qlen = torch.randint(0, 100, (B,), device=dev); fr = torch.where(torch.rand(B, device=dev) < .5, qlen + torch.randint(0, 300, (B,), device=dev), torch.full_like(qlen, -1))
    e.s_nf = torch.randint(0, 3, (B,), device=dev); e.s_fr = fr; e.s_qlen = qlen
    e.w_Kx = torch.zeros(B + 1, S, E, device=dev); e.w_Qx = torch.zeros(B + 1, E, device=dev)
    e.w_ar = torch.arange(B + 1, device=dev); I = torch.eye(E, device=dev); e.w_WK = I; e.w_WQ = -(1 - I); e.w_M = 1e4
    # forgotten keys [qlen, fr) carry label 0 with nf=1 -> bias -1e4
    e.w_Kx[:B, :, 1] = 1.0
    for r in range(B):
        if fr[r] > 0: e.w_Kx[r, qlen[r]:fr[r]] = 0; e.w_Kx[r, qlen[r]:fr[r], 0] = 1.0
    e.s_nf.fill_(1); e.w_Qx[:B] = e.w_WQ[1] * 1e4
    sc = 1.0 / Q.HD ** 0.5
    out1 = torch.zeros(B, Q.H, Q.HD, device=dev, dtype=torch.bfloat16); out2 = torch.zeros_like(out1)
    mp = torch.zeros(B * Q.H, device=dev); lpf = torch.zeros_like(mp); accp = torch.zeros(B * Q.H, Q.HD, device=dev)
    e.mp, e.lpf, e.accp = mp, lpf, accp; e.c_forget = True; e.w_dbg = False
    e.Kp = [Kp, Kp]; e.Vp = [Vp, Vp]
    HK.attn_prefix_kernel[(Q.KVH, (B + 31) // 32)](q, Kp, Vp, mp, lpf, accp, B, Lp, sc, H=Q.H, HD=Q.HD, RBK=32, BN=64, IEEE=False, num_warps=4, num_stages=2)
    fastroll._attn_suffix_skip(e, 1, B, q, Kc, Vc, slot, out1, sc, False)
    attn3.attention(e, 1, B, q, Kc, Vc, slot, out2, sc)
    # fp64 reference
    qf = q.double(); ref = torch.zeros(B, Q.H, Q.HD, dtype=torch.float64, device=dev)
    for r in range(B):
        L = int(slot[r]) + 1
        K = torch.cat([Kp, Kc[r, :, :L]], 1).double().repeat_interleave(2, 0); V = torch.cat([Vp, Vc[r, :, :L]], 1).double().repeat_interleave(2, 0)
        s = torch.einsum('hd,hkd->hk', qf[r], K) * sc
        bias = torch.cat([torch.zeros(Lp, device=dev), (e.w_Kx[r, :L] @ e.w_Qx[r])]).double()
        s = s + bias
        ref[r] = torch.einsum('hk,hkd->hd', s.softmax(-1), V)
    d1 = (out1.double() - ref).abs().max().item(); d2 = (out2.double() - ref).abs().max().item(); d12 = (out1.double() - out2.double()).abs().max().item()
    print(f'B={B} Lp={Lp}: old-ref {d1:.3e} att3-ref {d2:.3e} old-att3 {d12:.3e} nfr={(fr > 0).sum().item()}', flush=True)
