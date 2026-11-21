# attn3 suffix: original vs pipelined (attn_suffix3p) — bitwise compare + timing on realistic forget-skip states
import os, torch, triton
import fastqwen as Q
from attn3 import attn_prefix_split_kernel, attn_suffix3_kernel, attn_suffix3p_kernel, CH
torch.manual_seed(0); dev = 'cuda'
E = 16; Lp = 1066; S = 1600


def bench(f, n=50):
    for _ in range(3): f()
    torch.cuda.synchronize(); a = torch.cuda.Event(True); b = torch.cuda.Event(True); a.record()
    for _ in range(n): f()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / n * 1e3


for B, ctx, seg in [(384, 768, 170), (384, 1536, 170), (384, 400, 0), (128, 768, 170), (32, 768, 170), (32, 300, 0), (8, 768, 170)]:
    q = torch.randn(B, Q.H, Q.HD, device=dev, dtype=torch.bfloat16)
    Kp = torch.randn(Q.KVH, Lp, Q.HD, device=dev, dtype=torch.bfloat16); Vp = torch.randn_like(Kp)
    Kc = torch.randn(B + 1, Q.KVH, S, Q.HD, device=dev, dtype=torch.bfloat16); Vc = torch.randn_like(Kc)
    slot = (ctx - 1 - torch.randint(0, 40, (B,), device=dev))
    qlen = torch.full((B,), 150, device=dev, dtype=torch.long) + torch.randint(-20, 20, (B,), device=dev)
    fr = (slot - seg) if seg else torch.full_like(slot, -1)
    fr = torch.where(fr > qlen, fr, torch.full_like(fr, -1))
    Kx = torch.randn(B + 1, S, E, device=dev); Qx = torch.randn(B + 1, E, device=dev)
    nspl = triton.cdiv(Lp, CH); BH = (B + 1) * Q.H
    pm = torch.zeros(nspl * BH, device=dev); pl = torch.zeros_like(pm); pacc = torch.zeros(nspl * BH, Q.HD, device=dev)
    sc = Q.HD ** -0.5; rbk = 32 if B > 16 else 8
    attn_prefix_split_kernel[(Q.KVH, triton.cdiv(B, rbk), nspl)](q, Kp, Vp, pm, pl, pacc, B, Lp, BH, sc, H=Q.H, HD=Q.HD,
                                                                 RBK=rbk, BN=64, CH=CH, num_warps=4, num_stages=2)
    o1 = torch.zeros(B, Q.H, Q.HD, device=dev, dtype=torch.bfloat16)
    args = lambda o: (q, Kc, Vc, slot, Qx, Kx, Kx.stride(0), qlen, fr, pm, pl, pacc, BH, o, Kc.stride(0), Kc.stride(1), Kc.stride(2), sc)
    f1 = lambda: attn_suffix3_kernel[(B, Q.KVH)](*args(o1), H=Q.H, HD=Q.HD, BN=64, E=E, NSPL=nspl, num_warps=4, num_stages=2)
    t1 = bench(f1)
    lo = torch.minimum(qlen, slot + 1); L = slot + 1
    s2 = torch.maximum((lo + 63) // 64, ((fr - 63).clamp(min=0) + 63) // 64) * 64
    keys = (((lo + 63) // 64) * 64 + ((L - s2).clamp(min=0) + 63) // 64 * 64).sum().item()
    gbs = lambda t: keys * Q.KVH * Q.HD * 2 * 2 / (t * 1e-6) / 1e9
    res = [f'B{B} ctx{ctx} seg{seg}: orig {t1:.1f}us {gbs(t1):.0f}GB/s']
    for stg in (2, 3, 4):
        for nw in (4, 8):
            o2 = torch.zeros_like(o1)
            f2 = lambda: attn_suffix3p_kernel[(B, Q.KVH)](*args(o2), H=Q.H, HD=Q.HD, BN=64, E=E, NSPL=nspl, STG=stg,
                                                         num_warps=nw, num_stages=stg)
            t2 = bench(f2); eq = torch.equal(o1, o2)
            res.append(f's{stg}w{nw} {t2:.1f}us {gbs(t2):.0f}GB/s eq={eq}')
    print(' | '.join(res), flush=True)
print('ATT4DONE')
