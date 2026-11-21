# ladnext fuzz: the three WPR variants (WPR_K = 0 / 1 / 2, comp_serial._step_triton) vs the programmatic reset (AW_PR=1:
# host RoPE at pos - off) and vs exact fp64 RoPE, using the REAL host kernel (hostkern.qk_rope_kernel) on random q/k rows.
# Metric (as implants/wpr/stk/wpr_fuzz.py): max |q.k diff| / mean |q.k| for question keys (absolute position j < qend) and
# in-segment keys (rotated by the same method), plus the fraction of bit-identical q / K elements vs the reset.
import sys, os, torch
sys.path.insert(0, os.environ['WIRING_DIR']); sys.path.insert(0, os.environ['STK_DIR'])
import fastqwen as Q, hostkern as HK
torch.manual_seed(0); dev = 'cuda'; bf = torch.bfloat16
MAXP = 8400
cos_t, sin_t = Q.rope_tables(MAXP, dev, bf)
hc32, hs32 = Q.rope_tables(MAXP, dev, torch.float32)
inv = 1.0 / (Q.THETA ** (torch.arange(0, Q.HD, 2, dtype=torch.int64).to(device=dev, dtype=torch.float) / Q.HD))
fr = torch.arange(MAXP, device=dev, dtype=torch.float64)[:, None] * inv.double()[None]; emb = torch.cat([fr, fr], -1)
wc, ws = emb.cos().float(), emb.sin().float()                   # comp_serial._wpr_tab (implant offset table)
invd = 1.0 / (Q.THETA ** (torch.arange(0, Q.HD, 2, device=dev).double() / Q.HD))
NH = Q.H + 2 * Q.KVH
qn = (1 + 0.3 * torch.randn(Q.HD, device=dev)).to(bf); kn = (1 + 0.3 * torch.randn(Q.HD, device=dev)).to(bf)


def kern(qkv, cos, sin, pos):
    """host kernel: returns q (B,H,HD) and the K row (B,KVH,HD)"""
    B = qkv.shape[0]
    q = torch.empty(B, Q.H, Q.HD, device=dev, dtype=bf)
    Kc = torch.zeros(B, Q.KVH, 1, Q.HD, device=dev, dtype=bf); Vc = torch.zeros_like(Kc)
    slot = torch.zeros(B, device=dev, dtype=torch.long)
    HK.qk_rope_kernel[(B,)](qkv, qn, kn, cos, sin, pos, q, Kc, Vc, slot, Kc.stride(0), Kc.stride(1), Kc.stride(2),
                            H=Q.H, KVH=Q.KVH, HD=Q.HD, eps=Q.EPS)
    return q, Kc[:, :, 0]


def rotq(x, c, s):      # comp_serial._wpr_rot
    xf = x.float()
    return (xf * c - Q.rot(xf) * s).to(x.dtype)


def method(m, qkv, p, off):
    B = qkv.shape[0]; ar = torch.arange(B, device=dev)
    if m == 'reset': return kern(qkv, cos_t, sin_t, p - off)
    if m == 'k0':
        q, k = kern(qkv, cos_t, sin_t, p); c, s = wc[off][:, None], ws[off][:, None]
        return rotq(q, c, s), rotq(k, c, s)
    if m == 'k1':
        hc, hs = hc32[p], hs32[p]; oc, os_ = wc[off], ws[off]
        kc = (hc * oc + hs * os_).to(bf).contiguous(); ks = (hs * oc - hc * os_).to(bf).contiguous()
        return kern(qkv, kc, ks, ar)
    if m == 'k2':
        return kern(qkv, cos_t[p - off].contiguous(), sin_t[p - off].contiguous(), ar)
    if m == 'exact':
        y, yk = kern(qkv, cos_t, sin_t, torch.zeros_like(p))          # pos 0: cos 1, sin 0 -> the normed rows y
        a = (p - off).double()[:, None] * invd[None]; a = torch.cat([a, a], -1)
        c, s = a.cos()[:, None], a.sin()[:, None]
        f = lambda z: z.double() * c + Q.rot(z.double()) * s
        return f(y), f(yk)
    raise ValueError(m)


M = ['k0', 'k1', 'k2']
worst = {}; ident = {m: [0, 0] for m in M}; n0 = {m: True for m in M}
for trial in range(200):
    B = 64
    qend = torch.randint(60, 600, (B,), device=dev)
    off = torch.randint(0, 6000, (B,), device=dev); off[:8] = 0
    p = qend + off + torch.randint(0, 400, (B,), device=dev)
    j = (torch.rand(B, device=dev) * qend.float()).long()                                   # a question key
    pi = qend + off + (torch.rand(B, device=dev) * (p - qend - off + 1).float()).long()     # an in-segment key (abs pos)
    qkv = (torch.randn(B, NH * Q.HD, device=dev) * 2).to(bf); kkv = (torch.randn(B, NH * Q.HD, device=dev) * 2).to(bf)
    jkv = (torch.randn(B, NH * Q.HD, device=dev) * 2).to(bf)
    _, kj = kern(jkv, cos_t, sin_t, j)                                                       # question keys: host RoPE at j
    _, kjE = method('exact', jkv, j, torch.zeros_like(j))
    qR, _ = method('reset', qkv, p, off); _, kiR = method('reset', kkv, pi, off)
    qE, _ = method('exact', qkv, p, off); _, kiE = method('exact', kkv, pi, off)
    for m in M:
        qm, _ = method(m, qkv, p, off); _, kim = method(m, kkv, pi, off)
        ident[m][0] += int((qm == qR).sum()) + int((kim == kiR).sum()); ident[m][1] += qm.numel() + kim.numel()
        z = off == 0
        qP, _ = kern(qkv, cos_t, sin_t, p)
        n0[m] &= bool(torch.equal(qm[z], qP[z]))
        lg = lambda a_, b_: torch.einsum('bhd,bd->bh', a_.double(), b_[:, 0].double())       # q heads vs kv head 0
        for nm, a_, b_, c_ in (('question', lg(qm, kj), lg(qR, kj), lg(qE, kjE)), ('segment', lg(qm, kim), lg(qR, kiR), lg(qE, kiE))):
            sc = c_.abs().mean().item()
            w = worst.setdefault((m, nm), [0, 0, 0])
            w[0] = max(w[0], (a_ - b_).abs().max().item() / sc); w[1] = max(w[1], (a_ - c_).abs().max().item() / sc)
            w[2] = max(w[2], (b_ - c_).abs().max().item() / sc)
for (m, nm), w in sorted(worst.items()):
    print(f"WPRK {m} {nm:8s}: max|logit diff|/mean|logit|  vs-reset {w[0]:.2e}  vs-exact {w[1]:.2e}  (reset-vs-exact {w[2]:.2e})", flush=True)
for m in M:
    print(f"WPRK {m}: q/K elements bit-identical to the reset {100 * ident[m][0] / ident[m][1]:.3f}%  off=0 rows == host RoPE: {n0[m]}")
ok = n0['k1'] and n0['k2'] and ident['k2'][0] == ident['k2'][1]
print('WPRK FUZZ', 'PASS' if ok else 'FAIL', flush=True)
sys.exit(0 if ok else 1)
