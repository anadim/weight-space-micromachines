# lane wpr fuzz: R(-off) applied after the host RoPE at the absolute position (weight-space re-phasing) vs the programmatic
# reset (RoPE at pos - off). Compares attention logits q.k against question keys (positions < qend) and in-segment keys,
# using the engine's bf16 RoPE tables and the comp_serial._wpr_rot writer. Reference noise: bf16 RoPE vs exact fp64.
import sys, os, torch
sys.path.insert(0, os.environ['WIRING_DIR']); sys.path.insert(0, os.environ['STK_DIR'])
import fastqwen as Q
torch.manual_seed(0); dev = 'cuda'; bf = torch.bfloat16
MAXP = 6000
cos_t, sin_t = Q.rope_tables(MAXP, dev, bf)
inv = 1.0 / (Q.THETA ** (torch.arange(0, Q.HD, 2, device=dev).double() / Q.HD))
fr = torch.arange(MAXP, device=dev).double()[:, None] * inv[None]; emb = torch.cat([fr, fr], -1)
wc, ws = emb.cos().float(), emb.sin().float()
def rope_bf(y, p):          # kernel-equivalent rounding (t1, t2 rounded to bf16, sum rounded)
    c, s = cos_t[p][:, None].float(), sin_t[p][:, None].float()
    yf = y.float(); t1 = (yf * c).to(bf).float(); t2 = (Q.rot(yf) * s).to(bf).float()
    return (t1 + t2).to(bf)
def rope_ex(y, p):
    c, s = emb.cos()[p][:, None], emb.sin()[p][:, None]; yd = y.double()
    return yd * c + Q.rot(yd) * s
def wpr(x, off):
    xf = x.float(); c, s = wc[off][:, None], ws[off][:, None]
    return (xf * c - Q.rot(xf) * s).to(x.dtype)
worst = {}
for trial in range(200):
    B = 64
    qend = torch.randint(60, 600, (B,), device=dev)
    off = torch.randint(0, 4000, (B,), device=dev); off[:8] = 0
    p = qend + off + torch.randint(0, 400, (B,), device=dev)          # absolute position of the attempt token
    j = torch.randint(0, 1, (B,), device=dev) + (torch.rand(B, device=dev) * qend.float()).long()   # a question key
    pi = p - torch.randint(0, 300, (B,), device=dev).clamp(max=0) - (torch.rand(B, device=dev) * (p - qend - off).float()).long()  # in-segment key
    yq = (torch.randn(B, 16, Q.HD, device=dev) * 3).to(bf); yk = (torch.randn(B, 1, Q.HD, device=dev) * 3).to(bf)
    qW = wpr(rope_bf(yq, p), off); qR = rope_bf(yq, p - off); qE = rope_ex(yq, p - off)
    kj = rope_bf(yk, j); kjE = rope_ex(yk, j)
    kiW = wpr(rope_bf(yk, pi), off); kiR = rope_bf(yk, pi - off); kiE = rope_ex(yk, pi - off)
    for nm, a_, b_, c_ in (('question', (qW.float() * kj.float()).sum(-1), (qR.float() * kj.float()).sum(-1), (qE * kjE).sum(-1)),
                           ('segment', (qW.float() * kiW.float()).sum(-1), (qR.float() * kiR.float()).sum(-1), (qE * kiE).sum(-1))):
        sc = c_.abs().mean().item()
        e_wr = (a_ - b_).abs().max().item() / sc; e_we = (a_.double() - c_).abs().max().item() / sc; e_re = (b_.double() - c_).abs().max().item() / sc
        w = worst.setdefault(nm, [0, 0, 0]); w[0] = max(w[0], e_wr); w[1] = max(w[1], e_we); w[2] = max(w[2], e_re)
    z = off == 0
    assert torch.equal(qW[z], rope_bf(yq, p)[z]), 'off=0 rows must be bit-identical'
for nm, w in worst.items():
    print(f"WPRFUZZ {nm}: max|logit diff|/mean|logit|  wpr-vs-reset {w[0]:.2e}  wpr-vs-exact {w[1]:.2e}  reset-vs-exact {w[2]:.2e}", flush=True)
print('WPRFUZZ off=0 rows bit-identical: OK')
