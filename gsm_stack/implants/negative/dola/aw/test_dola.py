"""CPU fuzz of aw_dola.dola_logits vs a float64 log-prob reference (DoLa: log p_fin + alpha (log p_fin - log p_l), APC on p_fin).
python3 test_dola.py  (run from the aw dir; STK_DIR or ./stk on the path)"""
import os, sys, math, torch
sys.path.insert(0, os.environ.get('STK_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stk')))
import aw_dola as A
g = torch.Generator().manual_seed(0)
bad = 0; n = 0; maxd = 0.0
for it in range(400):
    B, V, D = int(torch.randint(1, 9, (1,), generator=g)), int(torch.randint(50, 3000, (1,), generator=g)), 64
    E = torch.randn(V, D, generator=g).to(torch.bfloat16)
    w = (1 + 0.1 * torch.randn(D, generator=g)).to(torch.bfloat16)
    hf = (torch.randn(B, D, generator=g) * (1 + 5 * torch.rand(1, generator=g))).to(torch.bfloat16)
    hl = (torch.randn(B, D, generator=g) * 3).to(torch.bfloat16)
    def nrm(x):
        xf = x.float(); y = (xf / torch.sqrt((xf * xf).mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)
        return (w.float() * y.float()).to(torch.bfloat16)
    z = nrm(hf) @ E.T; zl = nrm(hl) @ E.T
    if it % 7 == 0: z[:, :3] = z.max() + 2      # ties at the top
    alpha = [0.1, 0.25, 0.5, 1.0, 2.0][it % 5]; beta = [0.0, 0.1, 0.01, 0.5][it % 4]
    on = torch.rand(B, generator=g) < 0.7
    out = A.dola_logits(z, zl, on, alpha, beta)
    # reference: float64 log-probs
    lp = torch.log_softmax(z.double(), -1); lpl = torch.log_softmax(zl.double(), -1)
    ref = lp + alpha * (lp - lpl)
    if beta > 0:
        keep = lp >= lp.max(-1, keepdim=True).values + math.log(beta)
        band = (lp - (lp.max(-1, keepdim=True).values + math.log(beta))).abs() < 1e-5
        ref = ref.masked_fill(~keep, float('-inf'))
    else:
        band = torch.zeros_like(lp, dtype=torch.bool)
    ref = torch.where(on[:, None], ref, lp)
    for b in range(B):
        n += 1
        ko, kr = torch.isfinite(out[b]), torch.isfinite(ref[b])
        if ((ko != kr) & ~band[b]).any(): bad += 1; continue
        m = ko & kr
        po = torch.softmax(out[b].double(), -1); pr = torch.softmax(ref[b], -1)
        d = (po[m] - pr[m]).abs().max().item(); maxd = max(maxd, d)
        if d > 1e-5: bad += 1
        if not on[b] and not torch.equal(out[b], z[b].float()): bad += 1
# Gumbel-max sampling never returns an APC-removed token
print(f'DOLA FUZZ rows {n} bad {bad} max |p - p_ref| {maxd:.2e}')
assert bad == 0
print('DOLA FUZZ OK')
