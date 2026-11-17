"""Stage 2c (TRAIN side only): HYBRID mistake detector + rethink. One MLP over the host states at the read point 'pre'
(the token before '####', before the answer is committed) with three heads: P(correct now), P(correct after a rethink
written INSTEAD of '####' with a CAP-token budget), P(correct after a rethink written after the committed answer).
  sB = l1 - l0 > thB  -> rethink at '####' (before committing; B-CAP)
  else sA = l2 - l0 > thA -> rethink after the committed answer (A, the v1 point; the first answer stays as fallback)
  else no rethink.
Outcomes are exact for this policy: when B does not fire the generation is the base one, so A's outcome applies.
fit 70 / sel 15 / rep 15 by source problem; (thB, thA) grid-searched on SEL greedy pooled (GSM8K-train + scaled-train)
for max accuracy; REP reported.   python scripts/rethink_v2/policy3.py OUTDIR CAP LAYERS_JSON HIDDEN"""
import os, sys, json, torch, numpy as np
OUT_, CAP, LAYS, HID = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]), int(sys.argv[4])
sys.argv = [sys.argv[0], OUT_]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fit_detector.py')).read().split('# 1) logistic per read point x layer')[0])
greedy = kind == 'greedy'
oB = cat(f'cap{CAP}_ok').astype(bool); exB = cat(f'extra_cap{CAP}').astype(float)
oA = cat('rta_ok').astype(bool); exA = cat('extra_rta').astype(float); yb = yn.astype(bool)
Y = torch.tensor(np.stack([yb, oB, oA], 1), dtype=torch.float32)
X = feats('pre', LAYS)
mu = X[fit].mean(0); sd = X[fit].std(0) + 1e-3; Z = ((X - mu) / sd).to(dev); Yd = Y.to(dev)
def outcome(sB, sA, tB, tA, m=None):
    m = np.ones(len(yb), bool) if m is None else m
    fB = sB > tB; fA = ~fB & (sA > tA)
    return np.where(fB, oB[m], np.where(fA, oA[m], yb[m])), np.where(fB, exB[m], np.where(fA, exA[m], 0.0)), fB, fA
def grid(L, m):
    sB = L[:, 1] - L[:, 0]; sA = L[:, 2] - L[:, 0]
    tBs = np.quantile(sB[m], np.linspace(0, 1, 41)); tAs = np.quantile(sA[m], np.linspace(0, 1, 41))
    best = None
    for tB in tBs:
        for tA in tAs:
            o, e, _, _ = outcome(sB[m], sA[m], tB, tA, m)
            k = (o.mean(), -e.mean())
            if best is None or k > best[0]: best = (k, float(tB), float(tA))
    return best
def run(wd, drop, epochs=60, lr=3e-4):
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Dropout(drop), torch.nn.Linear(Z.shape[1], HID), torch.nn.ReLU(), torch.nn.Linear(HID, 3)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    fi = torch.tensor(np.where(fit)[0], device=dev); best = None
    for ep in range(epochs):
        net.train(); perm = fi[torch.randperm(len(fi), device=dev)]
        for b in range(0, len(perm), 512):
            ix = perm[b:b + 512]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Z[ix]), Yd[ix])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 9:
            net.eval()
            with torch.no_grad(): L = net(Z).cpu().numpy()
            g = grid(L, sel & greedy)
            if best is None or g[0] > best[0][0]: best = (g, {k: v.detach().clone().cpu() for k, v in net.state_dict().items()}, L, ep)
    return best
best = None
for wd in [1e-2, 1e-1]:
    for drop in [0.0, 0.3]:
        r = run(wd, drop); print('wd', wd, 'drop', drop, 'sel acc', round(100 * r[0][0][0], 2), 'ep', r[3], flush=True)
        if best is None or r[0][0] > best[0][0]: best = r + (wd, drop)
(g, st, L, ep, wd, drop) = best; _, tB, tA = g
sB = L[:, 1] - L[:, 0]; sA = L[:, 2] - L[:, 0]
res = dict(cap=CAP, layers=LAYS, hidden=HID, wd=wd, drop=drop, ep=ep, thB=tB, thA=tA)
o, e, fB, fA = outcome(sB, sA, tB, tA)
for part, m0 in [('sel', sel), ('rep', rep)]:
    for nm in D:
        mm = m0 & greedy & (src == nm)
        res[f'{part}_{nm}'] = dict(n=int(mm.sum()), base=100 * yb[mm].mean(), alwaysA=100 * oA[mm].mean(), alwaysA_extra=exA[mm].mean(),
                                   alwaysB=100 * oB[mm].mean(), alwaysB_extra=exB[mm].mean(),
                                   hybrid=100 * o[mm].mean(), hybrid_extra=e[mm].mean(), fireB=fB[mm].mean(), fireA=fA[mm].mean(),
                                   auroc_correct=auroc(L[mm, 0], yb[mm]))
        print(part, nm, {k: round(float(v), 3) for k, v in res[f'{part}_{nm}'].items()}, flush=True)
os.makedirs(OUT_, exist_ok=True); json.dump(res, open(f'{OUT_}/policy3_cap{CAP}.json', 'w'), indent=1, default=float)
W = st['3.weight']; b = st['3.bias']
torch.save(dict(layers=LAYS, mu=mu, sd=sd, W1=st['1.weight'], b1=st['1.bias'],
                WB=W[1] - W[0], bB=float(b[1] - b[0]), thB=tB, WA=W[2] - W[0], bA=float(b[2] - b[0]), thA=tA, cap=int(CAP), read='pre',
                W0=W[0], b0=float(b[0]),
                fit='3-head MLP (P(correct now), P(correct after rethink at ####, cap), P(correct after rethink after the answer)) on GSM8K TRAIN + scaled-number TRAIN generations (fit split); thresholds on sel split'),
           f'{OUT_}/detector_hybrid_cap{CAP}.pt')
print('saved', tB, tA)
