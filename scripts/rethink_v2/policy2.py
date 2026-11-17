"""Stage 2b (TRAIN side only): a two-head detector at the read point 'pre' (before '####'): head 0 predicts "the answer
the host is about to commit is correct", head 1 predicts "the answer after a rethink (policy P) is correct". Fire iff
logit1 - logit0 > theta ("rethinking is expected to help"; with theta = 0 this is exactly P(right after) > P(right now)).
The difference of two linear heads over a shared ReLU layer is one linear readout -> same circuit as policy.py.
fit 70 / sel 15 / rep 15 by source problem; theta on SEL greedy pooled (GSM8K-train + scaled-train); REP reported.
python scripts/rethink_v2/policy2.py OUTDIR POLICY LAYERS_JSON HIDDEN"""
import os, sys, json, torch, numpy as np
OUT_, POL, LAYS, HID = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]), int(sys.argv[4])
sys.argv = [sys.argv[0], OUT_]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fit_detector.py')).read().split('# 1) logistic per read point x layer')[0])
greedy = kind == 'greedy'
OUTC = {'A': ('rta_ok', 'extra_rta'), 'B': ('rt_ok', 'extra_rt'), 'B150': ('cap150_ok', 'extra_cap150'), 'B300': ('cap300_ok', 'extra_cap300')}
o_ok = cat(OUTC[POL][0]).astype(bool); ex = cat(OUTC[POL][1]).astype(float); yb = yn.astype(bool)
Y = torch.tensor(np.stack([yb, o_ok], 1), dtype=torch.float32)
X = feats('pre', LAYS)
mu = X[fit].mean(0); sd = X[fit].std(0) + 1e-3; Z = ((X - mu) / sd).to(dev); Yd = Y.to(dev)
def run(wd, drop, epochs=60, lr=3e-4):
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Dropout(drop), torch.nn.Linear(Z.shape[1], HID), torch.nn.ReLU(), torch.nn.Linear(HID, 2)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    fi = torch.tensor(np.where(fit)[0], device=dev); best = None
    for ep in range(epochs):
        net.train(); perm = fi[torch.randperm(len(fi), device=dev)]
        for b in range(0, len(perm), 512):
            ix = perm[b:b + 512]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Z[ix]), Yd[ix])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 4:
            net.eval()
            with torch.no_grad(): L = net(Z).cpu().numpy()
            s = L[:, 1] - L[:, 0]
            # selection metric: best pooled sel-greedy accuracy over theta
            m = sel & greedy; th = np.quantile(s[m], np.linspace(0, 1, 101))
            acc = max(np.where(s[m] > t, o_ok[m], yb[m]).mean() for t in th)
            if best is None or acc > best[0]: best = (acc, {k: v.detach().clone().cpu() for k, v in net.state_dict().items()}, L, ep)
    return best
best = None
for wd in [1e-2, 1e-1]:
    for drop in [0.0, 0.3]:
        r = run(wd, drop); print('wd', wd, 'drop', drop, 'sel acc', round(100 * r[0], 2), 'ep', r[3], flush=True)
        if best is None or r[0] > best[0]: best = r + (wd, drop)
acc_sel, st, L, ep, wd, drop = best
s = L[:, 1] - L[:, 0]; s0 = L[:, 0]
m = sel & greedy; thetas = np.quantile(s[m], np.linspace(0, 1, 401))
accs = np.array([np.where(s[m] > t, o_ok[m], yb[m]).mean() for t in thetas])
bi = int(np.argmax(accs + 1e-9 * np.arange(len(accs))))     # ties -> larger theta (fewer firings)
th = float(thetas[bi])
res = dict(policy=POL, layers=LAYS, hidden=HID, wd=wd, drop=drop, ep=ep, theta=th)
for part, m0 in [('sel', sel), ('rep', rep)]:
    for nm in D:
        mm = m0 & greedy & (src == nm); f = s > th
        res[f'{part}_{nm}'] = dict(n=int(mm.sum()), base=100 * yb[mm].mean(), always=100 * o_ok[mm].mean(), always_extra=ex[mm].mean(),
                                   gated=100 * np.where(f, o_ok, yb)[mm].mean(), gated_extra=(ex * f)[mm].mean(), fire_rate=f[mm].mean(),
                                   auroc_correct=auroc(s0[mm], yb[mm]))
        print(part, nm, {k: round(float(v), 3) for k, v in res[f'{part}_{nm}'].items()}, flush=True)
res['curve_sel'] = [[float(t), float(a)] for t, a in zip(thetas[::8], accs[::8])]
os.makedirs(OUT_, exist_ok=True); json.dump(res, open(f'{OUT_}/policy2_{POL}.json', 'w'), indent=1, default=float)
W2 = (st['3.weight'][1] - st['3.weight'][0]); b2 = float(st['3.bias'][1] - st['3.bias'][0])
torch.save(dict(layers=LAYS, mu=mu, sd=sd, W1=st['1.weight'], b1=st['1.bias'], W2=W2, b2=b2, thr_logit=th, read='pre', policy=POL,
                fit='two-head MLP (P(correct now), P(correct after rethink)) on GSM8K TRAIN + scaled-number TRAIN generations (fit split); theta on sel split; fire iff logit1-logit0 > theta'),
           f'{OUT_}/detector_{POL}.pt')
print('saved', f'{OUT_}/detector_{POL}.pt', 'theta', th)
