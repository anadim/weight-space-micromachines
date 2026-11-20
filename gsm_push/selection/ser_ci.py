# paired bootstrap CI: serial-vote accuracy at N=k (probe vote + sanity) vs plain greedy, same items
import sys, os, numpy as np, torch
O, nm, mode, suf, gfile = sys.argv[1:6]
NN = 8
D = torch.load(f'{O}/ser_{nm}_{mode}_N{NN}_v1{suf}.pt'); R = D['reg']; G = torch.load(gfile)
gold = np.array([float(r['gold']) for r in D['rows']]); n = len(gold)
val, ok, bad, lg = R['val'].numpy(), R['ok'].numpy(), R['bad'].numpy(), R['lg'].numpy()
right = ok & (np.abs(val - gold[:, None]) <= 1e-6 * np.maximum(1, np.abs(gold[:, None])))
def pick(j, k):
    idx = [i for i in range(k) if ok[j, i]]; s = [i for i in idx if not bad[j, i]]; idx = s or idx
    if not idx: return None
    w = {i: 1 / (1 + np.exp(-lg[j, i])) for i in idx}
    sc = {i: sum(w[t] for t in idx if abs(val[j, t] - val[j, i]) <= 1e-6 * max(1, abs(val[j, i]))) for i in idx}
    return max(idx, key=lambda i: (sc[i], -i))
g = np.array(G['ok'], float); rs = np.random.RandomState(0)
for k in (1, 2, 4, 8):
    a = np.array([float(p is not None and right[j, p]) for j, p in enumerate(pick(j, k) for j in range(n))])
    d = a - g; b = d[rs.randint(0, n, (4000, n))].mean(1)
    fx = int(((a == 1) & (g == 0)).sum()); br = int(((a == 0) & (g == 1)).sum())
    print(nm, mode, suf, f'N={k}', f'acc {100*a.mean():.2f} greedy {100*g.mean():.2f} diff {100*d.mean():+.2f} [{100*np.percentile(b,2.5):+.2f}, {100*np.percentile(b,97.5):+.2f}] fixes {fx} breaks {br}')
