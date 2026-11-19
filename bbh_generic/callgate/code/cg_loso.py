"""Leave-one-STYLE-out over POSITIVE groups (own data): train without a positive style group, test its pass rate at the
0.5th-pct rule. Own-data estimate of out-of-style false vetoes. python cg_loso.py S1.pt S2.pt ..."""
import sys, torch
byk = {}
for fi, f in enumerate(sys.argv[1:]):
    for j, r in enumerate(torch.load(f)):
        byk[(r['qid'], r['src']) if r['label'] == 1 else (fi, j)] = r
op = [r for r in byk.values() if r['opened']]
X = torch.stack([r['h'] for r in op]).float(); y = torch.tensor([float(r['label']) for r in op])
grp = lambda r: r['src'].split('/')[0] + '/' + r['src'].split('/')[1] if r['src'].startswith('train') else r['src'].split('/')[0]
G = sorted(set(grp(r) for r in op if r['label'] == 1))
for g in G:
    tr = torch.tensor([grp(r) != g or r['label'] == 0 for r in op]) & torch.tensor([r['split'] == 'train' for r in op])
    te = torch.tensor([grp(r) == g and r['label'] == 1 for r in op])
    mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6; Xn = (X - mu) / sd
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=0.5, max_iter=300, line_search_fn='strong_wolfe')
    def cl():
        opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy_with_logits(Xn[tr] @ w + b, y[tr]) + 1e-2 * (w * w).sum(); l.backward(); return l
    opt.step(cl)
    with torch.no_grad():
        z = Xn @ w + b; pz = z[tr & (y == 1)].sort().values; th = float(pz[max(0, int(0.005 * len(pz)) - 1)])
        print(f'held-out positive group {g:14s}: n={int(te.sum()):3d}  pass={float((z[te] >= th).float().mean()):.3f}')
