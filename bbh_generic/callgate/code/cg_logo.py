"""Leave-one-genre-out: train the probe without one generated puzzle genre (all its contexts, both splits), test veto on it."""
import sys, torch
byk = {}
for fi, f in enumerate(sys.argv[1:]):
    for j, r in enumerate(torch.load(f)):
        byk[(r['qid'], r['src']) if r['label'] == 1 else (fi, j)] = r
op = [r for r in byk.values() if r['opened']]
X = torch.stack([r['h'] for r in op]).float(); y = torch.tensor([float(r['label']) for r in op])
gens = sorted(set(r.get('gen') for r in op if r.get('gen')))
for g in gens:
    tr = torch.tensor([r.get('gen') != g and r['split'] == 'train' for r in op]); te = torch.tensor([r.get('gen') == g for r in op])
    mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6; Xn = (X - mu) / sd
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=0.5, max_iter=300, line_search_fn='strong_wolfe')
    def cl():
        opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy_with_logits(Xn[tr] @ w + b, y[tr]) + 1e-2 * (w * w).sum(); l.backward(); return l
    opt.step(cl)
    with torch.no_grad():
        z = Xn @ w + b; pz = z[tr & (y == 1)].sort().values; th = float(pz[max(0, int(0.005 * len(pz)) - 1)])
        print(f'held-out genre {g:10s}: n={int(te.sum()):3d}  veto={float((z[te] < th).float().mean()):.3f}  (neg max z {float(z[te].max()):.1f} vs theta {th:.1f})')
