"""Own-data model selection for the probe (no BBH, no A_oos2/B_oos/C_ood): for each (lambda, threshold rule) compute
leave-one-STYLE-out positive pass (out-of-style legit calls) and leave-one-GENRE-out negative veto (unseen puzzle genres)."""
import sys, torch
byk = {}
for fi, f in enumerate(sys.argv[1:]):
    for j, r in enumerate(torch.load(f)):
        byk[(r['qid'], r['src']) if r['label'] == 1 else (fi, j)] = r
op = [r for r in byk.values() if r['opened']]
X = torch.stack([r['h'] for r in op]).float(); y = torch.tensor([float(r['label']) for r in op])
trs = torch.tensor([r['split'] == 'train' for r in op])
pg = lambda r: (r['src'].split('/')[0] + '/' + r['src'].split('/')[1]) if r['src'].startswith('train') else r['src'].split('/')[0]
ng = lambda r: r.get('gen') or r['src'].split('/')[1]          # p3 genre, or gen/puz
PG = sorted(set(pg(r) for r in op if r['label'] == 1)); NG = sorted(set(ng(r) for r in op if r['label'] == 0))
def fit(tr, lam):
    mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6; Xn = (X - mu) / sd
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=0.5, max_iter=300, line_search_fn='strong_wolfe')
    def cl():
        opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy_with_logits(Xn[tr] @ w + b, y[tr]) + lam * (w * w).sum(); l.backward(); return l
    opt.step(cl)
    with torch.no_grad(): return Xn @ w + b
def theta(z, tr, rule):
    pz = z[tr & (y == 1)].sort().values; p05 = float(pz[max(0, int(0.005 * len(pz)) - 1)])
    if rule == 'p0.5': return p05
    nmax = float(z[tr & (y == 0)].max())
    return (p05 + nmax) / 2 if rule == 'mid' else nmax + 0.25 * (p05 - nmax)   # 'q1': quarter of the gap above neg max
for lam in (1e-2, 1e-1, 1.0):
    for rule in ('p0.5', 'mid', 'q1'):
        pp = []; nv = []
        for g in PG:
            tr = trs & torch.tensor([not (r['label'] == 1 and pg(r) == g) for r in op]); te = torch.tensor([r['label'] == 1 and pg(r) == g for r in op])
            z = fit(tr, lam); pp.append((float((z[te] >= theta(z, tr, rule)).float().mean()), int(te.sum()), g))
        for g in NG:
            tr = trs & torch.tensor([not (r['label'] == 0 and ng(r) == g) for r in op]); te = torch.tensor([r['label'] == 0 and ng(r) == g for r in op])
            z = fit(tr, lam); nv.append((float((z[te] < theta(z, tr, rule)).float().mean()), int(te.sum()), g))
        mp = sum(a * n for a, n, _ in pp) / sum(n for _, n, _ in pp); mn = sum(a * n for a, n, _ in nv) / sum(n for _, n, _ in nv)
        print(f'lam={lam:<5} rule={rule:4s}  LOSO pos-pass={mp:.3f} (min {min(pp)[0]:.2f} {min(pp)[2]})  LOGO neg-veto={mn:.3f} (min {min(nv)[0]:.2f} {min(nv)[2]})', flush=True)
