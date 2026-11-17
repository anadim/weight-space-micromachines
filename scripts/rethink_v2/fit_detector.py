"""Mistake detector for the v2 rethink implant: fit + select on GSM8K TRAIN and the scaled-number TRAIN analogue only.
Problems are split by source train problem (a scaled problem stays with its source) into fit 70% / sel 15% / rep 15%:
fit = parameters, sel = all choices (layer, read point, model, threshold), rep = the held-out numbers reported.
python scripts/rethink_v2/fit_detector.py OUTDIR"""
import os, sys, json, hashlib, torch, numpy as np
torch.manual_seed(0)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); dev = 'cuda'
shots = json.load(open('data/gsm8k/shots_v1.json')); sq = {s['question'] for s in shots}
train = [json.loads(l) for l in open('data/gsm8k/train.jsonl')]
tr_f = [r for r in train if r['question'] not in sq]
scaled = [json.loads(l) for l in open('data/gsm8k/train_scaled.jsonl')]
def split_of(qtext):
    h = int(hashlib.md5(qtext.encode()).hexdigest(), 16) % 100
    return 'fit' if h < 70 else ('sel' if h < 85 else 'rep')
D = {}
for name, gfile, sfile, grp in [('gsm', 'gen_train_gsm.json', 'states_gsm.pt', lambda k: tr_f[k]['question']),
                                ('scaled', 'gen_train_scaled.json', 'states_scaled.pt', lambda k: train[scaled[k]['src']]['question'])]:
    if not os.path.exists(f'results/rethink_v2/{sfile}'): print('missing', sfile); continue
    g = json.load(open(f'results/rethink_v2/{gfile}')); S = torch.load(f'results/rethink_v2/{sfile}')
    sols = [g['sols'][j] for j in S['sol_index']]
    D[name] = dict(X=S['X'], layers=S['layers'], pos=S['positions'], has_rt=S['has_rt'],
                   y=np.array([s['ok'] for s in sols]), kind=np.array([s['kind'] for s in sols]),
                   split=np.array([split_of(grp(s['q'])) for s in sols]), q=np.array([s['q'] for s in sols]),
                   rt_ok=np.array([s.get('rt_ok', False) for s in sols]),
                   rta_ok=np.array([s.get('rta_ok', s['ok']) for s in sols]),
                   cap150_ok=np.array([s.get('cap150_ok', s['ok']) for s in sols]), cap300_ok=np.array([s.get('cap300_ok', s['ok']) for s in sols]),
                   extra_rta=np.array([s.get('rta_len', len(s['ids'])) - len(s['ids']) for s in sols]),
                   extra_cap150=np.array([s.get('cap150_len', len(s['ids'])) - len(s['ids']) for s in sols]),
                   extra_cap300=np.array([s.get('cap300_len', len(s['ids'])) - len(s['ids']) for s in sols]),
                   extra_rt=np.array([len(s.get('rt_ids', s['ids'])) - len(s['ids']) for s in sols]),
                   n_problems=g['n_problems'])
    nh = sum(1 for s in g['sols'] if s.get('hash_pos') is None)
    print(name, 'solutions', len(g['sols']), 'with ####', len(sols), 'no ####', nh, 'acc(with ####)', D[name]['y'].mean())
LAY = D['gsm']['layers']; POS = D['gsm']['pos']
cat = lambda k: np.concatenate([D[n][k] for n in D])
y_all = torch.tensor(cat('y'), dtype=torch.float32); spl = cat('split'); kind = cat('kind')
src = np.concatenate([[n] * len(D[n]['y']) for n in D])


def auroc(s, y):
    s = np.asarray(s, dtype=np.float64); y = np.asarray(y, dtype=bool)
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    # ties: average ranks
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r); r = (sums / cnt)[inv]
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0)


def feats(p, ls):
    return torch.cat([torch.cat([D[n]['X'][:, POS.index(p), LAY.index(l)].float() for n in D]) for l in ls], 1)


def train_lin(X, y, fit, sel, wd, hidden=0, epochs=60, lr=1e-3, drop=0.0):
    mu = X[fit].mean(0); sd = X[fit].std(0) + 1e-3
    Z = ((X - mu) / sd).to(dev); yy = y.to(dev)
    d = Z.shape[1]
    if hidden: net = torch.nn.Sequential(torch.nn.Dropout(drop), torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1)).to(dev)
    else: net = torch.nn.Linear(d, 1).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    fi = torch.tensor(np.where(fit)[0], device=dev); best = (-1, None)
    for ep in range(epochs):
        net.train(); perm = fi[torch.randperm(len(fi), device=dev)]
        for b in range(0, len(perm), 512):
            ix = perm[b:b + 512]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Z[ix]).squeeze(-1), yy[ix])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 4 or ep == epochs - 1:
            net.eval()
            with torch.no_grad(): s = net(Z).squeeze(-1).cpu().numpy()
            a = auroc(s[sel], y.numpy()[sel])
            if a > best[0]: best = (a, {k: v.detach().clone().cpu() for k, v in net.state_dict().items()}, s, ep)
    return dict(auc_sel=best[0], state=best[1], score=best[2], ep=best[3], mu=mu, sd=sd, hidden=hidden)


fit = spl == 'fit'; sel = spl == 'sel'; rep = spl == 'rep'
res = {'n': {k: int(v) for k, v in [('fit', fit.sum()), ('sel', sel.sum()), ('rep', rep.sum())]}}
yn = y_all.numpy()
def aucs(score):
    out = {}
    for part, m in [('sel', sel), ('rep', rep)]:
        for nm in D:
            mm = m & (src == nm); out[f'{part}_{nm}'] = auroc(score[mm], yn[mm])
            mg = mm & (kind == 'greedy'); out[f'{part}_{nm}_greedy'] = auroc(score[mg], yn[mg])
    return out
# 1) logistic per read point x layer
sweep = {}
for p in ['pre', 'hash', 'end', 'mean']:
    for l in LAY:
        X = feats(p, [l]); best = None
        for wd in [1e-2, 1e-1, 1.0]:
            r = train_lin(X, y_all, fit, sel, wd, epochs=30)
            if best is None or r['auc_sel'] > best['auc_sel']: best = dict(r, wd=wd)
        sweep[f'{p}_L{l}'] = dict(aucs(best['score']), wd=best['wd'])
        print(p, l, {k: round(v, 3) for k, v in sweep[f'{p}_L{l}'].items() if 'sel' in k or k == 'wd'}, flush=True)
res['logistic_sweep'] = sweep
json.dump(res, open(f'{OUT}/detector_sweep.json', 'w'), indent=1)
