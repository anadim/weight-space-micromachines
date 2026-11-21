"""Predictability lane: fit probes on a train set's hidden states, score train (question-grouped 5-fold OOF) and test sets.
Runs on the node (states stay there); writes small score files to out/pred_{tag}/ for the local analysis (predict_ana.py).

    python3 predict_fit.py --train dev --test dev2 --tag cvdev        # CV phase: OOF on dev, dev-fit -> dev2
    python3 predict_fit.py --train trainC --test dev,dev2 --tag trainC

Outputs (out/pred_{tag}/):
  nl_{set}.npz    value probes at reasoning line ends (off >= phrase_len, before '####'): att, j, off, S (n, ncol) logits,
                  cols = per-layer 10..28 + 'c4' (concat 14,18,22,27); label = attempt ok. Train set = OOF logits.
  hash_{set}.npz  '####' verifiers: att, S (n, ncol), cols (per-layer, concat sets, layer-average, stacked, mean-over-line-ends)
  qend.json       question-end difficulty probes (ridge R^2 for p_right, AUROC never/always/mixed vs rest)
  probes.pt       weights (input units) of the full-train fits: value per layer + c4, hash per layer + c4 + stack
"""
import os, json, gzip, argparse, time
import numpy as np, torch
import fit

LAY = list(range(10, 29)); C4 = [14, 18, 22, 27]; C6 = [12, 15, 18, 21, 24, 27]
ap = argparse.ArgumentParser()
ap.add_argument('--train', default='dev'); ap.add_argument('--test', default='dev2'); ap.add_argument('--tag', default='cvdev')
ap.add_argument('--folds', type=int, default=5); ap.add_argument('--skip', default='')
a = ap.parse_args()
TEST = [s for s in a.test.split(',') if s]; SETS = [a.train] + TEST
OD = f'out/pred_{a.tag}'; os.makedirs(OD, exist_ok=True)
T0 = time.time()
def log(*x): print(f'[{time.time() - T0:6.0f}s]', *x, flush=True)


def light(S):
    with gzip.open(f'{fit.ROOT}/{S}/attempts_light.jsonl.gz', 'rt') as f:
        return [json.loads(l) for l in f]


ATT = {S: dict(np.load(f'{fit.ROOT}/{S}/attempts.npz')) for S in SETS}
for S in SETS:
    L_ = light(S); ATT[S]['phrase_len'] = np.array([int(r.get('phrase_len') or 0) for r in L_], np.int32)


def folds_of(groups, k=a.folds, seed=0):
    ug = np.unique(groups); rng = np.random.default_rng(seed); rng.shuffle(ug)
    m = {g: i % k for i, g in enumerate(ug)}; return np.array([m[g] for g in groups])


def oof_and_full(Xtr, y, groups, Xtests, l2, w=None):
    """-> oof logits on train, list of test logits (full-train fit), (W, b) of full fit."""
    f = folds_of(groups); oof = np.zeros(len(y), np.float32)
    for k in range(a.folds):
        tr, te = f != k, f == k
        p, _, _ = fit.fit_logreg(Xtr[tr], y[tr], l2=l2, w=None if w is None else w[tr]); oof[te] = p(Xtr[te])
    p, W, b = fit.fit_logreg(Xtr, y, l2=l2, w=w)
    return oof, [p(X) for X in Xtests], (W, b)


def wauc(y, s, idx):
    """within-item AUROC: pooled concordance over (right, wrong) pairs of the same item."""
    num = den = 0.0
    o = np.argsort(idx, kind='stable'); y, s, idx = y[o], s[o], idx[o]
    cuts = np.flatnonzero(np.diff(idx)) + 1
    for ys, ss in zip(np.split(y, cuts), np.split(s, cuts)):
        p = ss[ys == 1]; n = ss[ys == 0]
        if not len(p) or not len(n): continue
        d = p[:, None] - n[None, :]; num += (d > 0).sum() + 0.5 * (d == 0).sum(); den += d.size
    return num / den if den else float('nan')


PROBES = {}
# ---------------------------------------------------------------- value probes at line ends
def nl_mask(S):
    pm = np.load(f'{fit.ROOT}/{S}/pos_meta.npz'); A = ATT[S]
    k = pm['kind'] == 1; att = pm['att'][k]; off = pm['off'][k]
    ho = A['hash_off'][att]
    return (off >= A['phrase_len'][att]) & ((ho < 0) | (off < ho))


def step_of(m):
    att = m['att']; o = np.lexsort((m['off'], att)); st = np.zeros(len(att), np.int64)
    sa = att[o]; first = np.r_[0, np.flatnonzero(sa[1:] != sa[:-1]) + 1]
    run = np.arange(len(o)) - np.repeat(first, np.diff(np.r_[first, len(o)])); st[o] = run
    return st


def within_fit(X, m, smax, Xtests, l2=1e-1):
    """within-item value probe: trained on mixed items' line ends, states demeaned per (item, step) group (>= 2 attempts),
    so only the right-vs-wrong direction among attempts of the same item at the same step is learned. Scores raw states."""
    st = step_of(m); idx = m['idx']; ok = m['ok']
    pr = {i: ok[idx == i].mean() for i in np.unique(idx)}
    mixed = np.array([0 < pr[i] < 1 for i in idx]); sel = np.flatnonzero(mixed & (st <= smax))
    key = idx[sel].astype(np.int64) * 16 + np.minimum(st[sel], 10)
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    keep = cnt[inv] >= 2; sel, inv = sel[keep], inv[keep]
    Xt = torch.as_tensor(X[sel]).float(); it = torch.as_tensor(inv, dtype=torch.long)
    mu = torch.zeros(len(uk), Xt.shape[1]).index_add_(0, it, Xt) / torch.as_tensor(cnt, dtype=torch.float32)[:, None]
    Xd = (Xt - mu[it]).numpy(); del Xt
    y = ok[sel].astype(np.float32); att = m['att'][sel]; w = 1.0 / np.bincount(att)[att]
    f_all = folds_of(idx); f = f_all[sel]; oof = np.zeros(len(X), np.float32)
    for k in range(a.folds):
        p, _, _ = fit.fit_logreg(Xd[f != k], y[f != k], l2=l2, w=w[f != k]); oof[f_all == k] = p(X[f_all == k])
    p, W, b = fit.fit_logreg(Xd, y, l2=l2, w=w)
    return oof, [p(Xn) for Xn in Xtests], (W, b)


if 'nl' not in a.skip:
    M = {S: nl_mask(S) for S in SETS}
    WCOLS = {'w14': ([14], 99), 'w18': ([18], 99), 'w22': ([22], 99), 'w27': ([27], 99), 'wc4': (C4, 99),
             'we18': ([18], 3), 'we22': ([22], 3), 'wec4': (C4, 3)}   # within-item probes: (layers, max step trained on)
    cols = [str(L) for L in LAY] + ['c4'] + list(WCOLS)
    out = {S: None for S in SETS}; means = {S: {} for S in SETS}
    PROBES['nl'] = {}
    for ci, c in enumerate(cols):
        Ls = WCOLS[c][0] if c in WCOLS else C4 if c == 'c4' else [int(c)]
        Xs = {}; mt = {}
        for S in SETS:
            X, m = fit.load(S, Ls, 'nl', rows=M[S]); Xs[S] = X.reshape(len(X), -1); mt[S] = m
            if c not in WCOLS and c != 'c4':                  # per-attempt mean of line-end states (for the hash lane)
                att = m['att']; n_att = len(ATT[S]['ok'])
                cnt = torch.bincount(torch.as_tensor(att, dtype=torch.long), minlength=n_att).float()
                acc = torch.zeros(n_att, X.shape[-1]).index_add_(0, torch.as_tensor(att, dtype=torch.long), torch.as_tensor(X[:, 0]).float())
                means[S][int(c)] = (acc / cnt.clamp(min=1)[:, None]).half().numpy()
            if out[S] is None:
                out[S] = dict(att=m['att'], j=m['j'], off=m['off'], S=np.zeros((len(X), len(cols)), np.float32))
        y = mt[a.train]['ok'].astype(np.float32); g = mt[a.train]['idx']; att = mt[a.train]['att']
        w = 1.0 / np.bincount(att)[att]                       # each attempt weighs 1
        if c in WCOLS:
            oof, tes, (W, b) = within_fit(Xs[a.train], mt[a.train], WCOLS[c][1], [Xs[S] for S in TEST])
        else:
            oof, tes, (W, b) = oof_and_full(Xs[a.train], y, g, [Xs[S] for S in TEST], l2=1e-2, w=w)
        out[a.train]['S'][:, ci] = oof
        for S, t in zip(TEST, tes): out[S]['S'][:, ci] = t
        PROBES['nl'][c] = dict(layers=Ls, w=W.half(), b=b)
        log(f'nl {c}: train OOF AUROC {fit.auroc(y, oof):.4f} | ' + ' '.join(
            f'{S} {fit.auroc(mt[S]["ok"], t):.4f}/w{wauc(mt[S]["ok"], t, mt[S]["idx"]):.4f}' for S, t in zip(TEST, tes)))
        del Xs
    for S in SETS: np.savez_compressed(f'{OD}/nl_{S}.npz', cols=np.array(cols), **out[S])
    MEANS = means
else:
    MEANS = None

# ---------------------------------------------------------------- '####' verifiers
if 'hash' not in a.skip:
    H = {}; HM = {}
    for S in SETS:
        X, m = fit.load(S, LAY, 'hash'); H[S] = X; HM[S] = m
    y = HM[a.train]['ok'].astype(np.float32); g = HM[a.train]['idx']
    res = {S: {} for S in SETS}; PROBES['hash'] = {}
    def run(name, F, l2s=(1e-2,), keep=False):
        best = None
        for l2 in l2s:
            oof, tes, (W, b) = oof_and_full(F[a.train], y, g, [F[S] for S in TEST], l2=l2)
            au = fit.auroc(y, oof)
            if best is None or au > best[0]: best = (au, l2, oof, tes, W, b)
        au, l2, oof, tes, W, b = best
        res[a.train][name] = oof
        for S, t in zip(TEST, tes): res[S][name] = t
        if keep: PROBES['hash'][name] = dict(w=W.half(), b=b, l2=l2)
        log(f'hash {name} (l2 {l2:g}): train OOF {au:.4f} | ' + ' '.join(
            f'{S} {fit.auroc(HM[S]["ok"], t):.4f}/w{wauc(HM[S]["ok"], t, HM[S]["idx"]):.4f}' for S, t in zip(TEST, tes)))
    for i, L in enumerate(LAY):
        run(f'L{L}', {S: H[S][:, i] for S in SETS}, l2s=(1e-2, 1e-1, 1.0), keep=True)
    sub = lambda Ls: {S: H[S][:, [LAY.index(L) for L in Ls]].reshape(len(H[S]), -1) for S in SETS}
    run('c4', sub(C4), l2s=(1e-2, 1e-1, 1.0), keep=True)
    run('c6', sub(C6), l2s=(1e-2, 1e-1, 1.0))
    for name, F in (('w_L18', {S: H[S][:, LAY.index(18)] for S in SETS}), ('w_L22', {S: H[S][:, LAY.index(22)] for S in SETS}),
                    ('w_c4', sub(C4))):                       # within-item (per-item demeaned, mixed items) '####' probes
        mm = dict(idx=HM[a.train]['idx'], ok=HM[a.train]['ok'], att=HM[a.train]['att'], off=np.zeros(len(y), np.int64))
        oof, tes, (W, b) = within_fit(F[a.train], mm, 99, [F[S] for S in TEST], l2=1.0 if name == 'w_c4' else 1e-1)
        res[a.train][name] = oof
        for S, t in zip(TEST, tes): res[S][name] = t
        PROBES['hash'][name] = dict(w=W.half(), b=b)
        log(f'hash {name}: train OOF {fit.auroc(y, oof):.4f} | ' + ' '.join(
            f'{S} {fit.auroc(HM[S]["ok"], t):.4f}/w{wauc(HM[S]["ok"], t, HM[S]["idx"]):.4f}' for S, t in zip(TEST, tes)))
    run('call', sub(LAY), l2s=(1e-1, 1.0, 10.0), keep=True)
    run('avg14_27', {S: H[S][:, LAY.index(14):LAY.index(27) + 1].astype(np.float32).mean(1) for S in SETS}, l2s=(1e-3, 1e-2, 1e-1))
    # stacking: logistic over the per-layer OOF logits (+ register lg)
    ST = {S: np.stack([res[S][f'L{L}'] for L in LAY], 1) for S in SETS}
    run('stack', ST, l2s=(1e-3, 1e-2, 1e-1), keep=True)
    lgf = {S: np.nan_to_num(HM[S]['lg'], nan=-30.0)[:, None] for S in SETS}
    run('stack_lg', {S: np.concatenate([ST[S], lgf[S]], 1) for S in SETS}, l2s=(1e-3, 1e-2, 1e-1), keep=True)
    if MEANS is not None:                                     # mean of line-end states of the attempt (hash rows' attempts)
        MN = {S: np.stack([MEANS[S][L][HM[S]['att']] for L in LAY], 1) for S in SETS}
        for L in (14, 18, 22, 27):
            run(f'mnl_L{L}', {S: MN[S][:, LAY.index(L)] for S in SETS}, l2s=(1e-2, 1e-1))
        run('mnl_c4', {S: MN[S][:, [LAY.index(L) for L in C4]].reshape(len(MN[S]), -1) for S in SETS}, l2s=(1e-2, 1e-1, 1.0), keep=True)
        run('hash_mnl_c4', {S: np.concatenate([sub(C4)[S], MN[S][:, [LAY.index(L) for L in C4]].reshape(len(MN[S]), -1)], 1)
                            for S in SETS}, l2s=(1e-1, 1.0), keep=True)
        del MN
    for S in SETS:
        names = list(res[S]); np.savez_compressed(f'{OD}/hash_{S}.npz', att=HM[S]['att'], cols=np.array(names),
                                                  S=np.stack([res[S][n] for n in names], 1).astype(np.float32))
    del H

# ---------------------------------------------------------------- question end: difficulty
def ridge_oof(X, y, g, Xtests, alphas=(1e2, 1e3, 1e4, 1e5, 1e6)):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    def fitr(Xa, ya, al):
        Xt = torch.as_tensor(Xa, dtype=torch.float64, device=dev); mu = Xt.mean(0); sd = Xt.std(0) + 1e-4; Z = (Xt - mu) / sd
        yt = torch.as_tensor(ya, dtype=torch.float64, device=dev); ym = yt.mean()
        W = torch.linalg.solve(Z.T @ Z + al * torch.eye(Z.shape[1], device=dev, dtype=torch.float64), Z.T @ (yt - ym))
        return lambda Xn: (((torch.as_tensor(Xn, dtype=torch.float64, device=dev) - mu) / sd) @ W + ym).cpu().numpy()
    f = folds_of(g); best = None
    for al in alphas:
        oof = np.zeros(len(y))
        for k in range(a.folds):
            oof[f == k] = fitr(X[f != k], y[f != k], al)(X[f == k])
        r2 = 1 - ((oof - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        if best is None or r2 > best[0]: best = (r2, al, oof)
    p = fitr(X, y, best[1]); return best[2], [p(Xt) for Xt in Xtests], best[1]


if 'qend' not in a.skip:
    Q = {}; QM = {}
    for S in SETS:
        X, m = fit.load(S, LAY, 'qend'); Q[S] = X.astype(np.float32); QM[S] = m
    r2 = lambda y, p: float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    out = {}
    feats = {f'L{L}': (lambda S, i=i: Q[S][:, i]) for i, L in enumerate(LAY)}
    feats['c3'] = lambda S: Q[S][:, [LAY.index(L) for L in (14, 18, 22)]].reshape(len(Q[S]), -1)
    feats['qlen'] = lambda S: QM[S]['qlen'][:, None].astype(np.float32)
    for name, F in feats.items():
        yp = QM[a.train]['p_right'].astype(np.float64); g = QM[a.train]['idx']
        oof, tes, al = ridge_oof(F(a.train), yp, g, [F(S) for S in TEST])
        row = dict(alpha=al, r2={a.train + '_oof': r2(yp, oof), **{S: r2(QM[S]['p_right'], t) for S, t in zip(TEST, tes)}},
                   spearman_like_auc_mixed={}, auc={})
        for cname, cf in (('never', lambda p: p == 0), ('always', lambda p: p == 1), ('mixed', lambda p: (p > 0) & (p < 1)),
                          ('p_ge_half', lambda p: p >= 0.5)):
            yc = cf(QM[a.train]['p_right']).astype(np.float32)
            if yc.sum() < 5: continue
            if name == 'qlen':
                tr_s = -F(a.train)[:, 0]; te_s = [-F(S)[:, 0] for S in TEST]
            else:
                tr_s, te_s, _ = oof_and_full(F(a.train), yc, g, [F(S) for S in TEST], l2=1e-1)
            row['auc'][cname] = {a.train + '_oof': fit.auroc(yc, tr_s), **{S: fit.auroc(cf(QM[S]['p_right']), t) for S, t in zip(TEST, te_s)}}
        out[name] = row
        log('qend', name, json.dumps(row))
    json.dump(out, open(f'{OD}/qend.json', 'w'), indent=1)

torch.save(PROBES, f'{OD}/probes.pt')
log('done')
