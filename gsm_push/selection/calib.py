"""Threshold calibration for the verifier-triggered rethink, on split C ONLY.
Probe = logistic regression on layer-18 states at '####' of split-C T=1 samples (as in analyze.py, L18, C=0.001).
The greedy split-C answers get OUT-OF-FOLD logits (5 folds grouped by question: the fold model never saw that question),
and the threshold is set so that, among greedy answers the probe flags (and sanity does not), a target share is wrong.
Writes ver_probe.pt: full-fit w, b (standardisation folded) + thresholds per precision target.
usage: python calib.py OUTDIR
"""
import sys, json, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
O = sys.argv[1]; LI = 1; CC = 0.001          # layers (14, 18, 22) -> index 1 = L18
T = torch.load(f'{O}/trainC_T1.0_G8_s0.pt'); Gd = torch.load(f'{O}/trainC_greedy_G1_s0.pt')
assert Gd['q'] == T['q']


def XY(D):
    H = D['hcap'][:, LI].float().numpy(); has = D['hpos'].numpy() >= 0
    return H, np.array(D['ok']), has


X, y, has = XY(T); qid = np.arange(len(y)) // T['G']
Xg, yg, hasg = XY(Gd); flg = np.array(Gd['flag'])
fold = np.random.RandomState(0).permutation(qid.max() + 1) % 5
oof = np.full(len(yg), np.nan)
for f in range(5):
    tr = (fold[qid] != f) & has
    sc = StandardScaler().fit(X[tr]); m = LogisticRegression(C=CC, max_iter=3000).fit(sc.transform(X[tr]), y[tr])
    te = (fold == f) & hasg
    oof[te] = m.decision_function(sc.transform(Xg[te]))
sc = StandardScaler().fit(X[has]); m = LogisticRegression(C=CC, max_iter=3000).fit(sc.transform(X[has]), y[has])
w = m.coef_[0] / sc.scale_; b = m.intercept_[0] - (m.coef_[0] * sc.mean_ / sc.scale_).sum()
from sklearn.metrics import roc_auc_score
ok = hasg
print('split C greedy: acc %.2f, flagged %d, OOF AUROC (greedy rows) %.3f' % (100 * yg.mean(), (flg > 0).sum(), roc_auc_score(yg[ok], oof[ok])))
res = {}
cand = np.sort(oof[ok & (flg == 0)])
for tgt in (0.5, 0.6, 0.7, 0.8):
    best = None
    for thr in cand:                                     # largest threshold (most fires) whose precision >= tgt
        sel = ok & (flg == 0) & (oof < thr + 1e-9)
        if sel.sum() < 10: continue
        prec = 1 - yg[sel].mean()
        if prec >= tgt: best = (float(thr), int(sel.sum()), float(prec))
    if best:
        thr, nf, prec = best
        comb = (flg > 0) | (ok & (oof <= thr))
        res[str(tgt)] = dict(thr_logit=thr, probe_only_fires=nf, probe_only_fire_rate=nf / len(yg), probe_only_precision=prec,
                             combined_fire_rate=float(comb.mean()), combined_precision=float(1 - yg[comb].mean()),
                             wrong_caught=float((comb & (yg == 0)).sum() / max(1, (yg == 0).sum())))
        print(tgt, res[str(tgt)])
san = flg > 0
print('sanity only: fire rate %.3f precision %.3f' % (san.mean(), 1 - yg[san].mean()))
torch.save(dict(w=torch.tensor(w, dtype=torch.float32), b=float(b), layer=18, C=CC, thresholds=res,
                note='fit on split C T=1 samples; thresholds from OOF logits of split-C greedy answers'), f'{O}/ver_probe.pt')
json.dump(res, open(f'{O}/ver_calib.json', 'w'), indent=1)
