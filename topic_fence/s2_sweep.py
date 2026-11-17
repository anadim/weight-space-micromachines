"""Probe selection on DEV ONLY. For fence F: rows -> pos / neg / hard (drift excluded). Leave-one-style-out (LOSO):
train on all other styles, score the held-out style. Also leave-one-source-out (gen7b <-> public). Selection = mean over held-out
styles of AUROC(pos vs neg+hard), with AUROC(pos vs hard) reported. Threshold from pooled LOSO out-of-fold scores.
usage: s2_sweep.py F states.pt dev.jsonl out.json"""
import sys, json, torch, numpy as np
from sklearn.metrics import roc_auc_score
F, sp, dp, outp = sys.argv[1:5]
X = torch.load(sp); rows = [json.loads(l) for l in open(dp)]
other = 'ml' if F == 'math' else 'math'

def role(r):
    c = r['clf_' + F]
    if r['label'] == 'drift': return None
    if r['fence'] == F and r['label'] == 'pos': return 'pos' if c > 0 else None
    if r['source'] == 'gsm8k_train': return 'pos' if F == 'math' else ('neg' if c < 0 else None)
    if r['source'] == 'alpaca_train': return 'pos' if c > 2 else ('neg' if c < -2 else None)
    if r['fence'] == F and r['label'] == 'hard': return 'hard' if c < 0 else None
    return 'neg' if c < 0 else None
R = [role(r) for r in rows]
keep = [i for i, x in enumerate(R) if x is not None]
y = np.array([R[i] == 'pos' for i in keep], dtype=np.float32); kind = np.array([R[i] for i in keep])
sty = np.array([rows[i]['style'] for i in keep]); src = np.array([rows[i]['source'] for i in keep])
X = X[keep].float()
print(F, 'pos', (kind == 'pos').sum(), 'neg', (kind == 'neg').sum(), 'hard', (kind == 'hard').sum(), flush=True)
STY = sorted(set(sty))

def fit(Xtr, ytr, lam, iters=150):
    Xtr = Xtr.cuda(); ytr = torch.tensor(ytr).cuda()
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-3
    Z = (Xtr - mu) / sd
    wpos = 0.5 / ytr.mean(); wneg = 0.5 / (1 - ytr.mean()); sw = torch.where(ytr > 0, wpos, wneg)
    w = torch.zeros(Z.shape[1], device='cuda', requires_grad=True); b = torch.zeros(1, device='cuda', requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=iters, line_search_fn='strong_wolfe')
    def clo():
        opt.zero_grad()
        l = (torch.nn.functional.binary_cross_entropy_with_logits(Z @ w + b, ytr, reduction='none') * sw).mean() + lam * (w ** 2).sum()
        l.backward(); return l
    opt.step(clo)
    w = w.detach() / sd; b = b.detach() - (mu * w).sum()
    return w.cpu(), float(b)

def score(Xte, w, b): return (Xte @ w + b).numpy()

def auc(yy, s):
    return float('nan') if len(set(yy)) < 2 else roc_auc_score(yy, s)

res = {}
nl = X.shape[1]
import os
LS = [int(x) for x in os.environ['LAYERS'].split(',')] if 'LAYERS' in os.environ else range(2, nl, 2)
LAMS = [float(os.environ['LAM'])] if 'LAM' in os.environ else [1e-4, 1e-3, 1e-2]
for L in LS:
    for lam in LAMS:
        oof = np.zeros(len(y)); per = {}
        for s in STY:
            te = sty == s; w, b = fit(X[~te, L], y[~te], lam); oof[te] = score(X[te, L], w, b)
            msk_h = te & ((kind == 'pos') | (kind == 'hard'))
            per[s] = (auc(y[te], oof[te]), auc(y[msk_h], oof[msk_h]))
        lso = {}
        for a in ['gen7b']:
            tr = src == a; w, b = fit(X[tr, L], y[tr], lam); s_ = score(X[~tr, L], w, b)
            lso['gen7b->public'] = auc(y[~tr], s_)
            tr = src != 'gen7b'
            if y[tr].sum() > 10:
                w, b = fit(X[tr, L], y[tr], lam); s_ = score(X[~tr, L], w, b)
                lso['public->gen7b'] = auc(y[~tr], s_); hm = (~tr) & ((kind == 'pos') | (kind == 'hard')); lso['public->gen7b_hard'] = auc(y[hm], score(X[hm, L], w, b))
        mean_all = float(np.nanmean([v[0] for v in per.values()])); mean_h = float(np.nanmean([v[1] for v in per.values()]))
        worst = float(np.nanmin([v[0] for v in per.values()]))
        key = f'L{L}_lam{lam}'
        res[key] = dict(L=L, lam=lam, loso_auc=mean_all, loso_auc_hard=mean_h, loso_worst=worst, per_style=per, lso=lso,
                        oof_auc=auc(y, oof))
        print(key, f'LOSO {mean_all:.4f} hard {mean_h:.4f} worst {worst:.4f}', {k: round(v, 4) for k, v in lso.items()}, flush=True)
        res[key]['_oof'] = oof.tolist()
best = max(res, key=lambda k: res[k]['loso_auc'] + res[k]['loso_auc_hard'])
B = res[best]; oof = np.array(B['_oof'])
# threshold on LOSO OOF logits: max TPR s.t. FPR(hard) <= 5% and FPR(neg) <= 1%
cands = np.sort(oof)
def rates(t): return (oof[kind == 'pos'] >= t).mean(), (oof[kind == 'neg'] >= t).mean(), (oof[kind == 'hard'] >= t).mean()
ok = [t for t in cands if rates(t)[1] <= 0.01 and rates(t)[2] <= 0.05]
thr_logit = float(min(ok)) if ok else float(cands[-1])
tpr, fn, fh = rates(thr_logit)
w, b = fit(X[:, B['L']], y, B['lam'])
per_style_tpr = {s: float((oof[(sty == s) & (kind == 'pos')] >= thr_logit).mean()) for s in STY if ((sty == s) & (kind == 'pos')).any()}
per_style_fpr = {s: float((oof[(sty == s) & (kind != 'pos')] >= thr_logit).mean()) for s in STY if ((sty == s) & (kind != 'pos')).any()}
probe = dict(fence=F, layer=B['L'], lam=B['lam'], w=w.tolist(), b=b, thr=float(1 / (1 + np.exp(-thr_logit))), thr_logit=thr_logit,
             oof_tpr=float(tpr), oof_fpr_neg=float(fn), oof_fpr_hard=float(fh), per_style_tpr=per_style_tpr, per_style_fpr=per_style_fpr)
print('BEST', best, json.dumps({k: v for k, v in probe.items() if k not in ('w',)}, indent=1))
json.dump(dict(probe=probe, sweep={k: {kk: vv for kk, vv in v.items() if kk != '_oof'} for k, v in res.items()}), open(outp, 'w'))
