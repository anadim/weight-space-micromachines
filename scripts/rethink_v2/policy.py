"""Stage 2 (TRAIN side only): detector candidates at the read point 'pre' (before '####'), then gating policies.
Uses fit_detector.py's data loading/splits (fit 70 / sel 15 / rep 15 by source problem).  Outcomes per solution come
from gen_train.py:  A = rethink after the committed answer (v1 point);  B = rethink at '####' (before committing),
B150/B300 = B with a budget cap.  Gated policy: fire iff detector P(correct) logit < theta; theta chosen on SEL greedy
(pooled GSM8K-train + scaled-train) for max accuracy; numbers reported on REP greedy.  Saves the chosen detector.
python scripts/rethink_v2/policy.py OUTDIR CONFIGS_JSON"""
import os, sys, json, torch, numpy as np
sys.argv = [sys.argv[0], sys.argv[1]] + sys.argv[2:]
CFG = json.loads(sys.argv[2]) if len(sys.argv) > 2 else None
src_code = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fit_detector.py')).read()
exec(src_code.split('# 1) logistic per read point x layer')[0])          # data, splits, helpers
greedy = kind == 'greedy'
OUTC = {'A': ('rta_ok', 'extra_rta'), 'B': ('rt_ok', 'extra_rt'), 'B150': ('cap150_ok', 'extra_cap150'), 'B300': ('cap300_ok', 'extra_cap300')}
oc = {k: (cat(v[0]).astype(bool), cat(v[1]).astype(float)) for k, v in OUTC.items()}
res = {'fix_break': {}}
for k, (o, ex) in oc.items():
    for nm in D:
        m = greedy & (src == nm) & (fit | sel)
        res['fix_break'][f'{k}_{nm}'] = dict(base=yn[m].mean(), always=o[m].mean(), fix=o[m & ~yn.astype(bool)].mean(),
                                            brk=1 - o[m & yn.astype(bool)].mean(), extra=ex[m].mean())
print(json.dumps(res['fix_break'], indent=0, default=float))
cands = CFG or [['lin', 'pre', [18]], ['lin', 'pre', [14, 18, 22]], ['mlp', 'pre', [14, 18, 22, 27], 128]]
best = None; res['cands'] = []
for cfg in cands:
    X = feats(cfg[1], cfg[2])
    if cfg[0] == 'lin':
        r = max((dict(train_lin(X, y_all, fit, sel, wd, epochs=40), wd=wd) for wd in [1e-2, 1e-1, 1.0]), key=lambda r: r['auc_sel'])
    else:
        r = max((dict(train_lin(X, y_all, fit, sel, wd, hidden=cfg[3], epochs=60, drop=dr, lr=3e-4), wd=wd, drop=dr)
                 for wd in [1e-2, 1e-1] for dr in [0.0, 0.3]), key=lambda r: r['auc_sel'])
    a = aucs(r['score']); a['cfg'] = cfg; a['wd'] = r['wd']; res['cands'].append(a)
    print(cfg, {k: round(v, 3) for k, v in a.items() if isinstance(v, float)}, flush=True)
    key = sum(a[f'sel_{nm}_greedy'] for nm in D)
    if best is None or key > best[0]: best = (key, cfg, r)
_, cfg, r = best; s = r['score']               # logit P(correct)
res['chosen'] = cfg
def sim(policy, theta, m):
    o, ex = oc[policy]; f = s < theta
    return np.where(f, o, yn.astype(bool))[m].mean(), (ex * f)[m].mean(), f[m].mean()
thetas = np.quantile(s[greedy], np.linspace(0, 1, 201))
res['policies'] = {}
for pol in OUTC:
    msel = greedy & sel
    accs = [sim(pol, t, msel)[0] for t in thetas]
    # ties -> fewer firings (smaller theta)
    bi = int(np.argmax(np.array(accs) - 1e-9 * np.arange(len(accs))))
    th = float(thetas[bi])
    row = {'theta': th}
    for part, m0 in [('sel', sel), ('rep', rep)]:
        for nm in D:
            m = m0 & greedy & (src == nm)
            ga, ge, gf = sim(pol, th, m); o, ex = oc[pol]
            row[f'{part}_{nm}'] = dict(base=yn[m].mean() * 100, always=o[m].mean() * 100, always_extra=ex[m].mean(),
                                      gated=ga * 100, gated_extra=ge, fire_rate=gf, oracle=np.where(~yn.astype(bool), o, True)[m].mean() * 100)
    res['policies'][pol] = row
    print(pol, json.dumps({k: ({kk: round(vv, 2) for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in row.items()}, default=float), flush=True)
res['curve'] = {pol: [[float(t)] + [float(x) for x in sim(pol, t, greedy & sel)] for t in thetas[::10]] for pol in OUTC}
json.dump(res, open(f'{OUT}/policy.json', 'w'), indent=1, default=float)
st = r['state']
if r['hidden']:
    det = dict(W1=st['1.weight'], b1=st['1.bias'], W2=-st['3.weight'].reshape(-1), b2=-float(st['3.bias']))
else:
    det = dict(W1=None, b1=None, W2=-st['weight'].reshape(-1), b2=-float(st['bias']))
nl = 28
det.update(layers=cfg[2], mu=r['mu'], sd=r['sd'], read=cfg[1], cfg=cfg, thetas={p: res['policies'][p]['theta'] for p in OUTC},
           fit='GSM8K TRAIN + scaled-number TRAIN analogue generations (fit split); theta on sel split; see results/rethink_v2/policy.json')
torch.save(det, f'{OUT}/detector_base.pt')        # thr_logit set per policy: fire iff -logit >= -theta
print('saved detector', cfg)
