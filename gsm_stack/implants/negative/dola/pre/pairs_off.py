"""DoLa prescreen, offline part (CPU): pairwise logit differences at branch points / trie forks.
python3.11 implants/dola/pre/pairs_off.py
Uses probe/branch data (teacher-forced h at fork positions, layers 14/18/22/27 = layer-input residuals, 28 = final-norm input)
and the unembedding rows of all candidate tokens. For a pair (a, b) at one position:
  D_L = sum(rmsnorm(h_L) * g * (u_a - u_b))  (= z_L[a] - z_L[b], softmax normalisation cancels)
  DoLa: D' = D_28 + alpha * (D_28 - D_l)
Metrics: pairs (atlas branch pairs, a = right token, b = wrong token): acc = P(D' > 0), accW (per wrong attempt), flips toward
right (D<0, D'>0) and toward wrong (D>0, D'<0), mean(D' - D); soft fork pairs (all trie forks, label = child with higher
n_right/n, weight |dv|): weighted acc.  No APC here (needs full vocab: GPU job).
"""
import os, sys, json
import numpy as np
WS = os.path.expanduser(os.environ.get('WSM_ROOT', '.') + '/gsm_stack/implants'); sys.path.insert(0, f'{WS}/probe/branch')
import fit
LAYS = fit.LAYS
ALPHAS = [0.1, 0.25, 0.5, 1.0, 2.0]
out = {}
for S in ['trainC', 'dev', 'dev2']:
    d = fit.load_set(S)
    res = {}
    for kind, P in [('pairs', d['pairs']), ('forks', d['fpairs'])]:
        F = fit.feats(d, P)
        D = {L: (F['hn'][:, i] * F['du']).sum(-1).numpy() for i, L in enumerate(LAYS)}
        y = F['y'].numpy(); w = F['w'].numpy()
        wn = np.array([p.get('n_wrong', 1) for p in P], np.float32)
        D0 = D[28]
        sgn = np.where(y > 0.5, 1.0, -1.0)  # orient: positive = toward the right/better token
        def met(Dp):
            s = sgn * Dp; s0 = sgn * D0
            r = dict(acc=float(np.average(s > 0, weights=w)), dmean=float(np.average(s - s0, weights=w)),
                     to_right=int(((s0 < 0) & (s > 0)).sum()), to_wrong=int(((s0 > 0) & (s < 0)).sum()))
            if kind == 'pairs': r['accW'] = float(np.average(s > 0, weights=wn))
            return r
        res[kind] = {'n': len(P), 'host': met(D0), 'corr28_dlp': float(np.corrcoef(D0, F['dlp'].numpy())[0, 1])}
        for l in [14, 18, 22, 27]:
            for a in ALPHAS:
                res[kind][f'l{l}_a{a}'] = met(D0 + a * (D0 - D[l]))
            res[kind][f'l{l}_only'] = met(D[l])
    out[S] = res
    print(S, json.dumps({k: (v['n'], v['host'], v['corr28_dlp']) for k, v in res.items()}), flush=True)
    for kind in ['pairs', 'forks']:
        print(f'  {kind}: cfg acc dmean to_right/to_wrong' + (' accW' if kind == 'pairs' else ''))
        for k, v in res[kind].items():
            if k in ('n', 'corr28_dlp'): continue
            print(f'   {k:12s} {v["acc"]:.3f} {v["dmean"]:+.3f} {v["to_right"]}/{v["to_wrong"]}' + (f' {v["accW"]:.3f}' if 'accW' in v else ''))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pairs_off.json'), 'w'), indent=1)
