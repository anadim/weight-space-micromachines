"""ceiling check: within-item AUROC of the TRUE step labels aggregated per attempt (uses gold annotations -> NOT deployable;
diagnostic of how much within-item signal the step-label supervision itself carries), and of the probe on the same aggregations."""
import sys, os, numpy as np
sys.argv = [sys.argv[0]]
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
exec(open(f'{H}/off/ana.py').read().split("keys = sorted")[0])
for S in SETS:
    att = SC[f'att|{S}']; lab = SC[f'lab|{S}'].astype(float); e = lab >= 0; n = A[S]['n']
    c = np.bincount(att[e], minlength=n); g = np.bincount(att[e], lab[e], minlength=n)
    frac = np.where(c > 0, g / np.maximum(c, 1), 0.5); bad = c - g
    mn = np.ones(n); np.minimum.at(mn, att[e], lab[e])
    r = dict(frac_good=wauc(S, frac), min_good=wauc(S, mn), n_bad_neg=wauc(S, -bad))
    r['b+frac'] = max(wauc(S, A[S]['b'] + v * frac) for v in (0.5, 1, 2, 4, 8))
    r['corr_frac_b'] = wcorr(S, frac, A[S]['b'])
    print(S, {k: round(v, 4) for k, v in r.items()})
