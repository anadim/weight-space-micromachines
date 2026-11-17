"""Summarize the v2 final runs (results/rethink_v2/final/*.json, shards merged): accuracy, extra tokens vs the plain arm
of the same mode (per problem, n_tok(arm) - n_tok(none)), firing (phrase before the first '####' = B fire, after = A
fire), paired bootstrap 95% CIs (10,000 resamples of problems) for gated - always and gated - none.
python scripts/rethink_v2/summarize_v2.py [PREFIX=results/rethink_v2/final] [GATED=hybrid] > results/rethink_v2/summary.json
(the shipped runs are in results/rethink_v2/final_eval, reduced by scripts/reduce_outputs.py, which precomputes the
fire flags fire_b / fire_a with the same rule as fires() below)"""
import json, glob, sys, collections, numpy as np
PRE = sys.argv[1] if len(sys.argv) > 1 else 'results/rethink_v2/final'; GN = sys.argv[2] if len(sys.argv) > 2 else 'hybrid'
runs = collections.defaultdict(dict)
for f in sorted(glob.glob(f'{PRE}/*.json')):
    name = f.split('/')[-1][:-5]
    if name.startswith('tv_'): key = name[3:].rsplit('_', 1); key = (key[0], 'plain', key[1])
    else:
        p = name.split('_'); key = (p[0], p[1], p[2])
    for r in json.load(open(f))['rows']: runs[key][r['i']] = r
PH = 'Hmm, wait. Let me rethink this.'
def fires(o):
    k = o.find(PH)
    if k < 0: return 0, 0
    h = o.find('####')
    return (1, 0) if (h < 0 or k < h) else (0, 1)
rng = np.random.default_rng(0)
def boot(d):
    d = np.asarray(d, float); n = len(d); idx = rng.integers(0, n, (10000, n)); m = d[idx].mean(1)
    return [float(d.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))]
out = {}
for split, mode in sorted({(k[0], k[1]) for k in runs}):
    arms = {k[2]: runs[k] for k in runs if k[0] == split and k[1] == mode}
    if 'none' not in arms: continue
    common = set.intersection(*[set(v) for v in arms.values()])
    ids = sorted(common); base = arms['none']
    res = {'n_common': len(ids), 'n_per_arm': {a: len(v) for a, v in arms.items()}}
    for a, R in arms.items():
        ok = np.array([R[i]['ok'] for i in ids]); ntok = np.array([R[i]['n_tok'] for i in ids])
        extra = ntok - np.array([base[i]['n_tok'] for i in ids])
        fb = np.array([(R[i]['fire_b'], R[i]['fire_a']) if 'fire_b' in R[i] else fires(R[i]['out']) for i in ids])
        res[a] = dict(acc=100 * ok.mean(), mean_gen_tokens=ntok.mean(), extra_tokens=extra.mean(),
                      fire_before_hash=fb[:, 0].mean(), fire_after_answer=fb[:, 1].mean(),
                      w2r_vs_none=int(sum(R[i]['ok'] and not base[i]['ok'] for i in ids)), r2w_vs_none=int(sum(base[i]['ok'] and not R[i]['ok'] for i in ids)))
    if GN in arms:
        G = arms[GN]
        for ref in ['alwaysA', 'none']:
            if ref in arms:
                Rf = arms[ref]
                res[f'{GN}_minus_{ref}_acc_pp'] = boot([100 * (G[i]['ok'] - Rf[i]['ok']) for i in ids])
                res[f'{ref}_minus_{GN}_tokens'] = boot([Rf[i]['n_tok'] - G[i]['n_tok'] for i in ids])
    out[f'{split}_{mode}'] = res
print(json.dumps(out, indent=1, default=float))
