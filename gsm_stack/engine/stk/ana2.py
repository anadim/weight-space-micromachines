"""Summary of the forward-only single-call stack. usage: python ana2.py OUTDIR [OUTDIR2 ...]  -> prints JSON per set/config."""
import sys, os, json, glob, numpy as np
res = {}
for O in sys.argv[1:]:
    for fn in sorted(glob.glob(f'{O}/*_S*.jsonl')):
        nm, cfg = os.path.basename(fn)[:-6].split('_', 1)
        if cfg not in ('S0', 'S1') and not cfg.startswith('S2_'): continue
        tag = os.path.basename(O.rstrip('/'))
        R = [json.loads(l) for l in open(fn)]
        b0 = f'{O}/{nm}_S0.jsonl'
        if not os.path.exists(b0): b0 = f"{O.rstrip('/').rstrip('z')}/{nm}_S0.jsonl"
        B = [json.loads(l) for l in open(b0)]
        ok = np.array([x['ok'] for x in R]); bo = np.array([x['ok'] for x in B]); d = ok - bo
        rng = np.random.RandomState(0); bs = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(2000)]
        res[f'{tag}/{nm}/{cfg}'] = dict(n=len(R), acc=round(100 * ok.mean(), 2), tok_q=round(float(np.mean([x['ntok'] for x in R])), 1),
            fixes=int(((ok == 1) & (bo == 0)).sum()), breaks=int(((ok == 0) & (bo == 1)).sum()), delta=round(100 * d.mean(), 2),
            ci=[round(100 * np.percentile(bs, 2.5), 2), round(100 * np.percentile(bs, 97.5), 2)],
            c2_fire=round(100 * np.mean([x['c2fires'] > 0 for x in R]), 2), san_fire=round(100 * np.mean([x['san_fires'] > 0 for x in R]), 2),
            c2_events=int(sum(x['c2fires'] for x in R)))
for k, v in res.items(): print(k, json.dumps(v))
