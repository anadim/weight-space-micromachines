"""same-stream analysis of the vfeat GPU run: vf pick (written '#### X', row ok) vs pick0 (base sigmoid(lg) voter on the
same registers / tokens), and vs the base rollouts gen/out (different stream on dev waves > 0: global sampler counter)."""
import os, gzip, json
import numpy as np
H = os.path.expanduser(os.environ.get('WSM_ROOT', '.') + '/gsm_stack/implants'); HERE = os.path.dirname(os.path.abspath(__file__))


def vok(v, g):
    g = float(str(g).replace(',', '')); return abs(v - g) <= 1e-6 * max(1.0, abs(g))


tot = {}
for S in ('dev', 'dev2'):
    for sd in (7, 11, 13, 17):
        f = f'{HERE}/job/out/{S}_N8_R1_v1_s1_m0_T1.0_seed{sd}_vf.jsonl.gz'
        if not os.path.exists(f): continue
        R = [json.loads(l) for l in gzip.open(f)]
        B0 = [json.loads(l) for l in gzip.open(f'{H}/gen/out/{S}_N8_R1_v1_s1_m0_T1.0_seed{sd}.jsonl.gz')]
        a = np.array([r['ok'] for r in R], float)
        p0 = np.array([float(r['pick0'] >= 0 and r['rok'][r['pick0']] and vok(r['val'][r['pick0']], r['gold'])) for r in R])
        # rows where no final vote happened (natt == 1 etc.): pick0 = -1 -> use the row's own ok
        nv = np.array([r['pick0'] < 0 for r in R]); p0[nv] = a[nv]
        b = np.array([x['ok'] for x in B0], float)
        sm = np.array([r['natt'] == x['natt'] and r['tokens'][:r['end'][r['natt'] - 1] + 1] == x['tokens'][:x['end'][x['natt'] - 1] + 1] for r, x in zip(R, B0)])
        tk = np.mean([r['ntok'] for r in R]); tk0 = np.mean([x['ntok'] for x in B0])
        s = (f'{S:5s} s{sd}: n={len(R)} vf {100*a.mean():.2f} | same-stream pick0 {100*p0.mean():.2f} -> d {100*(a-p0).mean():+.2f} '
             f'(fix {int(((a==1)&(p0==0)).sum())} brk {int(((a==0)&(p0==1)).sum())}, picks differ {sum(r["pick"]!=r["pick0"] for r in R)}) | '
             f'base gen/out {100*b.mean():.2f} (token-identical rows {int(sm.sum())}; on those vf-base {100*(a-b)[sm].mean():+.2f}) | tok/q {tk:.1f} vs {tk0:.1f}')
        print(s); tot.setdefault(S, []).append((a.mean(), p0.mean(), b.mean()))
for S, L in tot.items():
    L = np.array(L); print(f'{S} mean over {len(L)} seeds: vf {100*L[:,0].mean():.2f}  pick0 {100*L[:,1].mean():.2f}  d {100*(L[:,0]-L[:,1]).mean():+.2f}  base gen/out {100*L[:,2].mean():.2f}')
