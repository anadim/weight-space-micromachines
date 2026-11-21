"""gap streams (some slot right, vote wrong): does the check score rank the right group above the vote winner?
Also the K-mean upper bound of the in-sample sim (all 4 check samples averaged = 4 checks per candidate)."""
import ana, itertools, sys
from cands import correct
for S in ('dev', 'dev2'): ana.DATA[S] = ana.load(S)
KEYS = sys.argv[1].split(',') if len(sys.argv) > 1 else ['dir_mean', 'v2', 'fwd_v', 'fwd_agree', 'fwd_va', 'bwd_v', 'bwd_agree', 'bwd_va']
KM = sys.argv[2].split(',') if len(sys.argv) > 2 else ['fwd_agree', 'fwd_va', 'fwd_v']
for S in ('dev', 'dev2'):
    _, _, st = ana.DATA[S]; seeds = sorted({s['seed'] for s in st}); si = {s: j for j, s in enumerate(seeds)}
    CH = {k: ana.check_map(S, k) for k in KEYS}
    res = {k: [0, 0, 0] for k in KEYS + ['lgmax', 'W']}; ng = 0
    for s in st:
        g = ana.groups(s['slots'])
        if not g: continue
        it = list(g.items()); best = max(it, key=lambda kv: round(kv[1][1], 6))
        if correct(best[0], s['gold']): continue
        rr = [kv for kv in it if correct(kv[0], s['gold'])]
        if not rr: continue
        ng += 1; r = rr[0]
        def cmp(k, a, b):
            if a is None or b is None: return
            res[k][0 if a > b else 1 if a < b else 2] += 1
        cmp('lgmax', r[1][2], best[1][2]); cmp('W', r[1][1], best[1][1])
        for k in KEYS:
            a, b = CH[k].get((s['idx'], r[0])), CH[k].get((s['idx'], best[0]))
            if a is None or b is None: res[k][2] += 0; continue
            if isinstance(a, list): a, b = a[si[s['seed']] % len(a)], b[si[s['seed']] % len(b)]
            cmp(k, a, b)
    print(f'{S}: {ng} gap streams (of {len(st)}); right-group check score vs winner: win/lose/tie')
    for k, v in res.items(): n = sum(v); print(f'   {k:10s} {v[0]:4d}/{v[1]:4d}/{v[2]:4d}  P(right>winner)={(v[0] + .5 * v[2]) / max(n, 1):.3f}')
# K-mean in-sample upper bound
for key in KM:
    line = f'{key}@Kmean'
    for S in ('dev', 'dev2'):
        CH = {k: (sum(v) / len(v) if isinstance(v, list) else v) for k, v in ana.check_map(S, key).items()}
        vals = list(CH.values()); mu = sum(vals) / len(vals); sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** .5 + 1e-9
        best = None
        for beta, tau, K in itertools.product([0, .02, .05, .1, .2, .3, .5, 1, 2, 5], [.5, .6, .7, .8, .9, 1.01], [2, 3]):
            r = ana.sim(S, key, beta, tau, K, mu, sd, CH); n = sum(t[1] for t in r) / len(r); b = sum(t[0] for t in r) / len(r)
            if best is None or n > best[0] + 1e-12: best = (n, b, beta, tau, K, sum(1 for t in r if t[1] > t[0]), sum(1 for t in r if t[1] < t[0]))
        n, b, beta, tau, K, fx, br = best
        line += f' | {S} {100*b:.2f}->{100*n:.2f} ({100*(n-b):+.2f} +{fx}/-{br}) b={beta} t={tau} K={K}'
    print(line)
