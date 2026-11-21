"""in-sample UPPER BOUND preview (dev/dev2 only, before trainC exists): best (beta,tau,K) chosen on the set itself."""
import itertools, sys, ana
for S in ('dev', 'dev2'): ana.DATA[S] = ana.load(S)
keys = sys.argv[1].split(',')
for key in keys:
    line = f'{key:14s}'
    for S in ('dev', 'dev2'):
        CH = ana.check_map(S, key)
        vals = [v for c in CH.values() for v in (c if isinstance(c, list) else [c])]
        mu = sum(vals) / len(vals); sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** .5 + 1e-9
        best = None
        for beta, tau, K in itertools.product([0, .02, .05, .1, .2, .3, .5, 1, 2, 5], [.5, .6, .7, .8, .9, 1.01], [2, 3]):
            r = ana.sim(S, key, beta, tau, K, mu, sd, CH)
            n = sum(t[1] for t in r) / len(r); b = sum(t[0] for t in r) / len(r)
            if best is None or n > best[0] + 1e-12:
                best = (n, b, beta, tau, K, sum(1 for t in r if t[1] > t[0]), sum(1 for t in r if t[1] < t[0]))
        n, b, beta, tau, K, fx, br = best
        line += f' | {S} {100*b:.2f}->{100*n:.2f} ({100*(n-b):+.2f} +{fx}/-{br}) b={beta} t={tau} K={K}'
    print(line, flush=True)
