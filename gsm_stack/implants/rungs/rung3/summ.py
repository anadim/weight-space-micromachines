"""summ.py REF_PREFIX PREFIX [...] : 4-seed summary vs reference rows (paired by set/seed/idx), rows in PREFIX_dev, PREFIX_dev2.
Per set: per-seed acc, mean (delta vs ref), tok/q, natt (tries/q), fixes/breaks; pooled fixes/breaks + one-sided sign test
p = P(X >= fixes | n = fixes+breaks, 1/2)."""
import gzip, json, glob, re, sys, os
from math import comb


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, '*.jsonl.gz'))):
        m = re.match(r'(dev2|dev)_.*_seed(\d+)', os.path.basename(f))
        if m: out[int(m.group(2))] = {r['idx']: r for r in map(json.loads, gzip.open(f, 'rt'))}
    return out


ref = sys.argv[1]
for P in sys.argv[2:]:
    FX = BR = 0; line = [os.path.basename(P)]
    for S in ('dev', 'dev2'):
        R, C = load(f'{ref}_{S}'), load(f'{P}_{S}')
        if not C: line.append(f'{S}: -'); continue
        acc, racc, tq, na, fx, br = [], [], [], [], 0, 0
        for sd in sorted(C):
            c, r = C[sd], R[sd]; ids = sorted(c); n = len(ids)
            acc.append(100 * sum(c[i]['ok'] for i in ids) / n); racc.append(100 * sum(r[i]['ok'] for i in ids) / n)
            tq.append(sum(c[i]['ntok'] for i in ids) / n); na.append(sum(c[i]['natt'] for i in ids) / n)
            fx += sum(c[i]['ok'] and not r[i]['ok'] for i in ids); br += sum(r[i]['ok'] and not c[i]['ok'] for i in ids)
        k = len(acc); m = sum(acc) / k; rm = sum(racc) / k
        line.append(f"{S} {'/'.join(f'{a:.2f}' for a in acc)} = {m:.2f} ({m - rm:+.2f}) tok/q {sum(tq)/k:.1f} natt {sum(na)/k:.2f} f/b {fx}/{br}")
        FX += fx; BR += br
    n = FX + BR; p = sum(comb(n, j) for j in range(FX, n + 1)) / 2 ** n if n else 1
    print(' | '.join(line) + f' | pooled f/b {FX}/{BR} p={p:.3f}')
