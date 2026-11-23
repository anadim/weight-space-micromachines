"""1.7B GSM-Plus testmini fresh set, rows 1-4 (j23919-21). Paired item bootstrap, 2000 resamples.
usage: python3 uniform/rl28/gsm8k/fe_gp/ana_gp17.py   (rows: uniform/items.py)"""
import glob, json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
import items   # uniform/items.py: reduced per-item rows (bundles) or your own full rows via $FE_ROWS

F = {n: n for n in items.names('gsmplus')}


def load(names):
    cols, toks = [], []
    for n in names:
        R, T = {}, []
        for r in items.load('gsmplus', F[n]):
            R[r['i']] = float(bool(r['ok'])); T.append(r['ntok'])
        cols.append(R); toks.append(sum(T) / len(T))
    ids = sorted(cols[0])
    assert all(sorted(c) == ids for c in cols), 'item sets differ'
    return [sum(c[i] for c in cols) / len(cols) for i in ids], [100 * sum(c.values()) / len(c) for c in cols], sum(toks) / len(toks), len(ids)


ROWS = {'1': ['gp_base17H'], '2': [f'gp_P17_s{s}' for s in range(3)], '3': ['gp_r2_17'], '4': [f'gp_r2_CW17_s{s}' for s in range(3)],
        '2-MV': [f'gp_scH19_P17_s{s}' for s in range(3)], '1-MV': ['gp_scH19_base17'], '1-MV3': ['gp_scH24_base17']}
V = {}
for k, names in ROWS.items():
    miss = [n for n in names if n not in F]
    if miss:
        print(f'row {k}: MISSING {miss}'); continue
    v, per, tq, n = load(names); V[k] = v
    print(f'row {k}  mean {100*sum(v)/len(v):.2f}  per-seed {" / ".join("%.2f" % p for p in per)}  tok/q {tq:.0f}  n {n}')
rng = random.Random(0)
for a, b in [('2', '1'), ('3', '1'), ('4', '1'), ('4', '2'), ('4', '3'), ('2', '3'), ('4', '2-MV'), ('2-MV', '2'), ('3', '1-MV3'), ('4', '1-MV')]:
    if a in V and b in V:
        d = [x - y for x, y in zip(V[a], V[b])]; n = len(d)
        bs = sorted(100 * sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(2000))
        print(f'  {a} - {b}: {100*sum(d)/n:+.2f} [{bs[50]:+.2f}, {bs[1949]:+.2f}]')
if all(k in V for k in '1234'):
    I = [(V['4'][i] - V['3'][i]) - (V['2'][i] - V['1'][i]) for i in range(len(V['1']))]; n = len(I)
    bs = sorted(100 * sum(I[rng.randrange(n)] for _ in range(n)) / n for _ in range(2000))
    print(f'  interaction (4-3)-(2-1): {100*sum(I)/n:+.2f} [{bs[50]:+.2f}, {bs[1949]:+.2f}]')
