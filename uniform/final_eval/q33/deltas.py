"""[33] paired item-bootstrap deltas for the Base rung-2 ablations (GSM8K test, seed 7). 2000 resamples, 95% CI.
usage: python3 uniform/final_eval/q33/deltas.py   (CPU; output in deltas.txt)"""
import json, os, random
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import items   # uniform/items.py: reduced per-item rows (bundles) or your own full rows via $FE_ROWS


def ok(n):
    R = items.load('test', n)
    return {r['i']: r['ok'] for r in R}, sum(r['ntok'] for r in R) / len(R)


ROWS = ['baseH', 'baseB', 'base_N4', 'q33_N1', 'q33_fixN4', 'q33_fixN8', 'r2_base', 'q33_vote0', 'q33_nowpr']
D = {n: ok(n) for n in ROWS}
print('| config | acc | tok/q |')
for n in ROWS:
    o, t = D[n]; print(f'| {n} | {100*sum(o.values())/len(o):.2f} | {t:.0f} |')
PAIRS = [('baseB', 'base_N4'), ('q33_N1', 'baseH'), ('q33_fixN4', 'base_N4'), ('q33_fixN4', 'q33_N1'), ('q33_fixN8', 'q33_fixN4'),
         ('r2_base', 'q33_fixN8'), ('q33_vote0', 'r2_base'), ('r2_base', 'q33_nowpr'), ('r2_base', 'baseH')]
rng = random.Random(0)
for a, b in PAIRS:
    oa, ob = D[a][0], D[b][0]; ks = sorted(set(oa) & set(ob)); d = [oa[k] - ob[k] for k in ks]
    bs = sorted(100 * sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(2000))
    print(f'{a} − {b}: {100*sum(d)/len(d):+.2f} [{bs[49]:+.2f}, {bs[1949]:+.2f}]  n={len(ks)}')
