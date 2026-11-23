import json, random, sys
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import items   # uniform/items.py: reduced per-item rows (bundles) or your own full rows via $FE_ROWS
# [26] uniform GSM8K rows: paired item bootstrap of the 2x2 effects (seed-averaged per item for multi-seed rows)
# usage: python3 uniform/final_eval/ana_uni.py   (CPU)
ROWS = {
    '1': ['baseH'],
    '2': ['P64_s0', 'P64_s1', 'P64_s2'],
    '3': ['r2_base'],
    '4': ['r2_CW_s0', 'r2_CW_s1', 'r2_CW_s2'],
}
def load(st, names):
    cols = []
    for n in names:
        R = {}
        for r in items.load(st, n):
            R[r['i']] = float(bool(r['ok']))
        cols.append(R)
    ids = sorted(cols[0])
    return ids, [sum(c[i] for c in cols) / len(cols) for i in ids], [100 * sum(c.values()) / len(c) for c in cols]
random.seed(0)
for st in ['test', 'hard']:
    V = {}
    for k, names in ROWS.items():
        ids, v, per = load(st, names); V[k] = v
        print(st, 'row', k, 'mean %.2f' % (100 * sum(v) / len(v)), 'per-seed', ' / '.join('%.2f' % p for p in per))
    n = len(V['1'])
    eff = {
        'RL alone (2-1)': lambda i: V['2'][i] - V['1'][i],
        'implants alone (3-1)': lambda i: V['3'][i] - V['1'][i],
        'both (4-1)': lambda i: V['4'][i] - V['1'][i],
        'RL given implants (4-3)': lambda i: V['4'][i] - V['3'][i],
        'implants given RL (4-2)': lambda i: V['4'][i] - V['2'][i],
        'interaction (4-3-2+1)': lambda i: V['4'][i] - V['3'][i] - V['2'][i] + V['1'][i],
    }
    for name, f in eff.items():
        d = [f(i) for i in range(n)]
        pt = 100 * sum(d) / n
        bs = sorted(100 * sum(d[random.randrange(n)] for _ in range(n)) / n for _ in range(2000))
        print('  %-26s %+.2f [%+.2f, %+.2f]' % (name, pt, bs[50], bs[1949]))
