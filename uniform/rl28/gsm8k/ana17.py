import json, random, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import items   # uniform/items.py: reduced per-item rows (bundles) or your own full rows via $FE_ROWS
# [28] uniform GSM8K rows on Qwen3-1.7B-Base: paired item bootstrap of the 2x2 effects (ana_uni.py method: seed-averaged per
# item for multi-seed rows, 2000 resamples, seed 0, 95% percentile CI). usage: python3 ana17.py [SEEDS e.g. 0,1]
S = (sys.argv[1] if len(sys.argv) > 1 else '0,1').split(',')
ROWS = {
    '1': ['base17H'],
    '2': ['P17_s%s' % s for s in S],
    '3': ['r2_17'],
    '4': ['r2_CW17_s%s' % s for s in S],
}
def load(st, names):
    cols, tok = [], []
    for n in names:
        R = {}; T = []
        for r in items.load(st, n):
            R[r['i']] = float(bool(r['ok'])); T.append(r['ntok'])
        cols.append(R); tok.append(sum(T) / len(T))
    ids = sorted(cols[0])
    assert all(sorted(c) == ids for c in cols)
    return ids, [sum(c[i] for c in cols) / len(cols) for i in ids], [100 * sum(c.values()) / len(c) for c in cols], tok
random.seed(0)
for st in ['test', 'hard']:
    V = {}; I = None
    for k, names in ROWS.items():
        ids, v, per, tok = load(st, names); V[k] = v
        assert I is None or I == ids; I = ids
        print(st, 'row', k, 'n', len(v), 'mean %.2f' % (100 * sum(v) / len(v)), 'per-seed', ' / '.join('%.2f' % p for p in per),
              'tok/q', ' / '.join('%.0f' % t for t in tok))
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
