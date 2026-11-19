"""Q2/Q3 extras: unit-share per config, McNemar tests, digit bins, and candidate trace excerpts."""
import json, re, math
from collections import Counter
from analyze import T, A, N, kinds
from fmt_probe import feats
def mcnemar(k0, k1, kind=None):
    idx = [i for i in range(N) if kind is None or kinds[i] == kind]
    b = sum(1 for i in idx if not T[k0][i]['ok'] and T[k1][i]['ok']); c = sum(1 for i in idx if T[k0][i]['ok'] and not T[k1][i]['ok'])
    n = b + c; k = min(b, c)
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0
    return b, c, p
for pair in [('a1_off', 'a2_off'), ('a1_on', 'a2_on'), ('a1_off', 'a1_on'), ('a2_off', 'a2_on'), ('base_off', 'a1_off'), ('base_on', 'base_off')]:
    for kind in (None, 'gsm', 'scaled'):
        print('mcnemar', pair, kind or 'all', mcnemar(*pair, kind))
print()
for k in ('base_off', 'base_on', 'a1_off', 'a1_on', 'a2_off', 'a2_on'):
    eqs = [q for x in A[k] for q in x['eqs']]
    F = [feats(q) for q in eqs]
    print(f'{k:9s} eqs {len(eqs)}  units-in-expr {100*sum(f["units"] for f in F)/len(eqs):.1f}%  x-char {100*sum(f["xchar"] for f in F)/len(eqs):.1f}%  '
          f'hit320 {sum(x["hit_max"] for x in A[k])}  no#### {sum(1-r["fmt"] for r in T[k])}  '
          f'lines/ans {sum(x["nlines"] for x in A[k])/N:.2f}  verbal-ops/ans {sum(x["vwords"] for x in A[k])/N:.2f}  words/ans {sum(len(r["text"].split()) for r in T[k])/N:.0f}')
def dbin(q):
    m = max(len(str(int(abs(o)))) for o in q['operands']); return '1' if m == 1 else '2' if m == 2 else '3' if m == 3 else '4-5' if m <= 5 else '6+'
for k in ('base_on', 'a2_on'):
    c = Counter(dbin(cc['eq']) for x in A[k] for cc in x['calls'] if cc['eq'])
    print(k, 'calls by max operand digits', dict(sorted(c.items())))
    e = [q for x in A[k] for q in x['eqs']]
    for b in ['1', '2', '3', '4-5', '6+']:
        s = [q for q in e if dbin(q) == b]; print(f'   {b}: n={len(s)} called {100*sum(q["called"] for q in s)/max(1,len(s)):.0f}%', end='')
    print()
# per-kind calls
for k in ('base_on', 'a2_on'):
    for kind in ('gsm', 'scaled'):
        idx = [i for i in range(N) if kinds[i] == kind]
        print(k, kind, 'calls/ans %.2f' % (sum(len(A[k][i]['calls']) for i in idx) / 400), 'tok/ans %.0f' % (sum(A[k][i]['ntok'] for i in idx) / 400),
              'units %.1f%%' % (100 * sum(feats(q)['units'] for i in idx for q in A[k][i]['eqs']) / max(1, sum(len(A[k][i]['eqs']) for i in idx))))
