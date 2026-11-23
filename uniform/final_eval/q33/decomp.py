"""[33] offline decomposition of Base 28.43 (serial N=4, no WPR) vs 57.39 (rung 2) on GSM8K test, from the eval rows.
usage: python3 uniform/final_eval/q33/decomp.py [SET_NAME ...]   (CPU; per-attempt answers = the att column of the item bundles)"""
import json, sys, os
from collections import Counter
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import items   # uniform/items.py: reduced per-item rows (bundles) or your own full rows via $FE_ROWS


def norm(a):
    try:
        f = float(a)
        return str(int(f)) if f == int(f) else repr(round(f, 6))
    except (TypeError, ValueError):
        return None


def load(n):
    st, nm = n.split('_', 1)
    return items.load(st, nm)


def answers(r):
    return [norm(a or None) for a in items.att_of(r)]


def first_k_maj(R, k):
    c = 0
    for r in R:
        g = norm(r['gold']); V = [a for a in answers(r)[:k] if a is not None]
        if V:
            top = Counter(V).most_common(); best = top[0][1]
            c += [a for a, m in top if m == best][0] == g
    return round(100 * c / len(R), 2)


for name in sys.argv[1:] or ['test_base_N4', 'test_r2_base', 'test_r3_base']:
    R = load(name); n = len(R)
    k = max(r['natt'] for r in R)
    pt = []
    for t in range(k):
        rows = [r for r in R if r['natt'] > t]
        pt.append((len(rows), round(100 * sum(answers(r)[t] == norm(r['gold']) for r in rows) / max(1, len(rows)), 1)))
    natt = Counter(r['natt'] for r in R)
    print(f"== {name}: acc {100*sum(r['ok'] for r in R)/n:.2f} tok/q {sum(r['ntok'] for r in R)/n:.0f} natt {dict(sorted(natt.items()))}")
    print('  per-attempt acc (rows reaching it, acc%):', pt[:16])
    print('  first-k majority (offline):', {kk: first_k_maj(R, kk) for kk in (1, 2, 4, 8, 16) if kk <= k})
    print('  no-answer % per attempt 1..4:',
          [round(100 * sum(answers(r)[t] is None for r in R if r['natt'] > t) / max(1, sum(r['natt'] > t for r in R)), 1)
           for t in range(min(4, k))])
