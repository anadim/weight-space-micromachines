"""Cause of each wrong->right flip, judged on the BEFORE trace: no_answer (no parsable '#### n': loop/truncation/'#### $'),
arith (an 'a op b = c' in it is wrong), plan (everything else: wrong setup/reading)."""
import json, re
from collections import Counter
from analyze import T, A, N, kinds
def cause(k, i):
    r = T[k][i]
    if not r['fmt']:
        if re.search(r'####\s*\$', r['text']): return 'no_answer:####$'
        if len(r['ids']) >= 320: return 'no_answer:truncated'
        return 'no_answer:other'
    if A[k][i]['nwrong']: return 'arith'
    return 'plan'
out = {}
for k0, k1 in [('base_off', 'a1_off'), ('base_on', 'a2_on'), ('a1_off', 'a2_on'), ('a1_on', 'a2_on'), ('a1_off', 'a2_off'), ('base_off', 'a2_off')]:
    for kind in ('gsm', 'scaled'):
        up = Counter(cause(k0, i) for i in range(N) if kinds[i] == kind and not T[k0][i]['ok'] and T[k1][i]['ok'])
        dn = Counter(cause(k1, i) for i in range(N) if kinds[i] == kind and T[k0][i]['ok'] and not T[k1][i]['ok'])
        out[f'{k0}->{k1}|{kind}'] = dict(up=dict(up), down=dict(dn))
        print(f'{k0}->{k1} {kind}: gains {sum(up.values())} {dict(up)} | losses {sum(dn.values())} {dict(dn)}')
for k in ('base_off', 'a1_off', 'a2_off'):
    print(k, 'no-answer causes', Counter(cause(k, i) for i in range(N) if not T[k][i]['fmt']))
json.dump(out, open('causes.json', 'w'), indent=1)
