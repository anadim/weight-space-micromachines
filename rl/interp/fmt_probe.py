"""Which equation surface forms does the calculator fire on, and does RL shift the host toward them?"""
import json, re, sys
from collections import Counter
from eqparse import equations
from analyze import T, A, N, kinds
def feats(q):
    l = q['lhs']; tail = l
    # take the expression part: from first operand on
    nums = list(re.finditer(r'\d[\d,]*(?:\.\d+)?', l))
    expr = l[nums[-len(q['operands'])].start():] if len(nums) >= len(q['operands']) else l
    return dict(units=bool(re.search(r'[A-Za-z]', expr)), dollar='$' in expr, comma=bool(re.search(r'\d,\d', expr)),
                dec='.' in expr, multi=len(q['ops']) > 1, paren='(' in expr, xchar=bool(re.search(r'\sx\s', expr)),
                rhsdec='.' in q['rhs'], big=max(len(str(int(abs(o)))) for o in q['operands']) >= 6)
for k in ('base_on', 'a2_on', 'a1_on'):
    eqs = [q for x in A[k] for q in x['eqs']]
    F = [feats(q) for q in eqs]
    print('==', k, 'n eq', len(eqs), 'called', sum(q['called'] for q in eqs))
    for f in F[0]:
        has = [q for q, ff in zip(eqs, F) if ff[f]]
        print(f'  {f:7s} share {100*len(has)/len(eqs):5.1f}%  called-rate {100*sum(q["called"] for q in has)/max(1,len(has)):5.1f}%  '
              f'(without: {100*sum(q["called"] for q,ff in zip(eqs,F) if not ff[f])/max(1,sum(1 for ff in F if not ff[f])):5.1f}%)')
# samples of uncalled single-op equations in base_on
import random; random.seed(0)
unc = [(i, q) for i, x in enumerate(A['base_on']) for q in x['eqs'] if not q['called'] and len(q['ops']) == 1]
print('uncalled single-op base_on:', len(unc))
for i, q in random.sample(unc, 12): print('   ', repr(q['lhs'][-50:] + ' = ' + q['rhs']), q['ok'])
