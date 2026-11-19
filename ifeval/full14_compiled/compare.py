"""Byte-identity of two generation runs, and the silence check.

python compare.py A.jsonl[,A2,...] B.jsonl[,...] [--plain P.jsonl,...]
  identical responses A vs B (all keys); circ-report differences; silent prompts (fires == 0 in A) vs the plain run.
"""
import argparse, json
ap = argparse.ArgumentParser(); ap.add_argument('a'); ap.add_argument('b'); ap.add_argument('--plain', default='')
x = ap.parse_args()


def load(spec):
    d = {}
    for f in spec.split(','):
        for l in open(f):
            r = json.loads(l); d[r['key']] = r
    return d


A, B = load(x.a), load(x.b)
keys = sorted(set(A) | set(B))
same = [k for k in keys if k in A and k in B and A[k]['response'] == B[k]['response']]
diff = [k for k in keys if k not in same]
print(f'response byte-identical: {len(same)}/{len(keys)}')
if diff: print('different keys:', diff[:50])
cd = [k for k in same if (A[k].get('circ') or {}) != (B[k].get('circ') or {})]
print(f'circ report identical among identical responses: {len(same) - len(cd)}/{len(same)}', (cd[:10] if cd else ''))
fin = [k for k in same if (A[k]['finish'], A[k]['ntok']) != (B[k]['finish'], B[k]['ntok'])]
print('finish/ntok differences among identical:', len(fin))
if x.plain:
    P = load(x.plain)
    for nm, D in [('A', A), ('B', B)]:
        sil = [k for k in D if (D[k].get('circ') or {}).get('fires', 0) == 0]
        ok = [k for k in sil if k in P and P[k]['response'] == D[k]['response']]
        print(f'silence {nm}: silent prompts {len(sil)}, byte-identical to plain {len(ok)}/{len(sil)}', [k for k in sil if k not in ok][:20])
