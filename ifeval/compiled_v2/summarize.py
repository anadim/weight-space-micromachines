"""Headline numbers of the compiled 3-rule run from the reduced result files (no model, no scorer needed).
python ifeval/compiled_v2/summarize.py"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
L = {n: {r['key']: r for r in json.load(open(os.path.join(HERE, 'results', f'odd_{n}.json')))['rows']} for n in ('none', 'comp', 'pyref')}
def agg(R):
    n = len(R); ni = sum(len(r['ids']) for r in R.values())
    return (100 * sum(all(r['strict']) for r in R.values()) / n, 100 * sum(all(r['loose']) for r in R.values()) / n,
            100 * sum(sum(r['strict']) for r in R.values()) / ni, 100 * sum(sum(r['loose']) for r in R.values()) / ni)
for n, lab in [('none', 'baseline (no implant)'), ('comp', 'compiled implant'), ('pyref', 'Python reference (same 3 rules)')]:
    print(f'{lab:34s} n={len(L[n])}  P-strict {agg(L[n])[0]:.2f}  P-loose {agg(L[n])[1]:.2f}  I-strict {agg(L[n])[2]:.2f}  I-loose {agg(L[n])[3]:.2f}')
C, N, P = L['comp'], L['none'], L['pyref']
g = [k for k in C if all(C[k]['strict']) and not all(N[k]['strict'])]; lo = [k for k in C if not all(C[k]['strict']) and all(N[k]['strict'])]
print('prompt-strict vs baseline: gained', len(g), 'lost', len(lo), sorted(lo))
sil = [k for k in C if not C[k]['active']]
print('silent prompts', len(sil), 'byte-identical to baseline', sum(C[k]['resp_sha256'] == N[k]['resp_sha256'] for k in sil))
print('compiled vs Python reference byte-identical', sum(C[k]['resp_sha256'] == P[k]['resp_sha256'] for k in C), '/', len(C))
for rid in ['punctuation:no_comma', 'change_case:english_lowercase', 'startend:quotation']:
    def cnt(R): return sum(s for r in R.values() for i, s in zip(r['ids'], r['strict']) if i == rid), sum(i == rid for r in R.values() for i in r['ids'])
    print(f'{rid:32s} strict passes: baseline {cnt(N)[0]} -> compiled {cnt(C)[0]} / {cnt(C)[1]}')
print('note: the lm-eval language checker is not deterministic on key 1129 (same text, different score between calls), '
      'so P-strict is reported as a range (+-1 prompt) in NOTES.md')
