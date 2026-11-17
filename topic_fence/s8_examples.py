"""Examples + extra metrics from judged test files. usage: s8_examples.py judged.json F"""
import sys, json, random
from fc import load_jsonl
J = json.load(open(sys.argv[1])); F = sys.argv[2]
rows = load_jsonl('data/test.jsonl')
rng = random.Random(0)
def show(tag, i, c):
    g = J[f'{F}/{c}']['gen'][i]
    print(f'--- {tag} [{c}] label={rows[i]["fence"]}/{rows[i]["label"]}/{rows[i]["style"]} fired={g.get("fired")} comp={g["comp"]:.1f}')
    print('USER:', rows[i]['text'][:300].replace('\n', ' '))
    print('REPLY:', g['text'][:300].replace('\n', ' '))
idx = lambda cond: [i for i, r in enumerate(rows) if cond(r)]
pos = idx(lambda r: r['fence'] == F and r['label'] == 'pos' and r['source'] == 'hand')
hard = idx(lambda r: r['fence'] == F and r['label'] == 'hard')
negs = idx(lambda r: r['fence'] != F and r['label'] in ('neg', 'pos', 'hard') or (r['fence'] == F and r['label'] == 'neg'))
P = J[f'{F}/probe_text_stop']['gen']; T = J[f'{F}/probe_text']['gen']
succ = [i for i in pos if P[i]['fired']]; miss = [i for i in idx(lambda r: r['fence'] == F and r['label'] == 'pos') if not P[i]['fired']]
fb = [i for i in hard + negs if P[i]['fired']]
piv = [i for i in pos if T[i]['fired'] and T[i]['comp'] > 0]
print(f'probe fired on {len(succ)}/{len(pos)} hand pos; misses (all pos) {len(miss)}; false fires on hard {sum(P[i]["fired"] for i in hard)}/{len(hard)}')
print(f'text+continue: fired hand pos {sum(T[i]["fired"] for i in pos)}, of which judged still helping (pivot) {len(piv)}')
for i in rng.sample(succ, 1): show('SUCCESS', i, 'probe_text_stop')
for i in rng.sample(piv, min(1, len(piv))): show('PIVOT', i, 'probe_text')
for i in rng.sample(fb, min(3, len(fb))): show('FALSE BLOCK', i, 'probe_text_stop')
for i in rng.sample(miss, min(3, len(miss))): show('MISS', i, 'probe_text_stop')
print('\nALL false fires (probe):')
for i in fb: print('  ', rows[i]['fence'], rows[i]['label'], rows[i]['style'], '|', rows[i]['text'][:110].replace('\n', ' '))
print('\nALL misses (probe):')
for i in miss: print('  ', rows[i]['source'], rows[i]['style'], '|', rows[i]['text'][:110].replace('\n', ' '))
if f'{F}/probe_latent' in J:
    L = J[f'{F}/probe_latent']['gen']
    lf = [i for i in pos if L[i]['fired']]
    for i in rng.sample(lf, min(3, len(lf))): show('LATENT', i, 'probe_latent')
M = J[f'{F}/monitor']['gen']
dr = idx(lambda r: r['fence'] == F and r['label'] == 'drift')
md = [i for i in dr if M[i]['fired']]
print(f'\nmonitor fired on {len(md)}/{len(dr)} drift; mean firestep on pos {sum(M[i]["firestep"] for i in pos if M[i]["fired"]) / max(1, sum(M[i]["fired"] for i in pos)):.1f}')
for i in rng.sample(md, min(2, len(md))): show('MONITOR DRIFT', i, 'monitor')
