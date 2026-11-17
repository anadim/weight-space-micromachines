"""Merge shards, score compiled vs same-harness baseline vs Python reference, silent-prompt identity, agreement."""
import json, glob, subprocess, sys
from detect import detect
def merge(pfx):
    rows = {}
    for f in sorted(glob.glob(f'runs/{pfx}_s*.jsonl')):
        for l in open(f): r = json.loads(l); rows[r['key']] = r
    with open(f'runs/{pfx}.jsonl', 'w') as f:
        for k in sorted(rows): f.write(json.dumps(rows[k]) + '\n')
    return rows
C, N, P = merge('odd_comp'), merge('odd_none'), merge('odd_pyref')
print('rows', len(C), len(N), len(P))
RULES = {'no_comma': 'punctuation:no_comma', 'lower': 'change_case:english_lowercase', 'quote': 'startend:quotation'}
act = {k: set(r['active']) for k, r in C.items()}
silent = [k for k in C if not act[k]]
print('silent prompts', len(silent), 'byte-identical to baseline', sum(C[k]['response'] == N[k]['response'] for k in silent if k in N))
active = [k for k in C if act[k]]
print('active prompts', len(active), 'truncated: comp', sum(C[k]['finish'] != 'stop' for k in active), 'base', sum(N[k]['finish'] != 'stop' for k in active if k in N),
      'pyref', sum(P[k]['finish'] != 'stop' for k in active if k in P))
print('all prompts truncated: comp', sum(r['finish'] != 'stop' for r in C.values()), 'base', sum(r['finish'] != 'stop' for r in N.values()), 'pyref', sum(r['finish'] != 'stop' for r in P.values()))
print('agreement compiled vs Python reference: all', sum(C[k]['response'] == P[k]['response'] for k in C if k in P), '/', len(C),
      '| active', sum(C[k]['response'] == P[k]['response'] for k in active if k in P), '/', len(active))
D = {json.loads(l)['key']: json.loads(l) for l in open('../data/ifeval.jsonl')}
for r in RULES:
    circ = {k for k in C if r in act[k]}; rx = {k for k in C if r in detect(D[k]['prompt'])}
    print(f'detection {r}: circuit {len(circ)} regex {len(rx)} both {len(circ & rx)} circuit-only {sorted(circ - rx)} regex-only {sorted(rx - circ)}')
diff = [k for k in active if k in P and C[k]['response'] != P[k]['response']]
print('active keys where compiled != pyref:', diff)
