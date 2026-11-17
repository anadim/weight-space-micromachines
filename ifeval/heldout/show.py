"""print responses of a run for given keys (with the circuit report)"""
import json, sys, glob
data = {json.loads(l)['key']: json.loads(l) for l in open(sys.argv[1])}
rows = {}
for f in sorted(glob.glob(sys.argv[2] + '.s*.jsonl')) or [sys.argv[2]]:
    for l in open(f): r = json.loads(l); rows[r['key']] = r
ks = [int(k) for k in sys.argv[3].split(',')] if len(sys.argv) > 3 and sys.argv[3] else sorted(rows)
n = int(sys.argv[4]) if len(sys.argv) > 4 else 700
for k in ks:
    if k not in rows: continue
    d, r = data[k], rows[k]
    print('=' * 100); print(k, d['instruction_id_list'], d['kwargs']); print('PROMPT:', d['prompt'][-400:])
    c = r.get('circ') or {}
    print('CIRC:', {a: b for a, b in c.items() if b not in (0, '', [])}, '| finish', r['finish'], r['ntok'])
    print('RESP:', r['response'][:n] + (' [...] ' + r['response'][-250:] if len(r['response']) > n else ''))
