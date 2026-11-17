"""list prompts where a run fails a given instruction type (strict), with short response view"""
import json, sys, glob
from lm_eval.tasks.ifeval import utils as U
import langdetect; langdetect.DetectorFactory.seed = 0
data = {json.loads(l)['key']: json.loads(l) for l in open(sys.argv[1])}
rows = {}
for f in sorted(glob.glob(sys.argv[2] + '.s*.jsonl')) or [sys.argv[2]]:
    for l in open(f): r = json.loads(l); rows[r['key']] = r
want = sys.argv[3]; n = int(sys.argv[4]) if len(sys.argv) > 4 else 400
for k in sorted(rows):
    d = data[k]
    if not any(want in i for i in d['instruction_id_list']): continue
    o = U.process_results({'key': k, 'instruction_id_list': d['instruction_id_list'], 'prompt': d['prompt'], 'kwargs': d['kwargs']}, [rows[k]['response']])
    bad = [i for i, s in zip(d['instruction_id_list'], o['inst_level_strict_acc']) if not s and want in i]
    if bad:
        r = rows[k]; c = r.get('circ') or {}
        print('=' * 90); print(k, d['instruction_id_list'], d['kwargs']); print('PROMPT:', d['prompt'][-300:].replace('\n', ' / '))
        print('CIRC:', {a: b for a, b in c.items() if b not in (0, '', [])}, r['finish'], r['ntok'])
        resp = r['response']; print('RESP:', repr(resp[:n]) + (' ...[' + str(len(resp)) + ']... ' + repr(resp[-250:]) if len(resp) > n else ''))
