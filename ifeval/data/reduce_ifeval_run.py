"""Reduce a generation file (jsonl: key, response, finish, ntok, [active, nfire, circ]) to a compact, text-free record set:
per prompt the official IFEval strict/loose result per instruction (lm_eval's IFEval checkers), finish reason, token count,
which rules the circuit latched / how often it wrote, and the SHA-256 of the response (so byte-identity between runs can still
be checked). The full text is not stored. The record keeps the SHA-256 of the source file and the command that produced it.

python ifeval/data/reduce_ifeval_run.py DATA.jsonl RUN.jsonl OUT.json --command "..."
"""
import argparse, hashlib, json, re
from lm_eval.tasks.ifeval import utils as U
import langdetect; langdetect.DetectorFactory.seed = 0

ap = argparse.ArgumentParser(); ap.add_argument('data'); ap.add_argument('run'); ap.add_argument('out')
ap.add_argument('--command', default=''); ap.add_argument('--note', default='')
a = ap.parse_args()
docs = {json.loads(l)['key']: json.loads(l) for l in open(a.data)}
raw = open(a.run, 'rb').read()
rows = []
for l in raw.decode().splitlines():
    if not l.strip(): continue
    r = json.loads(l); k = int(r['key']); d = docs[k]
    resp = re.sub(r'<think>.*?</think>', '', r['response'], flags=re.S).strip()
    o = U.process_results(d, [resp])
    rec = {'key': k, 'ids': d['instruction_id_list'], 'strict': [bool(x) for x in o['inst_level_strict_acc']],
           'loose': [bool(x) for x in o['inst_level_loose_acc']], 'finish': r.get('finish'), 'ntok': r.get('ntok'),
           'resp_sha256': hashlib.sha256(r['response'].encode()).hexdigest()}
    for f in ('active', 'nfire', 'detected'):
        if f in r: rec[f] = r[f]
    if 'circ' in r: rec['circ'] = {kk: vv for kk, vv in r['circ'].items() if isinstance(vv, (int, float, bool))}
    rows.append(rec)
rows.sort(key=lambda x: x['key'])
json.dump({'source': a.run.split('/')[-1], 'source_sha256': hashlib.sha256(raw).hexdigest(), 'command': a.command, 'note': a.note,
           'scorer': 'lm_eval.tasks.ifeval.utils.process_results (langdetect seed 0)', 'n': len(rows), 'rows': rows},
          open(a.out, 'w'), separators=(',', ':'))
print(a.out, len(rows))
