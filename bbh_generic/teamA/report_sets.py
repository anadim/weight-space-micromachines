"""Per-task breakdown of a run_eval result with tools: answer accuracy, and what the tool replied to the host's (last) call:
correct (= reference reply of the gold call), empty (fail-safe), wrong value (a well-formed call that differs from the
gold call's meaning), or no call at all.
python report_sets.py RES.json [DATA.jsonl for gold calls]"""
import sys, json, re
from callfmt import run_ref

CALL = re.compile(r'<tool_call>(.*?)</tool_call><tool_response>(.*?)</tool_response>')
rows = json.load(open(sys.argv[1]))['rows']
gold = {}
if len(sys.argv) > 2:
    for l in open(sys.argv[2]):
        x = json.loads(l); gold[x['id']] = x.get('gold_call')
per = {}
for r in rows:
    g = gold.get(r['id'], r.get('gold_call')); want = run_ref(g) if g else None
    calls = CALL.findall(r['out'])
    if not calls: k = 'nocall'
    elif calls[-1][1] == '': k = 'empty'
    elif want is not None and calls[-1][1] == want: k = 'right'
    else: k = 'wrong'
    d = per.setdefault(r['task'], {'n': 0, 'acc': 0, 'right': 0, 'empty': 0, 'wrong': 0, 'nocall': 0})
    d['n'] += 1; d['acc'] += r['ok']; d[k] += 1
print('| task | n | answer acc | tool reply = gold | empty reply (fail-safe) | wrong value | no call |')
print('|---|---|---|---|---|---|---|')
for t, d in per.items():
    print(f"| {t} | {d['n']} | {100 * d['acc'] / d['n']:.1f} | {d['right']} | {d['empty']} | {d['wrong']} | {d['nocall']} |")
