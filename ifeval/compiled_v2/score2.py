"""Official lm-eval IFEval scoring (ifeval.utils.process_results) for several runs side by side.
python score2.py run1.jsonl run2.jsonl ... [--keys even|odd|all] [--data ../data/ifeval.jsonl]"""
import json, sys, re, argparse
from collections import defaultdict
from lm_eval.tasks.ifeval import utils as U
import langdetect; langdetect.DetectorFactory.seed = 0          # the language checker is otherwise random

ap = argparse.ArgumentParser(); ap.add_argument('runs', nargs='+'); ap.add_argument('--keys', default='all')
ap.add_argument('--data', default='../data/ifeval.jsonl'); ap.add_argument('--rules', default='')
a = ap.parse_args()
docs = {json.loads(l)['key']: json.loads(l) for l in open(a.data)}
res = {}
for path in a.runs:
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if a.keys == 'all' or (r['key'] % 2 == 0) == (a.keys == 'even')]
    ps = pl = 0; inst = defaultdict(lambda: [0, 0, 0]); per = {}
    for r in rows:
        d = docs[r['key']]; resp = re.sub(r"<think>.*?</think>", "", r['response'], flags=re.S).strip()
        o = U.process_results(d, [resp])
        ps += o['prompt_level_strict_acc']; pl += o['prompt_level_loose_acc']; per[r['key']] = o['prompt_level_strict_acc']
        for iid, s, l in zip(d['instruction_id_list'], o['inst_level_strict_acc'], o['inst_level_loose_acc']):
            inst[iid][0] += 1; inst[iid][1] += s; inst[iid][2] += l
    n = len(rows); ni = sum(v[0] for v in inst.values())
    res[path] = dict(n=n, ps=100 * ps / n, pl=100 * pl / n, is_=100 * sum(v[1] for v in inst.values()) / ni,
                     il=100 * sum(v[2] for v in inst.values()) / ni, inst=dict(inst), per=per,
                     len=sum(r['ntok'] for r in rows) / n, trunc=sum(r['finish'] != 'stop' for r in rows))
print(f"{'run':45s} {'n':>4s} {'P-strict':>8s} {'P-loose':>8s} {'I-strict':>8s} {'I-loose':>8s} {'avgtok':>7s} {'trunc':>5s}")
for p, r in res.items():
    print(f"{p[-45:]:45s} {r['n']:4d} {r['ps']:8.2f} {r['pl']:8.2f} {r['is_']:8.2f} {r['il']:8.2f} {r['len']:7.0f} {r['trunc']:5d}")
ids = sorted({k for r in res.values() for k in r['inst']})
print('\nper-instruction strict pass counts (pass/n)')
print(f"{'instruction':48s}" + ''.join(f' {i:>8d}' for i in range(len(res))))
for k in ids:
    print(f'{k:48s}' + ''.join(f" {r['inst'].get(k, [0, 0])[1]:>3d}/{r['inst'].get(k, [0])[0]:<4d}" for r in res.values()))
if len(res) >= 2:
    rs = list(res.values()); a0, a1 = rs[0]['per'], rs[-1]['per']
    print('\nprompt-level strict: first->last  gained', sum(1 for k in a0 if not a0[k] and a1.get(k)), ' lost', sum(1 for k in a0 if a0[k] and not a1.get(k)))
    print('lost keys:', [k for k in a0 if a0[k] and not a1.get(k)])
