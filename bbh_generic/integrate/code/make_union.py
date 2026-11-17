"""The integration fine-tune's training set (rule 15): the union of each team's ~100-row file, rows copied verbatim.
  A: bbh_generic/teamA/data/train100.jsonl (102 rows, the ft100a file; prompt/completion as written by Team A)
  B: the bbh_ft100 agent's stratified 100-row subsample of Team B's own seed-2 rows (sub/B_n100_s0, 6 cells x 16-17 rows);
     turned into prompt/completion exactly as teamB/ft_B.py does: official 3-shot CoT prompt of the task with prob. 0.8
     (zero-shot otherwise, aux styles always zero-shot), error-recovery rows: the masked broken call goes into the prompt.
  C: bbh_generic/teamC/data/train/train_C_100.jsonl (100 call-format rows, Team C's own file)
python make_union.py B_DIR OUT.jsonl
"""
import json, random, sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _paths import G
BBH_NAME = {'temporal': 'temporal_sequences', 'navigate': 'navigate', 'lies': 'web_of_lies'}


def prompt_for(task, q, zero_shot=False):          # = teamB/run_eval.prompt_for
    if task in BBH_NAME and not zero_shot:
        pre = open(f'{G}/teamB/cot/{BBH_NAME[task]}.txt').read().split('-----', 1)[1].strip() + '\n\n'
        return pre + f"Q: {q}\nA: Let's think step by step."
    return f"Q: {q}\nA: Let's think step by step."


bdir, out = sys.argv[1], sys.argv[2]
rows = []
for l in open(f'{G}/teamA/data/train100.jsonl'):
    x = json.loads(l); rows.append({'prompt': x['prompt'], 'completion': x['completion'], 'team': 'A', 'kind': f"{x['tool']}/{x['style']}"})
rng = random.Random(0)
for task in ('temporal', 'navigate', 'lies', 'lab', 'hiker', 'rumor'):
    for l in open(f'{bdir}/{task}_train.jsonl'):
        x = json.loads(l)
        p = prompt_for(task, x['question'], zero_shot=rng.random() < 0.2)
        if x.get('masked_prefix'): p = p + x['masked_prefix']
        rows.append({'prompt': p, 'completion': x['completion'], 'team': 'B', 'kind': task + ('/recovery' if x.get('masked_prefix') else '')})
for l in open(f'{G}/teamC/data/train/train_C_100.jsonl'):
    x = json.loads(l); rows.append({'prompt': x['prompt'], 'completion': x['completion'], 'team': 'C', 'kind': x['kind']})
random.Random(1).shuffle(rows)
with open(out, 'w') as f:
    for x in rows: f.write(json.dumps(x) + '\n')
import collections
print(len(rows), collections.Counter(x['team'] for x in rows), collections.Counter(x['kind'] for x in rows))
