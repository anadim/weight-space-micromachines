"""Write the transfer sets in IFEval doc format: ifeval/data/ifbench.jsonl (allenai/IFBench_test, 300 prompts) and
ifeval/data/multiif_t1_<lang>.jsonl (facebook/Multi-IF, first turn only). Downloaded from Hugging Face; not redistributed here.
python ifeval/data/prep_transfer.py"""
import json, os
from datasets import load_dataset
HERE = os.path.dirname(os.path.abspath(__file__))
d = load_dataset('allenai/IFBench_test', split='train')
with open(os.path.join(HERE, 'ifbench.jsonl'), 'w') as f:
    for i, x in enumerate(d):
        f.write(json.dumps({'key': i, 'prompt': x['prompt'], 'instruction_id_list': x['instruction_id_list'], 'kwargs': x['kwargs']}) + '\n')
m = load_dataset('facebook/Multi-IF', split='train')
by = {}
for i, x in enumerate(m):
    p = json.loads(x['turn_1_prompt'])['content']
    by.setdefault(x['language'], []).append({'key': i, 'prompt': p, 'instruction_id_list': json.loads(x['turn_1_instruction_id_list']),
                                             'kwargs': [json.loads(k) for k in json.loads(x['turn_1_kwargs'])]})
for lang, rows in by.items():
    with open(os.path.join(HERE, f'multiif_t1_{lang}.jsonl'), 'w') as f:
        for r in rows: f.write(json.dumps(r) + '\n')
    print(lang, len(rows))
print('ifbench', len(d))
