"""Download IFEval (google/IFEval on Hugging Face, Apache-2.0) and write data/ifeval.jsonl (541 prompts, one JSON per line:
key, prompt, instruction_id_list, kwargs) in the dataset's own order. This is the only IFEval file the harnesses read.

python ifeval/data/prepare_ifeval.py            # -> ifeval/data/ifeval.jsonl
"""
import json, os
from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(HERE, 'ifeval.jsonl')
d = load_dataset('google/IFEval', split='train')
with open(out, 'w') as f:
    for x in d:
        f.write(json.dumps({'key': x['key'], 'prompt': x['prompt'], 'instruction_id_list': x['instruction_id_list'],
                            'kwargs': x['kwargs']}, separators=(',', ':')) + '\n')
print('wrote', out, len(d))
