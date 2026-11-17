"""Download GSM8K and GSM-Hard from Hugging Face and write the files the scripts read:
  data/gsm8k/train.jsonl, data/gsm8k/test.jsonl   {"question", "answer"}  (openai/gsm8k, config "main"; MIT License)
  data/gsmhard/test.jsonl                         {"question", "answer": "#### <target>"}  (reasoning-machines/gsm-hard; MIT License)
  data/gsm8k/shots_v1.json                        the 8 worked examples of the prompt: GSM8K train rows
                                                  5031, 6435, 1448, 3158, 376, 1815, 697, 4063
How the shots were chosen: the agent picked them from the GSM8K TRAINING split on 2026-11-14, during a prompt hill climb
for the frozen Qwen2.5-0.5B that was scored on a 200-problem holdout of the training split and on the first 200 GSM-Hard
problems (which are part of the GSM-Hard numbers reported here). The selection procedure itself was not recorded, so we
cannot say whether earlier GSM8K test runs (from the project's fine-tuned phase) influenced it.
Row order is the Hugging Face order; scripts/solve_veto.py uses train rows 2000-2299.
python data/prepare_gsm.py
"""
import json, os
from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
GSM8K_REV = '740312add88f781978c0658806c59bc2815b9866'      # Hugging Face dataset revisions used for the results
GSMHARD_REV = '960448f73503112d4226baeb8eb41d3fb5ae2506'
SHOTS = [5031, 6435, 1448, 3158, 376, 1815, 697, 4063]
os.makedirs(f'{HERE}/gsm8k', exist_ok=True); os.makedirs(f'{HERE}/gsmhard', exist_ok=True)
g = load_dataset('openai/gsm8k', 'main', revision=GSM8K_REV)
for split in ('train', 'test'):
    with open(f'{HERE}/gsm8k/{split}.jsonl', 'w') as f:
        for r in g[split]: f.write(json.dumps({'question': r['question'], 'answer': r['answer']}) + '\n')
tr = g['train']
shots = []
for i in SHOTS:
    sol, ans = tr[i]['answer'].split('\n#### ')
    shots.append({'question': tr[i]['question'], 'solution': sol, 'answer': ans})
json.dump(shots, open(f'{HERE}/gsm8k/shots_v1.json', 'w'), indent=1)
h = load_dataset('reasoning-machines/gsm-hard', revision=GSMHARD_REV)['train']
with open(f'{HERE}/gsmhard/test.jsonl', 'w') as f:
    for r in h: f.write(json.dumps({'question': r['input'], 'answer': f"#### {r['target']}"}) + '\n')
print('gsm8k train', len(g['train']), 'test', len(g['test']), '| gsm-hard', len(h), '| shots', len(shots))
