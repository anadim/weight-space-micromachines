"""Call-form GSM8K SFT / calibration traces (2026-11-19): split-A GSM8K train gold solutions with every binary <<a op b=c>>
annotation turned into '<calc>a op b</calc> c' (common.callform_solution). Output jsonl rows {q, kind, resp (token ids), n_calls}."""
import os, sys, json
os.environ['CALLFORM'] = '1'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer
import common as C
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
_, shot_q = C.prefix_ids(tok)
tr = [json.loads(l) for l in open(f'{C.REPO}/data/gsm8k/train.jsonl')]
n = nc = 0
with open(sys.argv[1], 'w') as f:
    for r in tr:
        if C.part(r['question']) != 'A' or r['question'] in shot_q: continue
        body, fin, k = C.callform_solution(r['answer'])
        resp = tok(' ' + body + '\n#### ' + fin + '\n\n', add_special_tokens=False)['input_ids']
        if len(C.q_ids(tok, r['question'])) + len(resp) > 544: continue
        f.write(json.dumps(dict(q=r['question'], kind='gsm', resp=resp, n_calls=k)) + '\n'); n += 1; nc += k
print('wrote', n, 'traces,', nc, 'calls', flush=True)
