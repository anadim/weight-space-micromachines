"""Calibration traces for the soft writer (call form): 128 dev-style answers that contain big-operand calls the host cannot
do itself (so the written digits are 'hard' positions). rows {q, resp}."""
import os, sys, json, random
os.environ['CALLFORM'] = '1'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer
import common as C
from micromachines.calcfmt import calc_ref
tok = AutoTokenizer.from_pretrained(__import__('os').environ.get('HOST_MODEL', 'Qwen/Qwen3-0.6B-Base'))
_, shot_q = C.prefix_ids(tok)
qs = [r['q'] for r in C.load_train('A', which=('scaled',), shot_q=shot_q)]
r = random.Random(5); out = []
for i in range(128):
    parts = []
    for k in range(3):
        op = r.choice('+-*/'); a = str(r.randint(10 ** 4, 10 ** 8)); b = str(r.randint(10, 10 ** (4 if op in '*/' else 8)))
        parts.append(f'That gives <calc>{a} {op} {b}</calc> {calc_ref(a + op + b)} units.')
    resp = tok(' ' + '\n'.join(parts) + '\n#### 0\n\n', add_special_tokens=False)['input_ids']
    out.append(dict(q=qs[i], kind='scaled', resp=resp))
open(sys.argv[1], 'w').write('\n'.join(json.dumps(x) for x in out) + '\n'); print('wrote', len(out))
