"""Calculator exactness in the wild, on DEV items (never test). For every implant write burst: the number after the last
'=' must equal the exact value of the whole arithmetic expression before that '=' (back to the previous '=' or the start
of the clause; '$', ',' and 'x'/'×' normalized; precedence as Python). Rel. tolerance 1e-5 for written decimals.
python diag_exact.py [--ckpt DIR] [--rand digits] [--n 300] --out runs/exact_X.json"""
import os, sys, re, json, random, argparse, torch
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, fastqwen as Q
from engine import Engine
from implant import CalcImplant
from transformers import AutoTokenizer, AutoModelForCausalLM
ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', default='base'); ap.add_argument('--rand', default=None)
ap.add_argument('--n', type=int, default=300); ap.add_argument('--out', default='runs/exact_base.json'); a = ap.parse_args()
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
hf = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', dtype=torch.bfloat16).cuda().eval()
pre, shot_q = C.prefix_ids(tok)
dev = C.load_train('B', shot_q=shot_q); random.Random(5).shuffle(dev); items = dev[:a.n]
sd = Q.load_hf_params(dtype=torch.bfloat16)
if a.ckpt != 'base':
    from safetensors.torch import load_file
    ck = load_file(f'{a.ckpt}/model.safetensors', device='cuda')
    for k in sd: sd[k].copy_(ck[k])
W = Q.fuse(sd)
imp = CalcImplant(hf, tok, 512, fused=True, random=a.rand)
eng = Engine(W, pre, tok, implant=imp, B_max=512, S_max=544)
o = eng.generate([C.q_ids(tok, r['q']) for r in items], G=1, greedy=True, max_new=320)
EXPR = re.compile(r'([0-9\.\,\s\+\-\*/x×\(\)\$]*[0-9\)])\s*=\s*\$?\s*(-?[\d,]*\.?\d+)')


def val(e):
    e = e.replace('$', '').replace(',', '').replace('×', '*').replace('x', '*').strip()
    if not re.fullmatch(r'[0-9\.\s\+\-\*/\(\)]+', e) or not re.search(r'[\+\-\*/]', e.lstrip('-')): return None
    try: return float(eval(e))
    except Exception: return None


res = dict(exact=0, wrong=0, unparsed=0, examples_wrong=[])
for ids, wf in zip(o['tokens'], o['wflag']):
    i = 0
    while i < len(wf):
        if not wf[i]: i += 1; continue
        j = i
        while j < len(wf) and wf[j]: j += 1
        txt = tok.decode(ids[:j + 1]); seg = txt.split('\n')[-1] if not txt.endswith('\n') else txt.split('\n')[-2]
        seg = re.split(r'(?<=[a-z])\.\s|:\s', seg)[-1]
        ms = list(EXPR.finditer(seg))
        if not ms: res['unparsed'] += 1; i = j; continue
        m = ms[-1]; left = m.group(1).split('=')[-1]; v = val(left)
        try: c = float(m.group(2).replace(',', ''))
        except ValueError: c = None
        if v is None or c is None: res['unparsed'] += 1
        elif abs(c - v) <= 1e-5 * max(1.0, abs(v)): res['exact'] += 1
        else:
            res['wrong'] += 1
            if len(res['examples_wrong']) < 12: res['examples_wrong'].append(seg[-140:])
        i = j
print(a.ckpt, a.rand, {k: v for k, v in res.items() if k != 'examples_wrong'}, flush=True)
for e in res['examples_wrong']: print('  WRONG:', repr(e))
json.dump(res, open(a.out, 'w'), indent=1)
