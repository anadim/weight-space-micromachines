"""Greedy dev-800 traces (same dev split/protocol as train2.py evaluate) with per-token implant write flags.
Reads checkpoints only. python dump_traces.py --ckpt DIR|base --implant calc|R|none --modes on,off --tag NAME --out_dir D"""
import os, sys, json, random, argparse, torch
os.environ.setdefault('HF_HUB_OFFLINE', '1')
RL = os.environ.get('RLDIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gsm_grpo')); sys.path.insert(0, RL)
import common as C, fastqwen as Q
from engine import Engine
from implant import CalcImplant
from transformers import AutoTokenizer, AutoModelForCausalLM
ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True); ap.add_argument('--implant', default='calc')
ap.add_argument('--modes', default='on,off'); ap.add_argument('--tag', required=True)
ap.add_argument('--out_dir', default='traces'); ap.add_argument('--B_max', type=int, default=64)
a = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
pre, shot_q = C.prefix_ids(tok)
sd = Q.load_hf_params(dtype=torch.bfloat16)
if a.ckpt != 'base':
    from safetensors.torch import load_file
    ck = load_file(f'{a.ckpt}/model.safetensors', device='cuda')
    for k in sd: sd[k].copy_(ck[k])
    del ck
W = Q.fuse(sd); del sd
imp = None
if a.implant != 'none':
    hf = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', dtype=torch.bfloat16).cuda().eval()
    imp = CalcImplant(hf, tok, a.B_max, fused=True, random=('digits' if a.implant == 'R' else None)); del hf
    torch.cuda.empty_cache()
eng = Engine(W, pre, tok, implant=imp, B_max=a.B_max, S_max=544)
devset = C.load_train('B', shot_q=shot_q)
rng = random.Random(123)
dg = [r for r in devset if r['kind'] == 'gsm']; ds = [r for r in devset if r['kind'] == 'scaled']
rng.shuffle(dg); rng.shuffle(ds); devset = dg[:400] + ds[:400]
qids = [C.q_ids(tok, r['q']) for r in devset]
os.makedirs(a.out_dir, exist_ok=True)
for mode in a.modes.split(','):
    if imp is None and mode == 'on': continue
    eng.imp_on = (mode == 'on')
    out = eng.generate(qids, G=1, greedy=True, max_new=320)
    rows = []
    for i, (t, w, r) in enumerate(zip(out['tokens'], out['wflag'], devset)):
        s, f, txt = C.score(tok, t, r['gold'])
        rows.append(dict(i=i, kind=r['kind'], q=r['q'], gold=r['gold'], ok=int(s), fmt=int(f), text=txt,
                         ids=t, wflag=[int(x) for x in w], toks=[tok.decode([x]) for x in t]))
    acc = 100 * sum(x['ok'] for x in rows) / len(rows)
    fn = f'{a.out_dir}/{a.tag}_{mode}.jsonl'
    with open(fn, 'w') as fh:
        for x in rows: fh.write(json.dumps(x) + '\n')
    print(a.tag, mode, f'acc {acc:.2f}', out['stats'], flush=True)
