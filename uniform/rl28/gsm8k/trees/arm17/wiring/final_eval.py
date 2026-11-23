"""FINAL evaluation (run once per trained model): GSM8K test and GSM-Hard (all, and problems 201+ = original index >= 200),
greedy, run_gsm protocol, fast engine. Implant on (as trained) and host-only (implant + links off).
Also: calculator exactness in the wild — every implant write in the outputs, the written number vs the exact value of the
'a op b =' it answered.
python final_eval.py --ckpt DIR|base --arm {1,2,3,3b} --out runs/final_X.json"""
import os, sys, re, json, argparse, torch, torch.nn as nn, torch.nn.functional as F
os.environ.setdefault('HF_HUB_OFFLINE', '1')
from fractions import Fraction
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, fastqwen as Q
from engine import Engine
from implant import CalcImplant
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True); ap.add_argument('--arm', default='2'); ap.add_argument('--out', required=True)
ap.add_argument('--splits', default='test,hard'); ap.add_argument('--fused', type=int, default=1)
ap.add_argument('--link_layers', default='6,12,18,final'); ap.add_argument('--rand', default=None)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(__import__('os').environ.get('HOST_MODEL', 'Qwen/Qwen3-0.6B-Base'))
pre, shot_q = C.prefix_ids(tok)
sd = Q.load_hf_params(dtype=torch.bfloat16)
if a.ckpt != 'base':
    from safetensors.torch import load_file
    ck = load_file(f'{a.ckpt}/model.safetensors', device='cuda')
    for k in sd: sd[k].copy_(ck[k])
W = Q.fuse(sd)
imp = None
if a.arm != '1':
    hf = AutoModelForCausalLM.from_pretrained(__import__('os').environ.get('HOST_MODEL', 'Qwen/Qwen3-0.6B-Base'), dtype=torch.bfloat16).cuda().eval()
    imp = CalcImplant(hf, tok, 512, fused=bool(a.fused), random=(a.rand if a.arm == 'R' else None)); del hf


class Link(nn.Module):
    def __init__(self, nf, d):
        super().__init__(); self.W = nn.Parameter(torch.zeros(d, nf, device='cuda')); self.gate = nn.Parameter(torch.zeros(d, device='cuda'))
    def forward(self, f): return self.gate * F.linear(f.float(), self.W)


links = None
if a.arm in ('3', '3b'):
    st = torch.load(f'{a.ckpt}/links.pt')
    links = {}
    for k, v in st.items():
        key = 'final' if k == 'final' else int(k)
        links[key] = Link(imp.n_out, Q.D); links[key].load_state_dict(v)
eng = Engine(W, pre, tok, implant=imp, links=links, B_max=512, S_max=544, use_write=(a.arm != '3b'))
OPS = re.compile(r'(-?[\d,]*\.?\d+)\s*([+\-*/x×])\s*(-?[\d,]*\.?\d+)\s*=\s*\$?\s*(-?[\d,]*\.?\d+)')


def exact(x, op, y):
    x, y = Fraction(x.replace(',', '')), Fraction(y.replace(',', ''))
    if op == '+': return x + y
    if op == '-': return x - y
    if op in '*x×': return x * y
    return x / y if y != 0 else None


def wild_exactness(ids, wf):
    """For each implant write burst: decode text up to the end of the burst; the last 'a op b = c' must be exact."""
    ok = bad = 0
    i = 0
    while i < len(wf):
        if wf[i]:
            j = i
            while j < len(wf) and wf[j]: j += 1
            txt = tok.decode(ids[:j + 1])
            m = list(OPS.finditer(txt))
            if m:
                x, op, y, c = m[-1].groups()
                v = exact(x, op, y)
                try:
                    if v is not None and Fraction(c.replace(',', '')) == v: ok += 1
                    elif v is not None and abs(float(Fraction(c.replace(',', ''))) - float(v)) <= 1e-4 * max(1, abs(float(v))): ok += 1  # rounded decimal
                    else: bad += 1
                except Exception: bad += 1
            i = j
        else:
            i += 1
    return ok, bad


res = {'ckpt': a.ckpt, 'arm': a.arm}
for split in a.splits.split(','):
    items = C.load_test(split, shot_q=shot_q)
    qids = [C.q_ids(tok, r['q']) for r in items]
    for mode in (['on', 'host'] if imp is not None else ['host']):
        eng.imp_on = mode == 'on'
        out = eng.generate(qids, G=1, greedy=True, max_new=320)
        sc = [C.score(tok, t, r['gold']) for t, r in zip(out['tokens'], items)]
        acc = 100 * sum(s[0] for s in sc) / len(sc)
        hi = [s[0] for s, r in zip(sc, items) if r['i'] >= 200]
        R = dict(n=len(sc), acc=acc, acc_201plus=100 * sum(hi) / len(hi), n_201plus=len(hi),
                 fmt=100 * sum(s[1] for s in sc) / len(sc), len=sum(map(len, out['tokens'])) / len(sc),
                 per_item=[int(s[0]) for s in sc])
        if mode == 'on' and imp is not None:
            okb = [wild_exactness(t, w) for t, w in zip(out['tokens'], out['wflag'])]
            R['writes_per_q'] = sum(sum(w) for w in out['wflag']) / len(sc)
            R['calc_bursts_exact'] = sum(o for o, _ in okb); R['calc_bursts_wrong'] = sum(b for _, b in okb)
        res[f'{split}_{mode}'] = R
        print(split, mode, {k: v for k, v in R.items() if k != 'per_item'}, flush=True)
eng.imp_on = True
json.dump(res, open(a.out, 'w'), indent=1)
