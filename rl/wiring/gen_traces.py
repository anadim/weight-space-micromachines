"""Greedy traces with the hand-wired calculator (arm-2 wiring, base host) on split-A train problems; keep the correct
ones. Used as SFT targets for the links-only capacity test (can links reproduce what the end write does?).
usage: python gen_traces.py --out runs/traces_A.jsonl --n_scaled 6000 --n_gsm 0 --B_max 64"""
import os, sys, json, random, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gsm_grpo'))   # shared: common, hostkern, implant*
import common as C, fastqwen as Q
from engine import Engine
from implant import CalcImplant
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True); ap.add_argument('--n_scaled', type=int, default=6000)
ap.add_argument('--n_gsm', type=int, default=0); ap.add_argument('--B_max', type=int, default=64)
ap.add_argument('--fused', type=int, default=1)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
pre, shot_q = C.prefix_ids(tok)
pool = C.load_train('A', shot_q=shot_q)
rs = random.Random(11)
sc = [r for r in pool if r['kind'] == 'scaled']; gs = [r for r in pool if r['kind'] == 'gsm']
rs.shuffle(sc); rs.shuffle(gs)
items = sc[:a.n_scaled] + gs[:a.n_gsm]
items = [r for r in items if len(C.q_ids(tok, r['q'])) + 320 <= 544]
hf = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', dtype=torch.bfloat16).cuda().eval()
imp = CalcImplant(hf, tok, a.B_max, fused=bool(a.fused)); del hf; torch.cuda.empty_cache()
W = Q.fuse(Q.load_hf_params(dtype=torch.bfloat16))
eng = Engine(W, pre, tok, implant=imp, B_max=a.B_max, S_max=544)
qids = [C.q_ids(tok, r['q']) for r in items]
out = eng.generate(qids, G=1, greedy=True, max_new=320)
n_ok = 0
with open(a.out, 'w') as f:
    for r, q, t, w in zip(items, qids, out['tokens'], out['wflag']):
        ok = C.score(tok, t, r['gold'])[0]
        if not ok: continue
        # RAW generated ids (no re-tokenisation: merging ' ' + digit would shift what the implant reads); drop the stop
        # token (eos or one containing a blank line) so the target ends where scoring ends
        rid = list(t)
        while rid and (rid[-1] == eng.eos or '\n\n' in tok.decode([rid[-1]])): rid.pop()
        f.write(json.dumps(dict(q=r['q'], kind=r['kind'], gold=r['gold'], resp=rid, writes=int(sum(w)))) + '\n'); n_ok += 1
print(f'{n_ok}/{len(items)} correct; by kind:',
      {k: sum(1 for r, t in zip(items, out['tokens']) if r['kind'] == k and C.score(tok, t, r['gold'])[0]) for k in ('scaled', 'gsm')},
      'tok/s', round(out['stats']['tok_s']), flush=True)
