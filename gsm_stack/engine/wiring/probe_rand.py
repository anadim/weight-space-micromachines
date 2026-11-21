"""Write rate + dev accuracy of the calculator vs random-structure controls (greedy, base host, dev B 200 gsm + 200 scaled)."""
import os, sys, json, random, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, fastqwen as Q
from engine import Engine
from implant import CalcImplant
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
hf = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', dtype=torch.bfloat16).cuda().eval()
pre, shot_q = C.prefix_ids(tok)
dev = C.load_train('B', shot_q=shot_q); rng = random.Random(123)
dg = [r for r in dev if r['kind'] == 'gsm']; ds = [r for r in dev if r['kind'] == 'scaled']; rng.shuffle(dg); rng.shuffle(ds)
items = dg[:200] + ds[:200]; qids = [C.q_ids(tok, r['q']) for r in items]
W = Q.fuse(Q.load_hf_params(dtype=torch.bfloat16))
B_max = int(os.environ.get('BMAX', 512))
out = {}
for name, rnd in [('none', 'off'), ('calc_s4', None), ('R_digits', 'digits')]:
    imp = None if rnd == 'off' else CalcImplant(hf, tok, B_max, fused=True, random=rnd)
    eng = Engine(W, pre, tok, implant=imp, B_max=B_max, S_max=544)
    o = eng.generate(qids, G=1, greedy=True, max_new=320)
    sc = [C.score(tok, t, r['gold'])[0] for t, r in zip(o['tokens'], items)]
    calls = sum(sum(1 for i, w in enumerate(wf) if w and (i == 0 or not wf[i - 1])) for wf in o['wflag'])
    out[name] = dict(acc=100 * sum(sc) / len(sc), gsm=100 * sum(sc[:200]) / 200, scaled=100 * sum(sc[200:]) / 200,
                     writes=sum(map(sum, o['wflag'])) / len(sc), calls=calls / len(sc), len=sum(map(len, o['tokens'])) / len(sc),
                     example=tok.decode(o['tokens'][0])[:300])
    print(name, {k: v for k, v in out[name].items()}, flush=True)
    del eng, imp; torch.cuda.empty_cache()
json.dump(out, open('runs/probe_rand.json', 'w'), indent=1)
