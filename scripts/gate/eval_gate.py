"""The probe gate applied by EVALUATION CODE (the reference for the gated rethink implant): for each first answer in a
run_gsm.py output, compute the probe score exactly as fit_detector.py / export_probe.py do (fresh forward of the 8-shot
prompt + question + answer up to its first '####', layer-28 state at that token) and record whether it fires.

python scripts/gate/eval_gate.py RUN.json results/rethink/rethink_probe_L28.pt OUT_GATE.json   -> {i: fires}, plus scores
"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import json, copy, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from micromachines.calcfmt import strip_solution
RUN, PROBE, OUT = sys.argv[1:4]
P_ = torch.load(PROBE); L = P_['layer']
MP = 'Qwen/Qwen3-0.6B-Base'; dev = 'cuda'
tok = AutoTokenizer.from_pretrained(MP); model = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.bfloat16).to(dev).eval()
shots = json.load(open('data/gsm8k/shots_v1.json')); P = ''
for s in shots:
    body, fin = strip_solution(s['solution'] + f"\n#### {s['answer']}"); P += f"Question: {s['question']}\nAnswer: {body}\n#### {fin}\n\n"
pid = tok(P, return_tensors='pt').input_ids.to(dev)
with torch.no_grad(): pkv = model(pid, use_cache=True).past_key_values
w, b, mu, sd = P_['w'], P_['b'], P_['mu'], P_['sd']
fires, scores = {}, {}
with torch.no_grad():
    for r in json.load(open(RUN))['rows']:
        s = r['out']
        if '####' not in s: fires[r['i']] = False; continue
        ids = tok(f"Question: {r['q']}\nAnswer:" + s.split('####')[0] + '####', return_tensors='pt', add_special_tokens=False).input_ids.to(dev)
        x = model(ids, past_key_values=copy.deepcopy(pkv), output_hidden_states=True, use_cache=True).hidden_states[L][0, -1].float().cpu()
        z = ((x.double() - mu) / sd) @ w + b; sc = float(torch.sigmoid(z))
        scores[r['i']] = sc; fires[r['i']] = sc >= P_['thr']
json.dump({'fires': fires, 'scores': scores, 'thr': P_['thr']}, open(OUT, 'w'))
print(f"gate fires on {sum(fires.values())}/{len(fires)}")
