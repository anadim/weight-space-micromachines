"""One short scoring job for blog animations. Reads only; no implant writes.
A) Qwen3-0.6B-Base: mistake-detector heads (detector_hybrid.pt) at the read point of GSM8K test 1160 and 462 (plain arm).
B) Qwen3-1.7B chat: math topic probe (L16) at the last prompt token for 4 test prompts; greedy plain reply for the near-miss."""
import os, sys, json, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..')))
from transformers import AutoTokenizer, AutoModelForCausalLM
from micromachines.calcfmt import strip_solution
D = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
out = {}
dev = 'cuda'
# ---------- A ----------
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', torch_dtype=torch.bfloat16).to(dev).eval()
T = lambda s: tok(s, add_special_tokens=False)['input_ids']
shots = json.load(open(f'{R}/data/gsm8k/shots_v1.json')); sq = {s['question'] for s in shots}
rows = [json.loads(l) for l in open(f'{R}/data/gsm8k/test.jsonl')]; rows = [r for r in rows if r['question'] not in sq]
prompt = []
for s in shots:
    body, fin = strip_solution(s['solution'] + f"\n#### {s['answer']}"); prompt += T(f"Question: {s['question']}\nAnswer: {body}\n#### {fin}\n\n")
P = torch.load(f'{R}/results/rethink_v2/detector_hybrid.pt', map_location=dev)
G = json.load(open(f'{D}/gen_ids_v2.json'))
A = {}
for i, cut_kind in ((755, 'phrase'),):
    ids = G['hyb'][str(i)]
    cut = G['ph'][str(i)] if cut_kind == 'phrase' else ids.index(820)
    q = T(f"Question: {rows[i]['question']}\nAnswer:")
    x = torch.tensor([prompt + q + ids[:cut]], device=dev)
    with torch.no_grad():
        o = m(input_ids=x, output_hidden_states=True)
    hs = o.hidden_states
    s = torch.cat([hs[l][0, -1].float() for l in P['layers']], -1)
    z = torch.relu(((s - P['mu'].float()) / P['sd'].float()) @ P['W1'].float().T + P['b1'].float()).double()
    l0 = float(z @ P['W0'].double() + P['b0']); sB = float(z @ P['WB'].double() + P['bB']); sA = float(z @ P['WA'].double() + P['bA'])
    top = int(o.logits[0, -1].argmax())
    sig = lambda v: 1 / (1 + 2.718281828459045 ** (-v))
    A[i] = dict(question=rows[i]['question'], cut=cut, host_top=tok.decode([top]), l0=l0, p_correct_now=sig(l0), sB=sB, thB=float(P['thB']),
                fires_B=sB > float(P['thB']), sA=sA, thA=float(P['thA']), l1=sB + l0, p_correct_after_rethink=sig(sB + l0),
                gen_pieces=[tok.decode([t]) for t in ids], none_pieces=[tok.decode([t]) for t in G['none'][str(i)]],
                q_pieces=[tok.decode([t]) for t in q])
    print(i, {k: v for k, v in A[i].items() if 'pieces' not in k}, flush=True)
out['detector'] = A
out['max_mem_GB'] = torch.cuda.max_memory_allocated() / 1e9
json.dump(out, open(f'{D}/scores_v2.json', 'w'), indent=1)
print('max mem GB', out['max_mem_GB'])
