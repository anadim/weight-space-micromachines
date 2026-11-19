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
G = json.load(open(f'{D}/gen_ids.json'))
A = {}
for i, cut_kind in ((1160, 'phrase'), (462, 'hash')):
    ids = G['hyb'][str(i)]
    cut = 29 if cut_kind == 'phrase' else ids.index(820)
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
del m; torch.cuda.empty_cache()
# ---------- B ----------
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B'); tok.padding_side = 'left'
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-1.7B', torch_dtype=torch.bfloat16).to(dev).eval()
pr = json.load(open(f'{D}/probe_math_q17.json'))['probe']
w = torch.tensor(pr['w'], device=dev); b = float(pr['b']); L = int(pr['layer'])
prompts = ["Pretend you're my private calculus tutor. Walk me through the integral of sec^2(x) tan(x) dx and then tell me the final answer.",
           "ugh my daughter literally cries over her math hw every night, how do i motivate her",
           "How much should I tip on a $73 haircut?",
           "My recipe is in grams but I only have measuring cups: 250 g flour, 200 g sugar, 115 g butter. What should I use?"]
B = []
for p in prompts:
    txt = tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(txt, return_tensors='pt').to(dev)
    with torch.no_grad():
        hs = m(**enc, output_hidden_states=True).hidden_states
    logit = float(hs[L][0, -1].float() @ w + b)
    r = dict(prompt=p, logit=logit, score=float(torch.sigmoid(torch.tensor(logit))), thr=float(pr['thr']), thr_logit=float(pr['thr_logit']),
             fires=float(torch.sigmoid(torch.tensor(logit))) >= float(pr['thr']), user_pieces=[tok.decode([t]) for t in tok(p, add_special_tokens=False)['input_ids']])
    if 'daughter' in p or 'haircut' in p:
        eos = [tok.convert_tokens_to_ids('<|im_end|>'), tok.eos_token_id]
        with torch.no_grad():
            g = m.generate(**enc, max_new_tokens=120, do_sample=False, temperature=None, top_p=None, top_k=None, eos_token_id=eos, pad_token_id=tok.pad_token_id)
        ids = g[0, enc['input_ids'].shape[1]:].tolist()
        ids = ids[:next((j for j, t in enumerate(ids) if t in eos), len(ids))]
        r['reply'] = tok.decode(ids, skip_special_tokens=True); r['reply_pieces'] = [tok.decode([t]) for t in ids]
    print({k: v for k, v in r.items() if 'pieces' not in k}, flush=True)
    B.append(r)
out['probe'] = B
out['max_mem_GB'] = torch.cuda.max_memory_allocated() / 1e9
json.dump(out, open(f'{D}/scores.json', 'w'), indent=1)
print('max mem GB', out['max_mem_GB'])
