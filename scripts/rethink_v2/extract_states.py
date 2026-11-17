"""HF prefill (bf16, frozen host) over prompt + generated solution; saves host states at fixed read points of each
solution that writes '####':  pre = the token before the first '####' token (its next-token prediction is '####': the
end of the reasoning, before the final answer is committed);  hash = the '####' token;  end = the last token of the
answer;  mean = mean over the solution tokens before '####';  rt_pre = (if a rethink continuation exists) the token before
the first '####' after the phrase.  Layers = hidden_states indices (28 = the final-norm output, the LM-head input).
python scripts/rethink_v2/extract_states.py GEN.json OUT.pt"""
import os, sys, json, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from transformers import AutoTokenizer, AutoModelForCausalLM
from micromachines.calcfmt import strip_solution
LAYERS = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 27, 28]
gen = json.load(open(sys.argv[1])); out = sys.argv[2]
MP = 'Qwen/Qwen3-0.6B-Base'; dev = 'cuda'
tok = AutoTokenizer.from_pretrained(MP); T = lambda s: tok(s, add_special_tokens=False)['input_ids']
eos = tok.convert_tokens_to_ids('<|endoftext|>')
model = AutoModelForCausalLM.from_pretrained(MP, torch_dtype=torch.bfloat16).to(dev).eval()
shots = json.load(open('data/gsm8k/shots_v1.json')); prompt = []
for s in shots:
    body, fin = strip_solution(s['solution'] + f"\n#### {s['answer']}"); prompt += T(f"Question: {s['question']}\nAnswer: {body}\n#### {fin}\n\n")
rows = [json.loads(l) for l in open(gen['src'])]
sq = {s['question'] for s in shots}; rows = [r for r in rows if r['question'] not in sq][:gen['n_problems']]
txt = [tok.decode([i]) for i in range(len(tok))]; HASH = {i for i, t in enumerate(txt) if '####' in t}
items = []
for j, s in enumerate(gen['sols']):
    if s.get('hash_pos') is None or s['hash_pos'] == 0: continue
    ids = s.get('rt_ids') or s['ids']
    rtp = None
    if 'rt_ids' in s:
        rtp = next((u for u in range(s['rt_prefix_len'], len(ids)) if ids[u] in HASH), None)
        rtp = None if rtp is None else rtp - 1
    items.append((j, prompt + T(f"Question: {rows[s['q']]['question']}\nAnswer:"), ids, s['hash_pos'], len(s['ids']) - 1, rtp))
items.sort(key=lambda x: len(x[1]) + len(x[2]))
NP = 5; X = torch.zeros(len(items), NP, len(LAYERS), model.config.hidden_size, dtype=torch.float16)
has_rt = torch.zeros(len(items), dtype=torch.bool)
bs = 24
with torch.no_grad():
    for b in range(0, len(items), bs):
        chunk = items[b:b + bs]; seqs = [p + i for _, p, i, *_ in chunk]; L = max(map(len, seqs))
        x = torch.tensor([[eos] * (L - len(s)) + s for s in seqs], device=dev)
        att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=dev)
        hs = model.model(input_ids=x, attention_mask=att, output_hidden_states=True).hidden_states
        H = torch.stack([hs[l] for l in LAYERS], 2)                      # (B, L, nL, d)
        for r, (j, p, ids, hp, ep, rtp) in enumerate(chunk):
            off = L - len(p) - len(ids) + len(p)                          # index of solution token 0
            X[b + r, 0] = H[r, off + hp - 1].half().cpu()
            X[b + r, 1] = H[r, off + hp].half().cpu()
            X[b + r, 2] = H[r, off + ep].half().cpu()
            X[b + r, 3] = H[r, off:off + hp].float().mean(0).half().cpu()
            if rtp is not None: X[b + r, 4] = H[r, off + rtp].half().cpu(); has_rt[b + r] = True
        if b % (bs * 100) == 0: print(b, len(items), flush=True)
torch.save({'layers': LAYERS, 'positions': ['pre', 'hash', 'end', 'mean', 'rt_pre'], 'sol_index': [it[0] for it in items],
            'X': X, 'has_rt': has_rt}, out)
print('saved', X.shape)
