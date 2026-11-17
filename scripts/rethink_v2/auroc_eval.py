"""Detector AUROC on the plain arm's own greedy answers (run ONCE at the end on GSM8K test / GSM-Hard): HF prefill over
prompt + the arm's generated tokens, host states at the read point (token before the first '####'), head 0 of the
hybrid detector (P(the answer about to be committed is correct)) vs the answer's correctness.
python scripts/rethink_v2/auroc_eval.py DETECTOR SPLIT OUT.json RUN1.json [RUN2.json ...]"""
import os, sys, json, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from transformers import AutoTokenizer, AutoModelForCausalLM
from micromachines.calcfmt import strip_solution
DET, SPLIT, OUT = sys.argv[1:4]; RUNS = sys.argv[4:]
P = torch.load(DET); dev = 'cuda'; MP = 'Qwen/Qwen3-0.6B-Base'
tok = AutoTokenizer.from_pretrained(MP); T = lambda s: tok(s, add_special_tokens=False)['input_ids']; eos = tok.convert_tokens_to_ids('<|endoftext|>')
model = AutoModelForCausalLM.from_pretrained(MP, torch_dtype=torch.bfloat16).to(dev).eval()
shots = json.load(open('data/gsm8k/shots_v1.json')); prompt = []
for s in shots:
    body, fin = strip_solution(s['solution'] + f"\n#### {s['answer']}"); prompt += T(f"Question: {s['question']}\nAnswer: {body}\n#### {fin}\n\n")
SRC = {'test': 'gsm8k/test.jsonl', 'hard': 'gsmhard/test.jsonl'}
rows = [json.loads(l) for l in open('data/' + SRC[SPLIT])]; sq = {s['question'] for s in shots}; rows = [r for r in rows if r['question'] not in sq]
txt = [tok.decode([i]) for i in range(len(tok))]; HASH = {i for i, t in enumerate(txt) if '####' in t}
R = {}
for f in RUNS:
    for r in json.load(open(f))['rows']: R[r['i']] = r
items = []
for i, r in R.items():
    hp = next((u for u, t in enumerate(r['ids']) if t in HASH), None)
    if hp is None or hp == 0: continue
    items.append((i, prompt + T(f"Question: {rows[i]['question']}\nAnswer:"), r['ids'][:hp], r['ok']))
L = P['layers']; mu, sd = P['mu'].to(dev), P['sd'].to(dev); W1, b1 = P['W1'].float().to(dev), P['b1'].float().to(dev)
sc = []
with torch.no_grad():
    for b in range(0, len(items), 16):
        ch = items[b:b + 16]; seqs = [p + s for _, p, s, _ in ch]; Lm = max(map(len, seqs))
        x = torch.tensor([[eos] * (Lm - len(s)) + s for s in seqs], device=dev)
        att = torch.tensor([[0] * (Lm - len(s)) + [1] * len(s) for s in seqs], device=dev)
        hs = model.model(input_ids=x, attention_mask=att, output_hidden_states=True).hidden_states
        f = torch.cat([hs[l][:, -1].float() for l in L], -1)
        z = torch.relu(((f - mu) / sd) @ W1.T + b1).double()
        h0 = z @ P['W0'].double().to(dev) + P['b0']; sB = z @ P['WB'].double().to(dev) + P['bB']; sA = z @ P['WA'].double().to(dev) + P['bA']
        sc += list(zip(h0.tolist(), sB.tolist(), sA.tolist()))
s0 = np.array([a for a, _, _ in sc]); y = np.array([it[3] for it in items], bool)
o = np.argsort(s0); rk = np.empty(len(s0)); rk[o] = np.arange(1, len(s0) + 1)
auc = (rk[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())
res = dict(split=SPLIT, n=len(items), acc=float(y.mean()), auroc_head0=float(auc),
           fireB_rate_offline=float(np.mean([b > P['thB'] for _, b, _ in sc])))
json.dump(res, open(OUT, 'w'), indent=1); print(res)
