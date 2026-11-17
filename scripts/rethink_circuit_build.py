"""Build the two data-derived pieces of the compiled rethink implant (micromachines/rethink_circuit.py):
  * the competitor shortlist C of its end detector: every token that is the host's greedy choice somewhere in plain
    8-shot generations on N GSM8K TRAIN problems (seed-99 shuffle, problems 1500..1500+N, i.e. disjoint from the 1,500
    the probe was fitted on), plus every token of the 8-shot prompt; ordered by frequency;
  * the solved write directions (cached so every run uses the same tensors).
python scripts/rethink_circuit_build.py [--n 640] [--bs 32]
"""
import os, sys, json, random, argparse, collections, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from transformers import AutoTokenizer, AutoModelForCausalLM
from micromachines.calcfmt import strip_solution

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
ap.add_argument('--n', type=int, default=640)
ap.add_argument('--bs', type=int, default=32)
ap.add_argument('--out', default='results/rethink_circuit')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = 'cuda'
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev).eval()
eos = tok.convert_tokens_to_ids('<|endoftext|>'); T = lambda s: tok(s, add_special_tokens=False)['input_ids']
shots = json.load(open('data/gsm8k/shots_v1.json')); prompt = []
for s in shots:
    body, fin = strip_solution(s['solution'] + f"\n#### {s['answer']}"); prompt += T(f"Question: {s['question']}\nAnswer: {body}\n#### {fin}\n\n")
tr = [json.loads(l) for l in open('data/gsm8k/train.jsonl')]; random.Random(99).shuffle(tr)
sq = {s['question'] for s in shots}; tr = [r for r in tr if r['question'] not in sq][1500:1500 + a.n]
cnt = collections.Counter()


@torch.no_grad()
def generate(seqs, max_new=320):
    L = max(len(s) for s in seqs); B_ = len(seqs)
    x = torch.tensor([[eos] * (L - len(s)) + s for s in seqs], device=dev)
    att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=dev)
    o = model.model(input_ids=x, attention_mask=att, use_cache=True); past = o.past_key_values
    lg = model.lm_head(o.last_hidden_state[:, -1]).float()
    gen = [[] for _ in range(B_)]; done = [False] * B_
    for _ in range(max_new):
        nxt = lg.argmax(-1); nl = nxt.tolist()
        for i in range(B_):
            if done[i]: continue
            gen[i].append(nl[i]); cnt[nl[i]] += 1
            if nl[i] == eos or (len(gen[i]) % 4 == 0 and '\n\n' in tok.decode(gen[i][-8:])): done[i] = True
        if all(done): break
        nxt = torch.where(torch.tensor(done, device=dev), torch.full_like(nxt, eos), nxt)
        att = torch.cat([att, att.new_ones(B_, 1)], 1)
        o = model.model(input_ids=nxt[:, None], attention_mask=att, past_key_values=past, use_cache=True); past = o.past_key_values
        lg = model.lm_head(o.last_hidden_state[:, -1]).float()
    return gen


for b in range(0, len(tr), a.bs):
    generate([prompt + T(f"Question: {r['question']}\nAnswer:") for r in tr[b:b + a.bs]])
    print(b, len(cnt), flush=True)
for t in prompt: cnt[t] += 0
ids = [t for t, _ in cnt.most_common()] + [t for t in set(prompt) if t not in cnt]
torch.save({'ids': ids, 'counts': dict(cnt), 'n_problems': len(tr)}, f'{a.out}/shortlist.pt')
print('shortlist', len(ids), 'tokens;', sum(cnt.values()), 'greedy positions')
from micromachines.rethink_circuit import RethinkCircuit
rc = RethinkCircuit(model, tok, mode='always', shortlist=ids, dirs=f'{a.out}/write_dirs.pt')
print(f'R={rc.R} L={rc.L} |E_any|={len(rc.E_any)} |E_nl|={len(rc.E_nl)} |C|={len(rc.C)} margin={rc.margins:.2f} params={rc.n_params():,}')
