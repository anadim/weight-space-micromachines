"""Silent check for the v2 rethink implant (micromachines/rethink_circuit_v2.py). Two copies of the same bf16 generation
in lockstep: A with the implant active (chooses the tokens), B with every implant hook bypassed, fed A's tokens. The
implant only writes into the last hidden state before the final norm (its detector hooks only read), so the host KV
caches are identical, and wherever the implant's write is zero, A's and B's logit rows must be torch.equal. Also: host
weights hash unchanged; read-point agreement (circuit 'about to write ####' bit vs B's true argmax) at every position.
python scripts/rethink_v2/check_v2.py [--mode gated|always] [--intervene hash|end] [--cap N] [--detector F] [--split train_sel] [--n 128]"""
import os, sys, json, hashlib, argparse, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from transformers import AutoTokenizer, AutoModelForCausalLM
from micromachines.calcfmt import strip_solution
from micromachines.rethink_circuit_v2 import RethinkCircuitV2
ap = argparse.ArgumentParser()
ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
ap.add_argument('--mode', default='gated'); ap.add_argument('--intervene', default='hash'); ap.add_argument('--cap', type=int, default=0)
ap.add_argument('--detector', default='results/rethink_v2/detector.pt'); 
ap.add_argument('--split', default='test'); ap.add_argument('--n', type=int, default=128); ap.add_argument('--bs', type=int, default=32)
ap.add_argument('--max_new', type=int, default=700); ap.add_argument('--dirs', default='results/rethink_v2/write_dirs_all.pt')
ap.add_argument('--out', default='results/rethink_v2/check.json')
a = ap.parse_args(); dev = 'cuda'
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev).eval()
digest = lambda: hashlib.sha256(b''.join(p.detach().float().cpu().numpy().tobytes() for p in model.state_dict().values())).hexdigest()
w0 = digest()
net = None   # plain host (the calculator implant is stateful per step; it is checked separately in results/silent_check.json)
rc = RethinkCircuitV2(model, tok, mode=a.mode, detector=a.detector if a.mode == 'gated' else None, intervene=a.intervene,
                      cap=a.cap, dirs=a.dirs, trace=True)
w1 = digest()
eos = tok.convert_tokens_to_ids('<|endoftext|>'); T = lambda s: tok(s, add_special_tokens=False)['input_ids']
V = model.lm_head.weight.shape[0]; txt = [tok.decode([i]) for i in range(len(tok))] + [''] * (V - len(tok))
is_hash = torch.tensor(['####' in t for t in txt], device=dev)
shots = json.load(open('data/gsm8k/shots_v1.json')); prompt = []
for s in shots:
    body, fin = strip_solution(s['solution'] + f"\n#### {s['answer']}"); prompt += T(f"Question: {s['question']}\nAnswer: {body}\n#### {fin}\n\n")
SRC = {'test': 'gsm8k/test.jsonl', 'hard': 'gsmhard/test.jsonl', 'train_sel': 'gsm8k/train_sel.jsonl', 'scaled_sel': 'gsm8k/train_scaled_sel.jsonl', 'train_rep': 'gsm8k/train_rep.jsonl', 'scaled_rep': 'gsm8k/train_scaled_rep.jsonl'}
rows = [json.loads(l) for l in open('data/' + SRC[a.split])]
sq = {s['question'] for s in shots}; rows = [r for r in rows if r['question'] not in sq]
order = sorted(range(len(rows)), key=lambda i: len(rows[i]['question']))[:a.n]
C = dict(positions=0, written=0, quiet=0, quiet_mismatch=0, rows=0, rows_fired=0, hn_agree=0, hn_fp=0, hn_fn=0, fires=0, cap_fires=0)


@torch.no_grad()
def fwd(x, att, past, active):
    rc.active = active
    o = model.model(input_ids=x, attention_mask=att, past_key_values=past, use_cache=True)
    return model.lm_head(o.last_hidden_state[:, -1]).float(), o.past_key_values


for b in range(0, len(order), a.bs):
    seqs = [prompt + T(f"Question: {rows[i]['question']}\nAnswer:") for i in order[b:b + a.bs]]
    L = max(len(s) for s in seqs); B = len(seqs)
    x = torch.tensor([[eos] * (L - len(s)) + s for s in seqs], device=dev)
    att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=dev)
    if net is not None: net.prime(x)
    lA, pA = fwd(x, att, None, True); trc = rc.last_trace
    lB, pB = fwd(x, att, None, False)
    gen = [[] for _ in range(B)]; done = [False] * B; fired = torch.zeros(B, dtype=torch.bool, device=dev)
    for step in range(a.max_new):
        live = torch.tensor([not d for d in done], device=dev)
        wr = trc['wrote'][:, -1] & live
        eq = torch.tensor([torch.equal(lA[i], lB[i]) for i in range(B)], device=dev)
        q = live & ~wr
        C['positions'] += int(live.sum()); C['written'] += int(wr.sum()); C['quiet'] += int(q.sum())
        C['quiet_mismatch'] += int((q & ~eq).sum())
        true_hn = is_hash[lB.argmax(-1)]; circ = trc['hn'][:, -1] > 0.5
        C['hn_agree'] += int((live & (circ == true_hn)).sum()); C['hn_fp'] += int((live & circ & ~true_hn).sum()); C['hn_fn'] += int((live & ~circ & true_hn).sum())
        fi = (trc['fire'][:, -1] > 0.5) & live; C['fires'] += int(fi.sum()); fired |= fi
        if a.intervene in ('hash', 'hybrid') and a.cap:
            C['cap_fires'] += int(((trc['frow'][:, -1, [rc.r_cap_nl, rc.r_cap]].sum(-1) > 0.5) & live).sum())
        nxt = lA.argmax(-1); nl = nxt.tolist()
        for i in range(B):
            if done[i]: continue
            gen[i].append(nl[i])
            if nl[i] == eos or (len(gen[i]) % 4 == 0 and '\n\n' in tok.decode(gen[i][-8:])): done[i] = True
        if all(done): break
        nxt = torch.where(torch.tensor(done, device=dev), torch.full_like(nxt, eos), nxt)
        att = torch.cat([att, att.new_ones(B, 1)], 1)
        if net is not None: net.step(nxt)
        lA, pA = fwd(nxt[:, None], att, pA, True); trc = rc.last_trace
        lB, pB = fwd(nxt[:, None], att, pB, False)
    C['rows'] += B; C['rows_fired'] += int(fired.sum())
    print(b, C, flush=True)
rep = dict(vars(a), host_weights_unchanged=(w0 == w1), counts=C, hn_agreement=C['hn_agree'] / max(1, C['positions']))
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(rep, open(a.out, 'w'), indent=1)
print(json.dumps(rep))
assert rep['host_weights_unchanged'] and C['quiet_mismatch'] == 0
print('PASS')
