"""Silent check for the rethink implant: wherever it does not write, the host's logits are bit-identical to the
unmodified model, including at the '####' positions where the gated implant evaluates its probe (fired or not).

Two copies of the same generation run in lockstep on GSM8K test problems (same batch, same padding, same tokens):
copy A with the implant's hook active (it chooses the tokens), copy B with the hook bypassed and fed A's tokens. Because
the implant writes only into the last hidden state (before the final norm), the host's KV caches are the same in both
copies, so at every step where the implant's write is zero the two logit rows must be torch.equal. Also: the host's
weights hash the same before and after attaching.

python scripts/check_silent_rethink.py [--mode gated|always] [--n 64] [--bs 8] [--probe results/rethink/rethink_probe_L28.pt] [--out results/rethink/silent_check_rethink.json]
"""
import os, sys, json, hashlib, argparse, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from transformers import AutoTokenizer, AutoModelForCausalLM
from micromachines.calcfmt import strip_solution
from micromachines.rethink import RethinkImplant

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
ap.add_argument('--mode', default='gated', choices=['gated', 'always'])
ap.add_argument('--probe', default='results/rethink/rethink_probe_L28.pt')
ap.add_argument('--n', type=int, default=64)
ap.add_argument('--bs', type=int, default=8)
ap.add_argument('--out', default='results/rethink/silent_check_rethink.json')
a = ap.parse_args()
dev = 'cuda'
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev).eval()
digest = lambda: hashlib.sha256(b''.join(p.detach().float().cpu().numpy().tobytes() for p in model.state_dict().values())).hexdigest()
w0 = digest()
rt = RethinkImplant(model, tok, mode=a.mode, probe=a.probe if a.mode == 'gated' else None)
w1 = digest()
eos = tok.convert_tokens_to_ids('<|endoftext|>'); T = lambda s: tok(s, add_special_tokens=False)['input_ids']
shots = json.load(open('data/gsm8k/shots_v1.json')); prompt = []
for s in shots:
    body, fin = strip_solution(s['solution'] + f"\n#### {s['answer']}"); prompt += T(f"Question: {s['question']}\nAnswer: {body}\n#### {fin}\n\n")
rows = [json.loads(l) for l in open('data/gsm8k/test.jsonl')]; sq = {s['question'] for s in shots}
rows = [r for r in rows if r['question'] not in sq]
order = sorted(range(len(rows)), key=lambda i: len(rows[i]['question']))[:a.n]
C = dict(positions=0, written=0, quiet=0, quiet_mismatch=0, probe_evals=0, probe_below=0, probe_below_mismatch=0,
         probe_above=0, probe_above_mismatch=0, rows=0, rows_silent=0, rows_silent_all_identical=0)


@torch.no_grad()
def fwd(x, att, past, active):
    rt.active = active
    o = model.model(input_ids=x, attention_mask=att, past_key_values=past, use_cache=True)
    return model.lm_head(o.last_hidden_state[:, -1]).float(), o.past_key_values


for b in range(0, len(order), a.bs):
    seqs = [prompt + T(f"Question: {rows[i]['question']}\nAnswer:") for i in order[b:b + a.bs]]
    L = max(len(s) for s in seqs); B = len(seqs)
    x = torch.tensor([[eos] * (L - len(s)) + s for s in seqs], device=dev)
    att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=dev)
    rt.prime(x)
    lA, pA = fwd(x, att, None, True); lB, pB = fwd(x, att, None, False)
    done = torch.zeros(B, dtype=torch.bool, device=dev); n = 0; any_write = torch.zeros(B, dtype=torch.bool, device=dev)
    row_ok = torch.ones(B, dtype=torch.bool, device=dev)
    for step in range(700):
        live = ~done; wr = rt.wrote & live
        pr = (rt.probed if rt.probed is not None else torch.zeros_like(wr)) & live
        eq = torch.tensor([torch.equal(lA[i], lB[i]) for i in range(B)], device=dev)
        q = live & ~wr
        C['positions'] += int(live.sum()); C['written'] += int(wr.sum()); C['quiet'] += int(q.sum())
        C['quiet_mismatch'] += int((q & ~eq).sum()); row_ok &= ~(q & ~eq); any_write |= wr
        if a.mode == 'gated':
            below = pr & ~rt.state['gate']; above = pr & rt.state['gate']
            C['probe_evals'] += int(pr.sum()); C['probe_below'] += int(below.sum()); C['probe_above'] += int(above.sum())
            C['probe_below_mismatch'] += int((below & ~eq).sum()); C['probe_above_mismatch'] += int((above & ~eq).sum())
        nxt = lA.argmax(-1); n += 1
        done |= (nxt == eos) | torch.tensor(['\n\n' in tok.decode([t]) for t in nxt.tolist()], device=dev)   # the end the host chose
        if bool(done.all()): break
        nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
        att = torch.cat([att, att.new_ones(B, 1)], 1)
        rt.step(nxt)
        lA, pA = fwd(nxt[:, None], att, pA, True); lB, pB = fwd(nxt[:, None], att, pB, False)
    C['rows'] += B; C['rows_silent'] += int((~any_write).sum()); C['rows_silent_all_identical'] += int((~any_write & row_ok).sum())
    print(b, C, flush=True)
rep = {'model': a.model, 'mode': a.mode, 'n_problems': len(order), 'host_weights_unchanged': w0 == w1, 'counts': C}
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(rep, open(a.out, 'w'), indent=1)
print(json.dumps(rep))
assert rep['host_weights_unchanged'] and C['quiet_mismatch'] == 0
print('PASS')
