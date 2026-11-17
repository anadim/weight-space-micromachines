"""Checks for the compiled rethink implant (micromachines/rethink_circuit.py), on held-out problems.

Two copies of the same bf16 generation run in lockstep (same batch, same padding, same tokens): copy A with the implant
active (it chooses the tokens), copy B with both implant hooks bypassed and fed A's tokens. The implant writes only into
the last hidden state before the final norm, so the host's KV caches are identical in A and B, and:
  * silent: wherever the implant's write is zero, A's and B's logit rows must be torch.equal; host weights hash unchanged;
  * detector agreement: at every generated position, the circuit's end bit vs the exact rule computed from B's full
    logits (argmax over the whole vocabulary is an end token: contains '\\n\\n', <|endoftext|>, or starts with '\\n' after
    a token that ends with '\\n'); reported over all positions and over the positions that matter (before the implant
    has fired in that answer, gate open);
  * forced row: at each fire, the circuit's forced sequence vs the one rethink.py would pick from the true argmax.
Stop rule = the generic one (blank line in the last 8 tokens checked every 4, <|endoftext|>), max --max_new tokens.

python scripts/check_rethink_circuit.py [--mode always|gated] [--split test|hard] [--n 256] [--bs 32] [--out ...]
"""
import os, sys, json, hashlib, argparse, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from transformers import AutoTokenizer, AutoModelForCausalLM
from micromachines.calcfmt import strip_solution
from micromachines.rethink_circuit import RethinkCircuit

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
ap.add_argument('--mode', default='always', choices=['gated', 'always'])
ap.add_argument('--split', default='test', choices=['test', 'hard'])
ap.add_argument('--probe', default='results/rethink/rethink_probe_L28.pt')
ap.add_argument('--shortlist', default='', help='optional competitor shortlist (default: all head rows)')
ap.add_argument('--dirs', default='results/rethink_circuit/write_dirs.pt')
ap.add_argument('--n', type=int, default=256)
ap.add_argument('--skip', type=int, default=0, help='skip the first N problems of the length order')
ap.add_argument('--bs', type=int, default=32)
ap.add_argument('--max_new', type=int, default=700)
ap.add_argument('--out', default='results/rethink_circuit/check.json')
a = ap.parse_args()
dev = 'cuda'
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev).eval()
digest = lambda: hashlib.sha256(b''.join(p.detach().float().cpu().numpy().tobytes() for p in model.state_dict().values())).hexdigest()
w0 = digest()
rc = RethinkCircuit(model, tok, mode=a.mode, probe=a.probe if a.mode == 'gated' else None,
                    shortlist=torch.load(a.shortlist)['ids'] if a.shortlist else None, dirs=a.dirs, trace=True)
w1 = digest()
eos = tok.convert_tokens_to_ids('<|endoftext|>'); T = lambda s: tok(s, add_special_tokens=False)['input_ids']
V = model.lm_head.weight.shape[0]
txt = [tok.decode([i]) for i in range(len(tok))] + [''] * (V - len(tok))
end_any = torch.tensor([('\n\n' in t) or i == eos for i, t in enumerate(txt)], device=dev)
end_nl = torch.tensor([t.startswith('\n') for t in txt], device=dev) | end_any
shots = json.load(open('data/gsm8k/shots_v1.json')); prompt = []
for s in shots:
    body, fin = strip_solution(s['solution'] + f"\n#### {s['answer']}"); prompt += T(f"Question: {s['question']}\nAnswer: {body}\n#### {fin}\n\n")
rows = [json.loads(l) for l in open('data/gsm8k/test.jsonl' if a.split == 'test' else 'data/gsmhard/test.jsonl')]
sq = {s['question'] for s in shots}; rows = [r for r in rows if r['question'] not in sq]
order = sorted(range(len(rows)), key=lambda i: len(rows[i]['question']))[a.skip:a.skip + a.n]
C = dict(positions=0, written=0, quiet=0, quiet_mismatch=0, rows=0, rows_fired=0, rows_silent_all_identical=0,
         det_all=0, det_agree=0, det_fp=0, det_fn=0, det_pre=0, det_pre_agree=0, det_pre_fp=0, det_pre_fn=0,
         fires=0, fire_row_match=0, fire_true_end=0, probe_evals=0, probe_pass=0, prefill_written=0)
seq_index = {tuple(s): k for k, s in enumerate(rc.seqs)}


def ref_row(choice, prev_nl):                 # the forced sequence rethink.py would write for this end token
    t = txt[choice]
    stem = '' if choice == eos else (t.split('\n\n')[0] if '\n\n' in t else None)
    PH = "Hmm, wait. Let me rethink this.\n"
    if not prev_nl:
        s = None if stem is None else stem.rstrip() + '\n' + PH
    elif t.startswith('\n') or (stem is not None and stem.strip() == ''):
        s = PH if (stem is None or stem.strip() == '') else stem.rstrip() + '\n' + PH
    else:
        s = None if stem is None else stem.rstrip() + '\n' + PH
    return None if s is None else seq_index.get(tuple(T(s)))


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
    lA, pA = fwd(x, att, None, True); trc = rc.last_trace
    C['prefill_written'] += int(trc['wrote'].sum())          # writes at prompt positions (earlier shots' answers)
    lB, pB = fwd(x, att, None, False)
    gen = [[] for _ in range(B)]; done = [False] * B
    fired = torch.zeros(B, dtype=torch.bool, device=dev); row_ok = torch.ones(B, dtype=torch.bool, device=dev)
    for step in range(a.max_new):
        live = torch.tensor([not d for d in done], device=dev)
        wr = trc['wrote'][:, -1] & live
        eq = torch.tensor([torch.equal(lA[i], lB[i]) for i in range(B)], device=dev)
        q = live & ~wr
        C['positions'] += int(live.sum()); C['written'] += int(wr.sum()); C['quiet'] += int(q.sum())
        C['quiet_mismatch'] += int((q & ~eq).sum()); row_ok &= ~(q & ~eq)
        # detector agreement (B's logits = unmodified host)
        choice = lB.argmax(-1); prev_nl = trc['endsnl'][:, -1] > 0.5
        true_end = torch.where(prev_nl, end_nl[choice], end_any[choice])
        circ = trc['end'][:, -1] > 0.5
        C['det_all'] += int(live.sum()); C['det_agree'] += int((live & (circ == true_end)).sum())
        C['det_fp'] += int((live & circ & ~true_end).sum()); C['det_fn'] += int((live & ~circ & true_end).sum())
        pre = live & ~fired & (trc['gate'][:, -1] > 0.5)
        C['det_pre'] += int(pre.sum()); C['det_pre_agree'] += int((pre & (circ == true_end)).sum())
        C['det_pre_fp'] += int((pre & circ & ~true_end).sum()); C['det_pre_fn'] += int((pre & ~circ & true_end).sum())
        fi = (trc['fire'][:, -1] > 0.5) & live
        for i in fi.nonzero().flatten().tolist():
            C['fires'] += 1; C['fire_true_end'] += int(true_end[i])
            r = ref_row(int(choice[i]), bool(prev_nl[i]))
            C['fire_row_match'] += int(r is not None and wr[i] and int(lA[i].argmax()) == rc.seqs[r][0])
        fired |= fi
        if a.mode == 'gated':
            pe = (trc['fh'][:, -1] > 0.5) & live
            C['probe_evals'] += int(pe.sum()); C['probe_pass'] += int((pe & (trc['gate'][:, -1] > 0.5)).sum())
        nxt = lA.argmax(-1); nl = nxt.tolist()
        for i in range(B):
            if done[i]: continue
            gen[i].append(nl[i])
            if nl[i] == eos or (len(gen[i]) % 4 == 0 and '\n\n' in tok.decode(gen[i][-8:])): done[i] = True
        if all(done): break
        nxt = torch.where(torch.tensor(done, device=dev), torch.full_like(nxt, eos), nxt)
        att = torch.cat([att, att.new_ones(B, 1)], 1)
        lA, pA = fwd(nxt[:, None], att, pA, True); trc = rc.last_trace
        lB, pB = fwd(nxt[:, None], att, pB, False)
    C['rows'] += B; C['rows_fired'] += int(fired.sum()); C['rows_silent_all_identical'] += int((~fired & row_ok).sum())
    print(b, C, flush=True)
rep = {'model': a.model, 'mode': a.mode, 'split': a.split, 'n_problems': len(order), 'skip': a.skip,
       'host_weights_unchanged': w0 == w1, 'shortlist_size': len(rc.C), 'counts': C,
       'detector_agreement_all': C['det_agree'] / max(1, C['det_all']),
       'detector_agreement_pre_fire': C['det_pre_agree'] / max(1, C['det_pre'])}
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(rep, open(a.out, 'w'), indent=1)
print(json.dumps(rep))
assert rep['host_weights_unchanged'] and C['quiet_mismatch'] == 0
print('PASS')
