"""Collect host decision-point hidden states for the call gate.
Plain ft_int_v2 host (NO tools attached: the attach writes are exactly 0 before any tool-response region, so the
post-norm hidden state at the FIRST '<tool_call>' step is bit-identical with or without tools). For each question we
greedily decode and record the host's post-final-norm last hidden state h at the first step where argmax == '<tool_call>'
(token 151657) -- the exact moment the gate must decide. Label 1 = a tool fits (Team A/B own questions), 0 = no tool
(our own general/distractor questions, shown zero-shot and behind tool-task CoT prefixes to induce spurious calls).
Train only on OWN data (A_test/B_test synthetic sets + our negatives); never BBH; A_oos2/B_oos/C_ood untouched.

python cg_collect_states.py OUT.pt
"""
import sys, os, json, random, hashlib, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _cgpaths import G, CKPT, RES, CODE, INTEG_DATA
sys.path.insert(0, CODE)
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cg_neg_prompts import EXTRA_NEG
from nocall_prompts import EXEMPLARS, PROMPTS                      # integration team's own 100+9 no-tool prompts

CKPT = CKPT
dev = 'cuda'
tok = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForCausalLM.from_pretrained(CKPT, dtype=torch.bfloat16).to(dev).eval()
TC = tok.convert_tokens_to_ids('<tool_call>'); eos = tok.convert_tokens_to_ids('<|endoftext|>')
assert TC == 151657, TC
cot = lambda p: open(p).read().split('-----', 1)[1].strip() + '\n\n'
QA = lambda q: f"Q: {q}\nA: Let's think step by step."

P_track = cot(f'{G}/teamA/cot/tracking_shuffled_objects_three_objects.txt')
P_order = cot(f'{G}/teamA/cot/logical_deduction_three_objects.txt')
P_temporal = cot(f'{G}/teamB/cot/temporal_sequences.txt')
P_navigate = cot(f'{G}/teamB/cot/navigate.txt')
P_lies = cot(f'{G}/teamB/cot/web_of_lies.txt')
PREFIXES = [P_track, P_order, P_temporal, P_navigate, P_lies]


def split_of(key):                                                # deterministic 50/50 by question identity
    return 'test' if int(hashlib.md5(key.encode()).hexdigest(), 16) % 2 else 'train'


items = []   # (prompt, label, meta)
# -------- POSITIVES: Team A (track/order) + Team B (interval/walk/truth) own synthetic test questions --------
PER_A, PER_B = 50, 60
byt = {}
for l in open(f'{G}/teamA/data/test.jsonl'):
    e = json.loads(l); byt.setdefault(e['task'], []).append(e)
for task, es in byt.items():
    for e in es[:PER_A]:
        pre = P_track if e['tool'] == 'track' else P_order
        items.append((pre + QA(e['input']), 1, {'src': f'A/{task}', 'qid': 'A_' + e['id'], 'split': split_of('A_' + e['id']), 'pref': True}))
BN = {'temporal': 'temporal_sequences', 'navigate': 'navigate', 'lies': 'web_of_lies'}
for f, pre in (('temporal_test', P_temporal), ('navigate_test', P_navigate), ('lies_test', P_lies)):
    task = f.split('_')[0]
    rows = [json.loads(l) for l in open(f'{G}/teamB/data/{f}.jsonl')][:PER_B]
    for i, e in enumerate(rows):
        qid = f'B_{task}_{i}'
        items.append((pre + QA(e['question']), 1, {'src': f'B/{task}', 'qid': qid, 'split': split_of(qid), 'pref': True}))

# -------- NEGATIVES: our own no-tool questions, zero-shot + behind 2 random tool-task CoT prefixes --------
NEG = list(dict.fromkeys(EXEMPLARS + PROMPTS + EXTRA_NEG))         # dedup, keep order
rng = random.Random('callgate-neg')
for i, q in enumerate(NEG):
    qid = f'N_{i}'; sp = split_of(qid)
    items.append((QA(q), 0, {'src': 'neg/0shot', 'qid': qid, 'split': sp, 'pref': False}))
    for pfx in rng.sample(PREFIXES, 2):
        items.append((pfx + QA(q), 0, {'src': 'neg/pref', 'qid': qid, 'split': sp, 'pref': True}))

print(f'{len(items)} items (pos={sum(1 for _,l,_ in items if l==1)}, neg={sum(1 for _,l,_ in items if l==0)})', flush=True)


@torch.no_grad()
def capture(batch, max_new=120):
    """Return list of (h_fp32 or None, opened_bool) aligned with batch. h = post-norm last hidden at first '<tool_call>'."""
    prompts = [b[0] for b in batch]
    seqs = [tok(p, add_special_tokens=False)['input_ids'] for p in prompts]; L = max(map(len, seqs)); B = len(seqs)
    x = torch.tensor([[eos] * (L - len(s)) + s for s in seqs], device=dev)
    att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=dev)
    o = model.model(input_ids=x, attention_mask=att, use_cache=True); past = o.past_key_values
    h = o.last_hidden_state[:, -1]; lg = model.lm_head(h).float()
    H = [None] * B; opened = [False] * B; done = [False] * B
    for step in range(max_new):
        if step < 4: lg[:, eos] = -1e9
        nxt = lg.argmax(-1)
        for i in range(B):
            if done[i]: continue
            if int(nxt[i]) == TC:
                H[i] = h[i].float().cpu().clone(); opened[i] = True; done[i] = True
            elif int(nxt[i]) == eos:
                done[i] = True
        if all(done): break
        nxt = torch.where(torch.tensor(done, device=dev), torch.full_like(nxt, eos), nxt)
        att = torch.cat([att, att.new_ones(B, 1)], 1)
        o = model.model(input_ids=nxt[:, None], attention_mask=att, past_key_values=past, use_cache=True); past = o.past_key_values
        h = o.last_hidden_state[:, -1]; lg = model.lm_head(h).float()
    return list(zip(H, opened))


# group by prompt length bucket for efficient batching
items.sort(key=lambda it: len(it[0]))
BS = 48
recs = []
import time; t0 = time.time()
for b in range(0, len(items), BS):
    ch = items[b:b + BS]
    for (prompt, label, meta), (h, op) in zip(ch, capture(ch)):
        recs.append({'label': label, 'opened': op, 'h': h, **meta})
    if (b // BS) % 5 == 0:
        print(f'{b+len(ch)}/{len(items)}  {time.time()-t0:.0f}s', flush=True)

torch.save(recs, sys.argv[1])
no = sum(1 for r in recs if r['opened'])
print(f'saved {len(recs)} recs; opened {no}')
for lab in (1, 0):
    for sp in ('train', 'test'):
        g = [r for r in recs if r['label'] == lab and r['split'] == sp]
        op = sum(r['opened'] for g_ in [g] for r in g_)
        print(f'  label={lab} split={sp}: n={len(g)} opened={op}')
