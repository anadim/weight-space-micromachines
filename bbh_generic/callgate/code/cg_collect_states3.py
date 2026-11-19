"""Collect decision-point hidden states WITH v2 tools attached (exact test-time trajectory: Team C passive may fire at
'=' in the host's pre-opener reasoning). Same items/labels/splits as cg_collect_states.py. h = post-norm last hidden at
the first '<tool_call>' step (pending=0 at the opener itself, but the trajectory matches test time).
python cg_collect_states2.py OUT.pt
"""
import sys, os, json, random, hashlib, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _cgpaths import G, CKPT, RES, CODE, INTEG_DATA
sys.path.insert(0, CODE); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer, AutoModelForCausalLM
import integ

CKPT = CKPT; dev = 'cuda'
tok = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForCausalLM.from_pretrained(CKPT, dtype=torch.bfloat16).to(dev).eval()
net = integ.attach_all(model, tok, integ.TOOLS_V2, veto=integ.VETO)
TC = tok.convert_tokens_to_ids('<tool_call>'); eos = tok.convert_tokens_to_ids('<|endoftext|>')
cot = lambda p: open(p).read().split('-----', 1)[1].strip() + '\n\n'
QA = lambda q: f"Q: {q}\nA: Let's think step by step."
P_track = cot(f'{G}/teamA/cot/tracking_shuffled_objects_three_objects.txt')
P_order = cot(f'{G}/teamA/cot/logical_deduction_three_objects.txt')
P_temporal = cot(f'{G}/teamB/cot/temporal_sequences.txt'); P_navigate = cot(f'{G}/teamB/cot/navigate.txt'); P_lies = cot(f'{G}/teamB/cot/web_of_lies.txt')
PREFIXES = [P_track, P_order, P_temporal, P_navigate, P_lies]
split_of = lambda k: 'test' if int(hashlib.md5(k.encode()).hexdigest(), 16) % 2 else 'train'

items = []   # (prompt, label, meta, max_new)
from cg_gen_puz import GENS
byt = {}
for l in open(f'{G}/teamA/data/test.jsonl'):
    e = json.loads(l); byt.setdefault(e['task'], []).append(e)
for task in ('track5', 'track7', 'order5', 'order7'):           # long-preamble positives missed at max_new 120
    for e in byt[task][:50]:
        pre = P_track if e['tool'] == 'track' else P_order
        items.append((pre + QA(e['input']), 1, {'src': f'A/{task}', 'qid': 'A_' + e['id'], 'split': split_of('A_' + e['id'])}, 450))
rng = random.Random('callgate-puz3')
for g in GENS:
    r = random.Random('gen-' + g.__name__)
    qs = []
    while len(qs) < 34:
        x = g(r)
        if x[0] not in [q[0] for q in qs]: qs.append(x)
    shots, qs = qs[:4], qs[4:]                                  # 4 exemplars per genre are only used in self-prefixes
    for i, (q, ans, why) in enumerate(qs):
        qid = f'N_p3_{g.__name__}_{i}'; sp = split_of(qid)
        sel = rng.sample(shots, 3)
        selfpre = '\n\n'.join(QA(s[0]) + f"\n{s[2]}. So the answer is {s[1]}." for s in sel) + '\n\n'
        items.append((QA(q), 0, {'src': 'neg/p3/0shot', 'qid': qid, 'split': sp, 'gen': g.__name__}, 160))
        items.append((rng.choice(PREFIXES) + QA(q), 0, {'src': 'neg/p3/pref', 'qid': qid, 'split': sp, 'gen': g.__name__}, 160))
        items.append((selfpre + QA(q), 0, {'src': 'neg/p3/self', 'qid': qid, 'split': sp, 'gen': g.__name__}, 160))
print(f'{len(items)} items (pos={sum(1 for x in items if x[1]==1)}, neg={sum(1 for x in items if x[1]==0)})', flush=True)


@torch.no_grad()
def capture(batch, max_new):
    prompts = [b[0] for b in batch]
    seqs = [tok(p, add_special_tokens=False)['input_ids'] for p in prompts]; L = max(map(len, seqs)); B = len(seqs)
    x = torch.tensor([[eos] * (L - len(s)) + s for s in seqs], device=dev)
    att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=dev)
    net.prime(x); gs = integ.GraphStep(net, B)
    o = model.model(input_ids=x, attention_mask=att, use_cache=True); past = o.past_key_values
    h = o.last_hidden_state[:, -1]; lg = model.lm_head(h).float()
    H = [None] * B; opened = [False] * B; done = [False] * B
    for step in range(max_new):
        if step < 4: lg[:, eos] = -1e9
        nxt = lg.argmax(-1)
        for i in range(B):
            if done[i]: continue
            if int(nxt[i]) == TC: H[i] = h[i].float().cpu().clone(); opened[i] = True; done[i] = True
            elif int(nxt[i]) == eos: done[i] = True
        if all(done): break
        nxt = torch.where(torch.tensor(done, device=dev), torch.full_like(nxt, eos), nxt)
        att = torch.cat([att, att.new_ones(B, 1)], 1); gs.step(nxt)
        o = model.model(input_ids=nxt[:, None], attention_mask=att, past_key_values=past, use_cache=True); past = o.past_key_values
        h = o.last_hidden_state[:, -1]; lg = model.lm_head(h).float()
    return list(zip(H, opened))


items.sort(key=lambda it: (it[3], len(it[0]))); BS = 32; recs = []; t0 = time.time()
for b in range(0, len(items), BS):
    ch = items[b:b + BS]
    for (prompt, label, meta, mx), (h, op) in zip(ch, capture(ch, max(x[3] for x in ch))):
        recs.append({'label': label, 'opened': op, 'h': h, **meta})
    if (b // BS) % 5 == 0: print(f'{b+len(ch)}/{len(items)}  {time.time()-t0:.0f}s', flush=True)
torch.save(recs, sys.argv[1])
print(f'saved {len(recs)}; opened {sum(r["opened"] for r in recs)}')
import collections
c = collections.defaultdict(lambda: [0, 0])
for r in recs: c[r['src']][0] += 1; c[r['src']][1] += r['opened']
for k, v in sorted(c.items()): print(f'  {k:16s} n={v[0]} opened={v[1]}')
