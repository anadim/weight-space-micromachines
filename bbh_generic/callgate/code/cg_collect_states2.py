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
from cg_neg_prompts import EXTRA_NEG, PUZZLE_NEG
from nocall_prompts import EXEMPLARS, PROMPTS

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

items = []
byt = {}
for l in open(f'{G}/teamA/data/test.jsonl'):
    e = json.loads(l); byt.setdefault(e['task'], []).append(e)
for task, es in byt.items():
    for e in es[:50]:
        pre = P_track if e['tool'] == 'track' else P_order
        items.append((pre + QA(e['input']), 1, {'src': f'A/{task}', 'qid': 'A_' + e['id'], 'split': split_of('A_' + e['id'])}))
for f, pre in (('temporal_test', P_temporal), ('navigate_test', P_navigate), ('lies_test', P_lies)):
    task = f.split('_')[0]
    for i, e in enumerate([json.loads(l) for l in open(f'{G}/teamB/data/{f}.jsonl')][:60]):
        qid = f'B_{task}_{i}'; items.append((pre + QA(e['question']), 1, {'src': f'B/{task}', 'qid': qid, 'split': split_of(qid)}))
GEN_NEG = list(dict.fromkeys(EXEMPLARS + PROMPTS + EXTRA_NEG))        # general/chat/distractor
PUZ = list(dict.fromkeys(PUZZLE_NEG))                                 # puzzle/MC/word-problem (BBH-like format, no tool)
rng = random.Random('callgate-neg')
for grp, lst in (('gen', GEN_NEG), ('puz', PUZ)):
    for i, q in enumerate(lst):
        qid = f'N_{grp}_{i}'; sp = split_of(qid)
        items.append((QA(q), 0, {'src': f'neg/{grp}/0shot', 'qid': qid, 'split': sp}))
        for pfx in rng.sample(PREFIXES, 2):
            items.append((pfx + QA(q), 0, {'src': f'neg/{grp}/pref', 'qid': qid, 'split': sp}))
print(f'{len(items)} items (pos={sum(1 for _,l,_ in items if l==1)}, neg={sum(1 for _,l,_ in items if l==0)})', flush=True)


@torch.no_grad()
def capture(batch, max_new=120):
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


items.sort(key=lambda it: len(it[0])); BS = 32; recs = []; t0 = time.time()
for b in range(0, len(items), BS):
    ch = items[b:b + BS]
    for (prompt, label, meta), (h, op) in zip(ch, capture(ch)):
        recs.append({'label': label, 'opened': op, 'h': h, **meta})
    if (b // BS) % 5 == 0: print(f'{b+len(ch)}/{len(items)}  {time.time()-t0:.0f}s', flush=True)
torch.save(recs, sys.argv[1])
print(f'saved {len(recs)}; opened {sum(r["opened"] for r in recs)}')
for lab in (1, 0):
    for sp in ('train', 'test'):
        g = [r for r in recs if r['label'] == lab and r['split'] == sp]
        print(f'  label={lab} split={sp}: n={len(g)} opened={sum(r["opened"] for r in g)}')
