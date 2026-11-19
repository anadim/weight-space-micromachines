"""v2 "when NOT to call" rows = self-distillation of the FROZEN BASE on general prompts that need no tool (nocall_prompts.py).
50 rows zero-shot "Q: ...\\nA: Let's think step by step.", 50 rows behind a 3-shot prefix of our own exemplars (exemplar answers =
the base's own zero-shot answers). Completion = the base's greedy continuation of exactly that prompt (stop at '\\n\\nQ' /
<|endoftext|>, max 200 tokens, cut back to the last full sentence if truncated). Runs on a pod GPU.
python make_nocall.py OUT.jsonl"""
import sys, os, json, random, re, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from transformers import AutoTokenizer, AutoModelForCausalLM
from nocall_prompts import EXEMPLARS, PROMPTS
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', dtype=torch.bfloat16).cuda().eval()
eos = tok.convert_tokens_to_ids('<|endoftext|>')
QA = lambda q: f"Q: {q}\nA: Let's think step by step."


@torch.no_grad()
def gen(prompts, max_new=200):
    seqs = [tok(p, add_special_tokens=False)['input_ids'] for p in prompts]; L = max(map(len, seqs)); B = len(seqs)
    x = torch.tensor([[eos] * (L - len(s)) + s for s in seqs], device='cuda'); att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device='cuda')
    o = model.model(input_ids=x, attention_mask=att, use_cache=True); past = o.past_key_values; lg = model.lm_head(o.last_hidden_state[:, -1]).float()
    g = [[] for _ in range(B)]; done = [False] * B
    for step in range(max_new):
        if step < 4: lg[:, eos] = -1e9
        nxt = lg.argmax(-1); nl = nxt.tolist()
        for i in range(B):
            if done[i]: continue
            g[i].append(nl[i])
            if nl[i] == eos or '\n\nQ' in tok.decode(g[i][-6:]): done[i] = True
        if all(done): break
        nxt = torch.where(torch.tensor(done, device='cuda'), torch.full_like(nxt, eos), nxt); att = torch.cat([att, att.new_ones(B, 1)], 1)
        o = model.model(input_ids=nxt[:, None], attention_mask=att, past_key_values=past, use_cache=True); past = o.past_key_values
        lg = model.lm_head(o.last_hidden_state[:, -1]).float()
    out = []
    for s, d in zip(g, done):
        t = tok.decode(s).replace('<|endoftext|>', '').split('\n\nQ')[0].rstrip()
        if not d:                                            # truncated: cut back to the last full sentence
            m = list(re.finditer(r'[.!?](\s|$)', t)); t = t[:m[-1].end()].rstrip() if m else t
        out.append(t)
    return out


ex = gen([QA(q) for q in EXEMPLARS])
r = random.Random('nocall-v2'); prompts = []
for i, q in enumerate(PROMPTS):
    if i % 2 == 0: prompts.append(QA(q))
    else:
        sel = r.sample(range(len(EXEMPLARS)), 3)
        prompts.append('\n\n'.join(QA(EXEMPLARS[j]) + ex[j] for j in sel) + '\n\n' + QA(q))
outs = []
for b in range(0, len(prompts), 25): outs += gen(prompts[b:b + 25])
with open(sys.argv[1], 'w') as f:
    for p, c in zip(prompts, outs):
        assert '<tool_call>' not in c
        f.write(json.dumps({'prompt': p, 'completion': c, 'team': 'nocall', 'kind': 'nocall/' + ('3shot' if '\n\nQ:' in p else '0shot')}) + '\n')
print(len(outs), 'rows; mean completion chars', sum(map(len, outs)) / len(outs))
for c in outs[:3]: print(repr(c[:300]))
