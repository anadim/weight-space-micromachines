"""Evaluate a host (frozen base or our fine-tune), with or without the Team-A tools, on OUR generated questions, using the
OFFICIAL BBH 3-shot CoT prompt of the matching task, unchanged (cot/*.txt). Same decoding as the repo's scripts/run_bbh.py:
HF loop, greedy, <|endoftext|> blocked for the first 4 tokens, stop at <|endoftext|> or '\\n\\nQ', same answer extraction.

python run_eval.py DATA.jsonl OUT.json [--model PATH] [--tools none|track_gen,order_gen] [--bs 16] [--limit N] [--zeroshot]
"""
import os, sys, re, json, time, argparse, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import register  # noqa: F401
from micromachines.machine import attach
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument('data'); ap.add_argument('out')
ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
ap.add_argument('--tools', default='none')
ap.add_argument('--bs', type=int, default=16)
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--max_new', type=int, default=600)
ap.add_argument('--min_tokens', type=int, default=4)
ap.add_argument('--zeroshot', action='store_true')
a = ap.parse_args()
dev = 'cuda'
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
net = None
if a.tools != 'none':
    net = attach(model, tok, machines=tuple(a.tools.split(',')))
    print('attached', a.tools, 'min write margin %.1f' % min(m for m, _ in net.margins.values()), flush=True)
eos = tok.convert_tokens_to_ids('<|endoftext|>')
PROMPT = {'track': 'tracking_shuffled_objects_three_objects', 'order': 'logical_deduction_three_objects'}   # the 3 files per task are identical
pre = lambda tool: '' if a.zeroshot else open(f'{HERE}/cot/{PROMPT[tool]}.txt').read().split('-----', 1)[1].strip() + '\n\n'


def extract(text):
    m = re.findall(r'[Tt]he answer is\s*(.*?)(?:\.\s*$|\.\n|\n|$)', text.split('\n\nQ:')[0]); return m[-1].strip() if m else ''


def norm(x):
    x = x.strip().strip('.').strip()
    if re.fullmatch(r'\(?[A-R]\)?(\s.*)?', x): return '(' + re.match(r'\(?([A-R])', x).group(1) + ')'
    return x.lower()


@torch.no_grad()
def generate(prompts):
    seqs = [tok(p, add_special_tokens=False)['input_ids'] for p in prompts]; L = max(map(len, seqs)); B_ = len(seqs)
    x = torch.tensor([[eos] * (L - len(s)) + s for s in seqs], device=dev); att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=dev)
    if net is not None: net.prime(x)
    o = model.model(input_ids=x, attention_mask=att, use_cache=True); past = o.past_key_values; lg = model.lm_head(o.last_hidden_state[:, -1]).float()
    gen = [[] for _ in range(B_)]; done = [False] * B_
    for step in range(a.max_new):
        if step < a.min_tokens: lg[:, eos] = -1e9
        nxt = lg.argmax(-1); nl = nxt.tolist()
        for i in range(B_):
            if done[i]: continue
            gen[i].append(nl[i])
            if nl[i] == eos or (len(gen[i]) % 6 == 0 and '\n\nQ' in tok.decode(gen[i][-12:])): done[i] = True
        if all(done): break
        nxt = torch.where(torch.tensor(done, device=dev), torch.full_like(nxt, eos), nxt); att = torch.cat([att, att.new_ones(B_, 1)], 1)
        if net is not None: net.step(nxt)
        o = model.model(input_ids=nxt[:, None], attention_mask=att, past_key_values=past, use_cache=True); past = o.past_key_values
        lg = model.lm_head(o.last_hidden_state[:, -1]).float()
    return gen


rows = [json.loads(l) for l in open(a.data)][:a.limit or None]
t0 = time.time(); out = []
for b in range(0, len(rows), a.bs):
    ch = rows[b:b + a.bs]
    for e, g in zip(ch, generate([pre(e['tool']) + f"Q: {e['input']}\nA: Let's think step by step." for e in ch])):
        txt = tok.decode(g).replace('<|endoftext|>', ''); p = extract(txt)
        out.append({'id': e['id'], 'task': e['task'], 'gold': e['target'], 'pred': p, 'ok': norm(p) == norm(e['target']), 'out': txt,
                    'gold_call': e.get('gold_call'), 'style': e.get('style'), 'kind': e.get('kind'), 'setting': e.get('setting')})
    print(f'{len(out)}/{len(rows)} {time.time() - t0:.0f}s', flush=True)
per = {}
for x in out: per.setdefault(x['task'], []).append(x['ok'])
acc = {t: 100 * sum(v) / len(v) for t, v in per.items()}
for t in acc: print(f'{t:10s} {acc[t]:5.1f} (n={len(per[t])})')
json.dump({'model': a.model, 'tools': a.tools, 'data': a.data, 'zeroshot': a.zeroshot, 'per_task': acc, 'rows': out}, open(a.out, 'w'), indent=1)
