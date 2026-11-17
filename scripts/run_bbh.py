"""BIG-Bench Hard with the OFFICIAL 3-shot chain-of-thought prompts, a frozen base model, greedy decoding, zero training;
optionally with micromachine implants attached. This is the harness behind every BBH number in results/bbh/.

python scripts/run_bbh.py OUT.json --tasks navigate,web_of_lies [--model Qwen/Qwen3-0.6B-Base] [--machines all|none|a,b,...]
       [--veto runs/veto_qwen3.pt] [--limit N] [--bs 16] [--min_tokens 4] [--bbh data/bbh]

--machines all   : the nine BBH implants (micromachines.BBH_MACHINES), all attached at once (default)
--machines none  : the same harness with no implant (the same-harness baseline)
--min_tokens 4   : <|endoftext|> is blocked for the first 4 tokens (Qwen3 base otherwise sometimes ends at once);
                   the same rule the vLLM baselines used
Batches are left-padded; with bf16 on GPU the exact outputs depend on the batch composition, so compare runs with the
same --bs and a --limit that is a multiple of it.
"""
import os, sys, re, json, time, argparse, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from transformers import AutoTokenizer, AutoModelForCausalLM
from micromachines import attach, BBH_MACHINES, VETO_MACHINES

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--tasks', required=True)
ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
ap.add_argument('--machines', default='all')
ap.add_argument('--veto', default='runs/veto_qwen3.pt')
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--bs', type=int, default=16)
ap.add_argument('--min_tokens', type=int, default=4)
ap.add_argument('--max_new', type=int, default=400)
ap.add_argument('--bbh', default=os.environ.get('BBH', 'data/bbh'))
a = ap.parse_args()
tasks = a.tasks.split(',')
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16 if dev == 'cuda' else torch.float32).to(dev).eval()
machines = BBH_MACHINES if a.machines == 'all' else () if a.machines == 'none' else tuple(a.machines.split(','))
net = None
if machines:
    t0 = time.time()
    net = attach(model, tok, machines=machines, veto=a.veto if any(m in VETO_MACHINES for m in machines) else None)
    print(f'attached {len(machines)} implants: {", ".join(machines)} ({time.time() - t0:.0f}s)', flush=True)
eos = tok.convert_tokens_to_ids('<|endoftext|>')
pre = lambda t: open(f'{a.bbh}/cot/{t}.txt').read().split('-----', 1)[1].strip() + '\n\n'


def extract(text):
    m = re.findall(r'[Tt]he answer is\s*(.*?)(?:\.\s*$|\.\n|\n|$)', text.split('\n\nQ:')[0]); return m[-1].strip() if m else ''


def norm(x, task):
    x = x.strip().strip('.').strip()
    if re.fullmatch(r'\(?[A-R]\)?(\s.*)?', x) and task not in ('word_sorting', 'dyck_languages'): return '(' + re.match(r'\(?([A-R])', x).group(1) + ')'
    if task in ('multistep_arithmetic_two', 'object_counting'): m = re.search(r'-?\d+', x); return m.group() if m else x
    return x.lower()


@torch.no_grad()
def generate(prompts, max_new):
    seqs = [tok(p, add_special_tokens=False)['input_ids'] for p in prompts]; L = max(map(len, seqs)); B_ = len(seqs)
    x = torch.tensor([[eos] * (L - len(s)) + s for s in seqs], device=dev); att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=dev)
    if net is not None: net.prime(x)
    o = model.model(input_ids=x, attention_mask=att, use_cache=True); past = o.past_key_values; lg = model.lm_head(o.last_hidden_state[:, -1]).float()
    gen = [[] for _ in range(B_)]; done = [False] * B_
    for step in range(max_new):
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


t0 = time.time(); per = {}; rows = []
for t in tasks:
    ex = json.load(open(f'{a.bbh}/data/{t}.json'))['examples'][:a.limit or None]
    for b in range(0, len(ex), a.bs):
        ch = ex[b:b + a.bs]
        for e, g in zip(ch, generate([pre(t) + f"Q: {e['input']}\nA: Let's think step by step." for e in ch], a.max_new)):
            txt = tok.decode(g).replace('<|endoftext|>', ''); p = extract(txt); ok = norm(p, t) == norm(e['target'], t)
            per.setdefault(t, []).append(ok); rows.append({'task': t, 'gold': e['target'], 'pred': p, 'ok': ok, 'out': txt})
    print(f'{t:40s} {100 * sum(per[t]) / len(per[t]):5.1f} (n={len(per[t])}) | {time.time() - t0:.0f}s', flush=True)
acc = {t: 100 * sum(v) / len(v) for t, v in per.items()}; print(f'MEAN over {len(acc)} tasks: {sum(acc.values()) / len(acc):.2f}')
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump({'model': a.model, 'machines': list(machines), 'min_tokens': a.min_tokens, 'bs': a.bs, 'per_task': acc, 'rows': rows},
          open(a.out, 'w'), indent=1)
