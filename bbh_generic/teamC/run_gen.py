"""Team C harness: the run_bbh.py generation loop (official 3-shot CoT prompt, greedy, 4-token min, 400 new tokens, same answer
extraction) on OUR generated questions. Never reads BBH task data.

python run_gen.py OUT.json --data data/dev --tasks a,b --machines none|expr_gen,calc_gen,bool_gen [--oe veto|dotnl] [--bs 32]
       [--prompt_file F --qa_fmt '...']   (out-of-style runs: own few-shot prompt instead of a BBH CoT prompt)
"""
import os, sys, re, json, time, argparse, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', '..')); sys.path.insert(0, os.environ.get('WSM_REPO', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--data', default='data/dev'); ap.add_argument('--tasks', required=True)
ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base'); ap.add_argument('--machines', default='none')
ap.add_argument('--veto', default='runs/veto_qwen3.pt'); ap.add_argument('--bs', type=int, default=32)
ap.add_argument('--min_tokens', type=int, default=4); ap.add_argument('--max_new', type=int, default=400)
ap.add_argument('--cot', default='cot'); ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--oe', default='dotnl'); ap.add_argument('--bool_dot', type=int, default=1)
ap.add_argument('--prompt_file', default=''); ap.add_argument('--answer_re', default='')
a = ap.parse_args()
tasks = a.tasks.split(',')
dev = 'cuda'
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev).eval()
machines = () if a.machines == 'none' else tuple(a.machines.split(','))
net = None
if machines:
    import teamc_tools as TT
    t0 = time.time()
    net = TT.attach_gen(model, tok, machines, veto=a.veto, oe=a.oe, bool_dot=a.bool_dot)
    print(f'attached {machines} ({time.time() - t0:.0f}s)', flush=True)
eos = tok.convert_tokens_to_ids('<|endoftext|>')
pre = lambda t: (open(a.prompt_file).read() if a.prompt_file else open(f'{a.cot}/{t}.txt').read().split('-----', 1)[1].strip() + '\n\n')


def extract(text):
    m = re.findall(r'[Tt]he answer is\s*(.*?)(?:\.\s*$|\.\n|\n|$)', text.split('\n\nQ:')[0]); return m[-1].strip() if m else ''


def norm(x, task):
    x = x.strip().strip('.').strip()
    if task in ('multistep_arithmetic_two', 'object_counting') or task.startswith('num'):
        m = re.search(r'-?\d+', x.replace(',', '')); return m.group() if m else x
    return x.lower()


@torch.no_grad()
def generate(prompts, max_new):
    seqs = [tok(p, add_special_tokens=False)['input_ids'] for p in prompts]; L = max(map(len, seqs)); B_ = len(seqs)
    x = torch.tensor([[eos] * (L - len(s)) + s for s in seqs], device=dev); att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=dev)
    if net is not None: net.prime(x)
    o = model.model(input_ids=x, attention_mask=att, use_cache=True); past = o.past_key_values; lg = model.lm_head(o.last_hidden_state[:, -1]).float()
    gen = [[] for _ in range(B_)]; wrote = [[] for _ in range(B_)]; done = [False] * B_
    for step in range(max_new):
        if step < a.min_tokens: lg[:, eos] = -1e9
        nxt = lg.argmax(-1); nl = nxt.tolist()
        if net is not None:                                                  # did an implant write at this position?
            pd = net.pending[:, -1] if net.pending.dim() == 3 else net.pending
            w = (pd.abs().amax(-1) > 0).tolist()
        else: w = [False] * B_
        for i in range(B_):
            if done[i]: continue
            gen[i].append(nl[i]); wrote[i].append(bool(w[i]))
            if nl[i] == eos or (len(gen[i]) % 6 == 0 and '\n\nQ' in tok.decode(gen[i][-12:])): done[i] = True
        if all(done): break
        nxt = torch.where(torch.tensor(done, device=dev), torch.full_like(nxt, eos), nxt); att = torch.cat([att, att.new_ones(B_, 1)], 1)
        if net is not None: net.step(nxt)
        o = model.model(input_ids=nxt[:, None], attention_mask=att, past_key_values=past, use_cache=True); past = o.past_key_values
        lg = model.lm_head(o.last_hidden_state[:, -1]).float()
    return gen, wrote


def marked(g, w):
    """decoded text with implant-written tokens in [brackets]"""
    out = ''
    for t, f in zip(g, w):
        s = tok.decode([t]); out += f'[{s}]' if f else s
    return out


t0 = time.time(); per = {}; rows = []
for t in tasks:
    ex = json.load(open(f'{a.data}/{t}.json'))['examples'][:a.limit or None]
    for b in range(0, len(ex), a.bs):
        ch = ex[b:b + a.bs]
        gens, wrs = generate([pre(t) + f"Q: {e['input']}\nA: Let's think step by step." for e in ch], a.max_new)
        for e, g, wr in zip(ch, gens, wrs):
            txt = tok.decode(g).replace('<|endoftext|>', ''); p = extract(txt); ok = norm(p, t) == norm(e['target'], t)
            per.setdefault(t, []).append(ok)
            rows.append({'task': t, 'input': e['input'], 'gold': e['target'], 'pred': p, 'ok': ok, 'out': txt, 'marked': marked(g, wr), 'n_written': sum(wr)})
    print(f'{t:40s} {100 * sum(per[t]) / len(per[t]):5.1f} (n={len(per[t])}) | {time.time() - t0:.0f}s', flush=True)
acc = {t: 100 * sum(v) / len(v) for t, v in per.items()}
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump({'model': a.model, 'machines': list(machines), 'oe': a.oe, 'bs': a.bs, 'per_task': acc, 'rows': rows}, open(a.out, 'w'), indent=1)
