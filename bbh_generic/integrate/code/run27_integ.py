"""INTEGRATION copy of bbh_ft100/scripts/run27.py: adds --tools all (all 8 tools of Teams A, B, C in ONE Attached, integ.py).
All-27-task real-BBH run. Copy of Team C's run_gen.py generation loop (= scripts/run_bbh.py: official 3-shot CoT prompt
of each task, greedy, 4-token minimum, stop at '\\n\\nQ' or <|endoftext|>, same answer extraction); only the tool attach
is generalised so any team's tools can be attached. Tasks without a tool: the host just answers.
Scoring is done afterwards by score27.py (official norm of bbh_peek/scripts/score.py); this script prints only per-task %.

python run27.py OUT.json --model M --tools none|A|B|C --max_new N [--bs 32]
"""
import os, sys, re, json, time, argparse, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _paths import G, BBH, COT, VETO
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base'); ap.add_argument('--tools', default='none')
ap.add_argument('--bs', type=int, default=32); ap.add_argument('--min_tokens', type=int, default=4); ap.add_argument('--max_new', type=int, default=400)
ap.add_argument('--data', default=BBH); ap.add_argument('--cot', default=COT); ap.add_argument('--tasks', default='')
a = ap.parse_args()
tasks = a.tasks.split(',') if a.tasks else sorted(f[:-5] for f in os.listdir(a.data) if f.endswith('.json'))
dev = 'cuda'
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
net = None; t0 = time.time()
if a.tools == 'all':
    import integ
    net = integ.attach_all(model, tok, integ.TOOLS)
elif a.tools == 'A':
    sys.path.insert(0, f'{G}/teamA'); import register; from micromachines.machine import attach
    net = attach(model, tok, machines=register.TOOLS)
elif a.tools == 'B':
    sys.path.insert(0, f'{G}/teamB'); import circuits_B; from micromachines.machine import attach
    net = attach(model, tok, machines=circuits_B.TOOLS_B)
elif a.tools == 'C':      # exactly as bbh_peek run_job.sh C_tools (run_gen.py defaults: oe dotnl, bool_dot 1)
    sys.path.insert(0, f'{G}/teamC'); import teamc_tools as TT
    net = TT.attach_gen(model, tok, ('expr_gen', 'calc_gen', 'bool_gen'), veto=VETO, oe='dotnl', bool_dot=1)
if net is not None: print(f'attached {a.tools} ({time.time() - t0:.0f}s)', flush=True)
eos = tok.convert_tokens_to_ids('<|endoftext|>')
pre = lambda t: open(f'{a.cot}/{t}.txt').read().split('-----', 1)[1].strip() + '\n\n'


def extract(text):
    m = re.findall(r'[Tt]he answer is\s*(.*?)(?:\.\s*$|\.\n|\n|$)', text.split('\n\nQ:')[0]); return m[-1].strip() if m else ''


@torch.no_grad()
def generate(prompts, max_new):
    seqs = [tok(p, add_special_tokens=False)['input_ids'] for p in prompts]; L = max(map(len, seqs)); B_ = len(seqs)
    x = torch.tensor([[eos] * (L - len(s)) + s for s in seqs], device=dev); att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=dev)
    gs = None
    if net is not None:
        net.prime(x); import integ; gs = integ.GraphStep(net, B_)          # CUDA-graph replay of net.step (bit-identical, check_graph.py)
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
        if gs is not None: gs.step(nxt)
        o = model.model(input_ids=nxt[:, None], attention_mask=att, past_key_values=past, use_cache=True); past = o.past_key_values
        lg = model.lm_head(o.last_hidden_state[:, -1]).float()
    return gen


def norm(x, task):   # verbatim from bbh_peek/scripts/score.py
    x = x.strip().strip('.').strip()
    if re.fullmatch(r'\(?[A-R]\)?(\s.*)?', x) and task not in ('word_sorting', 'dyck_languages'): return '(' + re.match(r'\(?([A-R])', x).group(1) + ')'
    if task in ('multistep_arithmetic_two', 'object_counting'): m = re.search(r'-?\d+', x); return m.group() if m else x
    return x.lower()


t0 = time.time(); per = {}; rows = []
for t in tasks:
    ex = json.load(open(f'{a.data}/{t}.json'))['examples']
    for b in range(0, len(ex), a.bs):
        ch = ex[b:b + a.bs]
        for e, g in zip(ch, generate([pre(t) + f"Q: {e['input']}\nA: Let's think step by step." for e in ch], a.max_new)):
            txt = tok.decode(g).replace('<|endoftext|>', ''); p = extract(txt); ok = norm(p, t) == norm(e['target'], t)
            per.setdefault(t, []).append(ok)
            rows.append({'task': t, 'gold': e['target'], 'pred': p, 'ok': ok, 'out': txt})
    json.dump({'model': a.model, 'tools': a.tools, 'max_new': a.max_new, 'bs': a.bs, 'partial': True, 'per_task': {k: 100 * sum(v) / len(v) for k, v in per.items()}, 'rows': rows}, open(a.out + '.partial', 'w'))
    print(f'{t:40s} {100 * sum(per[t]) / len(per[t]):5.1f} (n={len(per[t])}) call={100 * sum("<tool_call>" in r["out"] for r in rows if r["task"] == t) / len(per[t]):5.1f} | {time.time() - t0:.0f}s', flush=True)
acc = {t: 100 * sum(v) / len(v) for t, v in per.items()}
print(f'MEAN over {len(acc)} tasks: {sum(acc.values()) / len(acc):.2f}')
json.dump({'model': a.model, 'tools': a.tools, 'max_new': a.max_new, 'bs': a.bs, 'per_task': acc, 'rows': rows}, open(a.out, 'w'), indent=1)
