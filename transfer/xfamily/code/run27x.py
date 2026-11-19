"""All-27-task real-BBH run. Copy of Team C's run_gen.py generation loop (= scripts/run_bbh.py: official 3-shot CoT prompt
of each task, greedy, 4-token minimum, stop at '\\n\\nQ' or <|endoftext|>, same answer extraction); only the tool attach
is generalised so any team's tools can be attached. Tasks without a tool: the host just answers.
Scoring is done afterwards by score27.py (official norm of bbh_peek/scripts/score.py); this script prints only per-task %.

python run27.py OUT.json --model M --tools none|A|B|C --max_new N [--bs 32]
"""
import os, sys, re, json, time, argparse, torch
R = os.environ.get('BFT_R', os.path.dirname(os.path.abspath(__file__))); G = os.environ.get('WSM_REPO', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')))+'/bbh_generic'
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base'); ap.add_argument('--tools', default='none')
ap.add_argument('--bs', type=int, default=32); ap.add_argument('--min_tokens', type=int, default=4); ap.add_argument('--max_new', type=int, default=400)
ap.add_argument('--data', default='data/bbh/data'); ap.add_argument('--cot', default='data/bbh/cot'); ap.add_argument('--tasks', default=''); ap.add_argument('--oe', default='.')
ap.add_argument('--veto', default='runs/veto_qwen3.pt')
a = ap.parse_args()
tasks = a.tasks.split(',') if a.tasks else sorted(f[:-5] for f in os.listdir(a.data) if f.endswith('.json'))
dev = 'cuda'
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
net = None; t0 = time.time()
if a.tools == 'A':
    sys.path.insert(0, f'{G}/teamA'); import register; from micromachines.machine import attach
    net = attach(model, tok, machines=register.TOOLS)
elif a.tools == 'B':
    sys.path.insert(0, f'{G}/teamB'); import circuits_B; from micromachines.machine import attach
    net = attach(model, tok, machines=circuits_B.TOOLS_B)
elif a.tools == 'C':      # xfamily: generic per-vocab sanitizer so calc_gen/expr_gen/bool_gen build on any single-digit tokenizer
    sys.path.insert(0, f'{G}/teamC'); import teamc_tools as TT
    from micromachines.specs import SPECS as _SP
    from micromachines import machine as _MM
    TT.register(a.oe, 1)
    def _san(name):
        sp = _SP[name]; txt = dict(sp['text'])
        for s, v in list(txt.items()):
            alts = list(v) if isinstance(v, (list, tuple)) else [v]
            keep = [x for x in alts if len(tok(x, add_special_tokens=False)['input_ids']) == 1]
            if not keep and s == 'OE' and a.oe == 'veto' and name in ('expr_gen', 'calc_gen'):
                keep = [' ']; print(f'  sanitize {name}.OE: {alts} -> placeholder " " (overwritten by the solved veto)', flush=True)
            if not keep:
                raise SystemExit(f'SANITIZE: symbol {s!r} in {name} has NO single-token form for this vocab -> tool cannot transfer')
            if len(keep) != len(alts):
                print(f'  sanitize {name}.{s}: {alts} -> {keep}', flush=True)
            txt[s] = keep[0] if len(keep) == 1 else tuple(keep)
        _SP[name] = dict(sp, text=txt)
    for nm in ('expr_gen', 'calc_gen', 'bool_gen'): _san(nm)
    A = _MM.Attached(model, tok, machines=('expr_gen', 'calc_gen', 'bool_gen'))
    for n in ('expr_gen', 'calc_gen', 'bool_gen'):
        if _SP[n].get('f64'): A.nets[n].stack1 = TT.F64(A.nets[n].stack1).to(model.device)
        if 'ANY' in _SP[n]['syms']:
            i = _SP[n]['syms'].index('ANY'); A.nets[n].reader_w[i] = 0; A.nets[n].reader_b[i] = 1
    import torch as _t
    _v = _t.load(a.veto)['v']; A.set_veto('calc_gen', _v, zero_os=True)
    if a.oe == 'veto': A.set_veto('expr_gen', _v, zero_os=False)      # as teamc_tools.attach_gen(oe='veto')
    print('Team C attached, oe =', a.oe, flush=True)
    net = A
if net is not None: print(f'attached {a.tools} ({time.time() - t0:.0f}s)', flush=True)
eos = tok.eos_token_id  # xfamily: host's own end-of-text (was the literal Qwen token)
pre = lambda t: open(f'{a.cot}/{t}.txt').read().split('-----', 1)[1].strip() + '\n\n'


def extract(text):
    m = re.findall(r'[Tt]he answer is\s*(.*?)(?:\.\s*$|\.\n|\n|$)', text.split('\n\nQ:')[0]); return m[-1].strip() if m else ''


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
    print(f'{t:40s} {100 * sum(per[t]) / len(per[t]):5.1f} (n={len(per[t])}) | {time.time() - t0:.0f}s', flush=True)
    acc = {t: 100 * sum(v) / len(v) for t, v in per.items()}      # saved after every task (a pod restart lost one run)
    json.dump({'model': a.model, 'tools': a.tools, 'max_new': a.max_new, 'bs': a.bs, 'per_task': acc, 'rows': rows}, open(a.out + '.tmp', 'w'))
    os.replace(a.out + '.tmp', a.out)
print(f'MEAN over {len(acc)} tasks: {sum(acc.values()) / len(acc):.2f}')
