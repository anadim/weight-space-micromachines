"""BBH (official 3-shot CoT prompt) with the micromachines attached: HF greedy decoding, plain argmax, no logic in the loop.
python bbh_gen.py MODEL OUT.json [--data data/bbh_dev.jsonl | --test [--tasks a,b]] [--machines calc,swap,...] [--limit N] [--no_mm]"""
import json, re, sys, glob, os, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from calcnet import attach_calcnet

A = sys.argv; model_path, out = A[1], A[2]
arg = lambda k, d=None: A[A.index(k) + 1] if k in A else d
machines = tuple(arg('--machines', 'calc,swap,bool,truth,logic,dyck,walk,time,sort').split(','))
BBH = os.environ.get('BBH', 'data/bbh')
pre = lambda t: open(f'{BBH}/cot/{t}.txt').read().split('-----', 1)[1].strip() + '\n\n'
if '--test' in A:
    tasks = sorted(os.path.basename(f)[:-5] for f in glob.glob(f'{BBH}/data/*.json'))
    if arg('--tasks'): tasks = arg('--tasks').split(',')
    rows = []
    for t in tasks:
        for i, e in enumerate(json.load(open(f'{BBH}/data/{t}.json'))['examples']):
            rows.append({'task': t, 'i': i, 'prompt': pre(t) + f"Q: {e['input']}\nA: Let's think step by step.", 'gold': e['target']})
else:
    rows = [dict(json.loads(l), i=k) for k, l in enumerate(open(arg('--data', 'data/bbh_dev.jsonl')))]
if arg('--limit'): rows = [r for t in sorted({x['task'] for x in rows}) for r in [x for x in rows if x['task'] == t][:int(arg('--limit'))]]

tok = AutoTokenizer.from_pretrained(model_path); tok.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).cuda().eval()
PYTOOL = '--pytool' in A          # external-tool baseline: the host runs Python at each call end and forces the box tokens
net = None if ('--no_mm' in A or PYTOOL) else attach_calcnet(model, tok, machines=machines)
eos = tok.convert_tokens_to_ids('<|endoftext|>')
if PYTOOL:
    from pytool import PyTool
    tool = PyTool(tok, machines)


def extract(text):
    m = re.findall(r'[Tt]he answer is\s*(.*?)(?:\.\s*$|\.\n|\n|$)', text.split('\n\nQ:')[0])
    return m[-1].strip() if m else ''


def norm(a, task):
    task = task.split('__')[0]                       # e.g. word_sorting__OOD
    a = a.strip().strip('.').strip()
    if re.fullmatch(r'\(?[A-R]\)?(\s.*)?', a) and task not in ('word_sorting', 'dyck_languages'):
        return '(' + re.match(r'\(?([A-R])', a).group(1) + ')'
    if task in ('multistep_arithmetic_two', 'object_counting'):
        m = re.search(r'-?\d+', a); return m.group() if m else a
    return a.lower()


@torch.no_grad()
def generate(prompts, max_new=512):
    enc = tok(prompts, return_tensors='pt', padding=True).to('cuda'); B = len(prompts); att = enc.attention_mask
    def fwd(x, att, past=None):
        o = model.model(input_ids=x, attention_mask=att, past_key_values=past, use_cache=True)
        return model.lm_head(o.last_hidden_state[:, -1]).float(), o.past_key_values
    if net is not None: net.prime(enc.input_ids)
    lg, past = fwd(enc.input_ids, att)
    gen = [[] for _ in range(B)]; done = [False] * B; text = [''] * B; queue = [[] for _ in range(B)]
    for _ in range(max_new):
        nxt = lg.argmax(-1); nl = nxt.tolist()
        if PYTOOL:
            for i in range(B):
                if done[i]: continue
                if queue[i]: nl[i] = queue[i].pop(0)
                elif nl[i] in tool.ce_ids: queue[i] = tool.box_tokens(gen[i] + [nl[i]])
            nxt = torch.tensor(nl, device='cuda')
        for i in range(B):
            if done[i]: continue
            gen[i].append(nl[i])
            if nl[i] == eos: done[i] = True; continue
            if len(gen[i]) % 8 == 0:
                tail = tok.decode(gen[i][-24:])
                if '\n\nQ:' in tail or '\nQ:' in tail: done[i] = True
        if all(done): break
        nxt = torch.where(torch.tensor(done, device='cuda'), torch.full_like(nxt, eos), nxt)
        att = torch.cat([att, att.new_ones(B, 1)], 1)
        if net is not None: net.step(nxt)
        lg, past = fwd(nxt[:, None], att, past)
    return gen


t0 = time.time(); res = []
order = sorted(range(len(rows)), key=lambda i: len(rows[i]['prompt']))
bs = int(arg('--bs', 48))
for b in range(0, len(order), bs):
    idx = order[b:b + bs]
    for i, g in zip(idx, generate([rows[i]['prompt'] for i in idx])):
        txt = tok.decode(g, skip_special_tokens=False).replace('<|endoftext|>', '')
        p = extract(txt); r = rows[i]
        res.append({'task': r['task'], 'i': r['i'], 'gold': r['gold'], 'pred': p, 'ok': norm(p, r['task']) == norm(r['gold'], r['task']), 'out': txt, 'ntok': len(g)})
per = {}
for r in res: per.setdefault(r['task'], [0, 0]); per[r['task']][0] += 1; per[r['task']][1] += r['ok']
acc = {t: 100 * v[1] / v[0] for t, v in sorted(per.items())}
json.dump({'model': model_path, 'macro': sum(acc.values()) / len(acc), 'per_task': acc, 'n': {t: v[0] for t, v in per.items()}, 'machines': None if net is None else machines, 'mode': 'pytool' if PYTOOL else ('none' if net is None else 'machines'), 'seconds': time.time() - t0, 'gen_tokens': sum(r['ntok'] for r in res), 'bs': bs}, open(out, 'w'), indent=1)
with open(out.replace('.json', '_rows.jsonl'), 'w') as f:
    for r in res: f.write(json.dumps(r) + '\n')
for t, v in acc.items(): print(f'{t:42s} {v:5.1f}  (n={per[t][0]})')
print(f'MACRO {model_path} ({"python tool" if PYTOOL else "no machines" if net is None else ",".join(machines)}): {sum(acc.values())/len(acc):.2f} over {len(acc)} tasks | {time.time()-t0:.0f}s')
