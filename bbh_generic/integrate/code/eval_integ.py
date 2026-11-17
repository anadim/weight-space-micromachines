"""Evaluate one host on each team's fresh-test / out-of-style sets and on the composition set, with the tools of ALL teams
attached at once (or none). Prompts, decoding, answer extraction and scoring are copied from each team's own harness:
  A (teamA/run_eval.py): official BBH 3-shot CoT prompt of tracking / logical_deduction + "Q: ..\\nA: Let's think step by step.",
     max_new 600, norm = MC letter / lower.
  B (teamB/run_eval.py): official prompt for temporal/navigate/lies, zero-shot for meeting/robot/knights, max_new 400,
     norm with the time-range rule.
  C (teamC/run_gen.py): official prompt for the 3 BBH-style tasks; own 3-shot prompt files for the ood sets; max_new 400;
     numeric norm; lenient re-score (teamC/rescore.py) also reported for ood.
  compose: zero-shot "Q: ...\\nA: Let's think step by step.", max_new 700; strict ('the answer is') and lenient (last number /
     True/False/Yes/No) extraction; numeric answers compared as numbers.
Generation loop = scripts/run_bbh.py (greedy, left-padded batches, 4-token minimum, stop at '\\n\\nQ' or <|endoftext|>).

python eval_integ.py OUT.json --model M --tools none|all --sets A_test,A_oos2,B_test,B_oos,C_test,C_ood,compose [--bs 32]
"""
import os, sys, re, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _paths import G, VETO
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base'); ap.add_argument('--tools', default='none')
ap.add_argument('--sets', default='A_test,A_oos2,B_test,B_oos,C_test,C_ood,compose'); ap.add_argument('--bs', type=int, default=32)
ap.add_argument('--min_tokens', type=int, default=4); ap.add_argument('--limit', type=int, default=0)
a = ap.parse_args()
dev = 'cuda'
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
net = None
if a.tools != 'none':
    import integ
    machines = {'all': integ.TOOLS, 'all+passiveC': integ.TOOLS + integ.TOOLS_C_PASSIVE}[a.tools]
    t0 = time.time(); net = integ.attach_all(model, tok, machines, veto=VETO if 'passive' in a.tools else None)
    print(f'attached {machines} ({time.time() - t0:.0f}s)', flush=True)
eos = tok.convert_tokens_to_ids('<|endoftext|>')
cot = lambda path: open(path).read().split('-----', 1)[1].strip() + '\n\n'
QA = lambda q: f"Q: {q}\nA: Let's think step by step."


def extract(text):
    m = re.findall(r'[Tt]he answer is\s*(.*?)(?:\.\s*$|\.\n|\n|$)', text.split('\n\nQ:')[0]); return m[-1].strip() if m else ''


def normA(x):
    x = x.strip().strip('.').strip()
    if re.fullmatch(r'\(?[A-R]\)?(\s.*)?', x): return '(' + re.match(r'\(?([A-R])', x).group(1) + ')'
    return x.lower()


def normB(x, task):
    x = x.strip().strip('.').strip()
    if task == 'temporal':
        m = re.match(r'\(?([A-D])\)?(\s.*)?$', x); return '(' + m.group(1) + ')' if m else x
    x = re.sub(r'\s+', ' ', x.lower())
    return re.sub(r'(\d(?:am|pm)?)\s*(?:-|to)\s*(\d)', r'\1-\2', x)


def normC(x, task):
    x = x.strip().strip('.').strip()
    if task in ('multistep_arithmetic_two', 'object_counting') or task.startswith('num'):
        m = re.search(r'-?\d+', x.replace(',', '')); return m.group() if m else x
    return x.lower()


def lenientC(out, task):                                     # teamC/rescore.py
    t = out.split('\n\nQ:')[0].replace('<|endoftext|>', '')
    m = re.findall(r'[Tt]he answer is\s*(.*?)(?:\.\s*$|\.\n|\n|$)', t); p = m[-1] if m else ''
    if task.startswith('num'):
        nums = re.findall(r'-?\d+', (p or t).replace(',', '').replace('$', ''))
        return nums[-1] if nums else '' if not p else (re.findall(r'-?\d+', t.replace(',', '')) or [''])[-1]
    b = re.findall(r'True|False', p or t); return b[-1] if b else ''


def num_eq(p, g):
    m = re.findall(r'-?\d+(?:\.\d+)?', p.replace(',', '').replace('$', ''))
    return bool(m) and abs(float(m[0 if p else -1]) - float(g)) < 1e-6


def compose_ok(out, item, lenient):
    t = out.split('\n\nQ:')[0]; p = extract(t)
    if item['atype'] == 'num':
        if p: return num_eq(p, item['answer'])
        if not lenient: return False
        m = re.findall(r'-?\d+(?:\.\d+)?', t.replace(',', '').replace('$', '')); return bool(m) and abs(float(m[-1]) - float(item['answer'])) < 1e-6
    B = {'true': 'True', 'yes': 'True', 'false': 'False', 'no': 'False'}
    if p:
        w = re.findall(r'true|false|yes|no', p.lower()); return bool(w) and B[w[0]] == item['answer']
    if not lenient: return False
    w = re.findall(r'\b(True|False|Yes|No)\b', t); return bool(w) and B[w[-1].lower()] == item['answer']


# ---------------- items: (set, task, prompt, max_new, scorer, meta) ----------------
items = []
for s in a.sets.split(','):
    if s in ('A_test', 'A_oos2', 'A_var3'):
        f = {'A_test': 'test', 'A_oos2': 'oos2', 'A_var3': 'var3'}[s]                 # var3 = Team A judges' held-out round 3
        P = {'track': cot(f'{G}/teamA/cot/tracking_shuffled_objects_three_objects.txt'), 'order': cot(f'{G}/teamA/cot/logical_deduction_three_objects.txt')}
        for l in open(f'{G}/teamA/judge/variants_3.jsonl' if f == 'var3' else f'{G}/teamA/data/{f}.jsonl'):
            e = json.loads(l); items.append((s, e['task'], P[e['tool']] + QA(e['input']), 600, e['target'], {'gold_call': e.get('gold_call')}))
    elif s in ('B_test', 'B_oos'):
        BN = {'temporal': 'temporal_sequences', 'navigate': 'navigate', 'lies': 'web_of_lies'}
        for f in (('temporal_test', 'navigate_test', 'lies_test') if s == 'B_test' else ('meeting_oos', 'robot_oos', 'knights_oos')):
            task = f.split('_')[0]
            for l in open(f'{G}/teamB/data/{f}.jsonl'):
                e = json.loads(l); pre = cot(f'{G}/teamB/cot/{BN[task]}.txt') if task in BN else ''
                items.append((s, task, pre + QA(e['question']), 400, e['target'], {'gold_call': e.get('call')}))
    elif s == 'C_test':
        for t in ('multistep_arithmetic_two', 'boolean_expressions', 'object_counting'):
            for e in json.load(open(f'{G}/teamC/data/test/{t}.json'))['examples']:
                items.append((s, t, cot(f'{G}/teamC/cot/{t}.txt') + QA(e['input']), 400, e['target'], {}))
    elif s == 'C_ood':
        for t in ('num_list', 'num_shop', 'bool_code'):
            pf = open(f'{G}/teamC/data/ood/{t}.prompt.txt').read()
            for e in json.load(open(f'{G}/teamC/data/ood/{t}.json'))['examples']:
                items.append((s, t, pf + QA(e['input']), 400, e['target'], {}))
    elif s == 'compose':
        for l in open(f'{HERE}/../data/compose.jsonl'):
            e = json.loads(l); items.append((s, e['family'], QA(e['question']), 700, e['answer'], e))
if a.limit: items = [x for i, x in enumerate(items) if i % max(1, len(items) // a.limit) == 0]


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


rows = []; t0 = time.time()
groups = {}
for it in items: groups.setdefault((it[0], it[3]), []).append(it)
for (s, mx), lst in groups.items():
    for b in range(0, len(lst), a.bs):
        ch = lst[b:b + a.bs]
        for (sname, task, prompt, _, gold, meta), g in zip(ch, generate([x[2] for x in ch], mx)):
            txt = tok.decode(g).replace('<|endoftext|>', ''); p = extract(txt)
            r = {'set': sname, 'task': task, 'gold': gold, 'pred': p, 'out': txt, 'called': '<tool_call>' in txt}
            if sname.startswith('A_'): r['ok'] = normA(p) == normA(gold)
            elif sname.startswith('B_'): r['ok'] = normB(p, task) == normB(gold, task)
            elif sname.startswith('C_'):
                r['ok'] = normC(p, task) == normC(gold, task)
                if sname == 'C_ood': r['ok_lenient'] = lenientC(txt, task).lower() == gold.lower()
            else:
                r['ok'] = compose_ok(txt, meta, False); r['ok_lenient'] = compose_ok(txt, meta, True); r['id'] = meta['id']
            if meta.get('gold_call'): r['gold_call'] = meta['gold_call']
            rows.append(r)
    sub = [r for r in rows if r['set'] == s]
    print(f'{s:10s} {100 * sum(r["ok"] for r in sub) / len(sub):5.1f} (n={len(sub)}) | {time.time() - t0:.0f}s', flush=True)
per = {}
for r in rows: per.setdefault(f"{r['set']}/{r['task']}", []).append(r['ok'])
acc = {k: 100 * sum(v) / len(v) for k, v in per.items()}
for k, v in acc.items(): print(f'  {k:40s} {v:5.1f} (n={len(per[k])})')
json.dump({'model': a.model, 'tools': a.tools, 'per_task': acc, 'rows': rows}, open(a.out, 'w'), indent=1)
