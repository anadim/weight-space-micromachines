"""[latgate copy: adds --mode compiled (cg_compiled.py: the gate as an LM-head implant + latch register; shadow check of
the Python gate at every step, counts saved under 'shadow')]
Own held-out / fresh sets eval with the call gate. Copy of eval_integ.py loaders + scoring, gated decode, v2 tools.
python cg_sets_gate.py OUT.json --mode none|gate|breaker --probe P.pt --sets A_test,A_oos2,B_test,B_oos,C_test,C_ood
"""
import os, sys, re, json, time, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _cgpaths import G, CKPT, RES, CODE
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer, AutoModelForCausalLM
import integ
from cg_compiled import generate, SHADOW   # latgate: + mode compiled

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--model', default=CKPT)
ap.add_argument('--mode', default='none'); ap.add_argument('--probe', default=f'{RES}/probe_v3.pt')
ap.add_argument('--sets', default='A_test,A_oos2,B_test,B_oos,C_test,C_ood'); ap.add_argument('--bs', type=int, default=32)
a = ap.parse_args()
dev = 'cuda'
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
net = integ.attach_all(model, tok, integ.TOOLS_V2, veto=integ.VETO)
probe = torch.load(a.probe) if a.mode in ('gate', 'compiled') else None
cot = lambda p: open(p).read().split('-----', 1)[1].strip() + '\n\n'
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
    x = re.sub(r'\s+', ' ', x.lower()); return re.sub(r'(\d(?:am|pm)?)\s*(?:-|to)\s*(\d)', r'\1-\2', x)
def normC(x, task):
    x = x.strip().strip('.').strip()
    if task in ('multistep_arithmetic_two', 'object_counting') or task.startswith('num'):
        m = re.search(r'-?\d+', x.replace(',', '')); return m.group() if m else x
    return x.lower()
def lenientC(out, task):
    t = out.split('\n\nQ:')[0].replace('<|endoftext|>', '')
    m = re.findall(r'[Tt]he answer is\s*(.*?)(?:\.\s*$|\.\n|\n|$)', t); p = m[-1] if m else ''
    if task.startswith('num'):
        nums = re.findall(r'-?\d+', (p or t).replace(',', '').replace('$', '')); return nums[-1] if nums else ''
    bl = re.findall(r'True|False', p or t); return bl[-1] if bl else ''

items = []
for s in a.sets.split(','):
    if s in ('A_test', 'A_oos2'):
        f = {'A_test': 'test', 'A_oos2': 'oos2'}[s]
        P = {'track': cot(f'{G}/teamA/cot/tracking_shuffled_objects_three_objects.txt'), 'order': cot(f'{G}/teamA/cot/logical_deduction_three_objects.txt')}
        for l in open(f'{G}/teamA/data/{f}.jsonl'):
            e = json.loads(l); items.append((s, e['task'], P[e['tool']] + QA(e['input']), 600, e['target']))
    elif s in ('B_test', 'B_oos'):
        BN = {'temporal': 'temporal_sequences', 'navigate': 'navigate', 'lies': 'web_of_lies'}
        for f in (('temporal_test', 'navigate_test', 'lies_test') if s == 'B_test' else ('meeting_oos', 'robot_oos', 'knights_oos')):
            task = f.split('_')[0]
            for l in open(f'{G}/teamB/data/{f}.jsonl'):
                e = json.loads(l); pre = cot(f'{G}/teamB/cot/{BN[task]}.txt') if task in BN else ''
                items.append((s, task, pre + QA(e['question']), 400, e['target']))
    elif s == 'C_test':
        for t in ('multistep_arithmetic_two', 'boolean_expressions', 'object_counting'):
            for e in json.load(open(f'{G}/teamC/data/test/{t}.json'))['examples']:
                items.append((s, t, cot(f'{G}/teamC/cot/{t}.txt') + QA(e['input']), 400, e['target']))
    elif s == 'C_ood':
        for t in ('num_list', 'num_shop', 'bool_code'):
            pf = open(f'{G}/teamC/data/ood/{t}.prompt.txt').read()
            for e in json.load(open(f'{G}/teamC/data/ood/{t}.json'))['examples']:
                items.append((s, t, pf + QA(e['input']), 400, e['target']))

_lim = int(os.environ.get('CG_LIMIT', '0'))                 # latgate: smoke runs only (first N items per set/task)
if _lim:
    _c, _keep = {}, []
    for it in items:
        _c[it[0], it[1]] = _c.get((it[0], it[1]), 0) + 1
        if _c[it[0], it[1]] <= _lim: _keep.append(it)
    items = _keep
rows = []; t0 = time.time(); groups = {}
for it in items: groups.setdefault((it[0], it[3]), []).append(it)
for (s, mx), lst in groups.items():
    for b in range(0, len(lst), a.bs):
        ch = lst[b:b + a.bs]
        for _k in range(40):
            try: gens, stats = generate(model, tok, net, [x[2] for x in ch], mx, dev, mode=a.mode, probe=probe); break
            except torch.OutOfMemoryError: torch.cuda.empty_cache(); print('OOM retry', _k + 1, flush=True); time.sleep(30)
        for (sn, task, prompt, _, gold), g, st in zip(ch, gens, stats):
            txt = tok.decode(g).replace('<|endoftext|>', ''); p = extract(txt)
            r = {'set': sn, 'task': task, 'gold': gold, 'pred': p, 'out': txt, 'called': '<tool_call>' in txt, 'ncall': st['ncall'], 'blocked': st['blocked']}
            if sn.startswith('A_'): r['ok'] = normA(p) == normA(gold)
            elif sn.startswith('B_'): r['ok'] = normB(p, task) == normB(gold, task)
            else:
                r['ok'] = normC(p, task) == normC(gold, task)
                if sn == 'C_ood': r['ok_lenient'] = lenientC(txt, task).lower() == gold.lower()
            rows.append(r)
    sub = [r for r in rows if r['set'] == s]
    print(f'{s:10s} {100*sum(r["ok"] for r in sub)/len(sub):5.1f} call={100*sum(r["called"] for r in sub)/len(sub):5.1f} blk={100*sum(r["blocked"] for r in sub)/len(sub):5.1f} (n={len(sub)}) | {time.time()-t0:.0f}s', flush=True)
per = {}
for r in rows: per.setdefault(f"{r['set']}/{r['task']}", []).append(r['ok'])
for k, v in per.items(): print(f'  {k:40s} {100*sum(v)/len(v):5.1f} (n={len(v)})')
print('SHADOW', SHADOW, flush=True)
json.dump({'mode': a.mode, 'rows': rows, 'shadow': SHADOW}, open(a.out, 'w'), indent=1)
