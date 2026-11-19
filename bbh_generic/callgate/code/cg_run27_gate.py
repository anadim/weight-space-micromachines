"""BBH 27-task eval with the call gate. Copy of run27_integ.py (same prompts/decoding/scoring, v2 tools attached:
A/B call form + Team C passive). The only change is the gated decode (cg_gate_gen). One run per --mode.

python cg_run27_gate.py OUT.json --mode none|gate|breaker --probe PROBE.pt [--bs 32 --max_new 600 --tasks a,b]
"""
import os, sys, re, json, time, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _cgpaths import CKPT, RES, BBH, COT, CODE
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer, AutoModelForCausalLM
import integ
from cg_gate_gen import generate, TC

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--model', default=CKPT)
ap.add_argument('--mode', default='none'); ap.add_argument('--probe', default=f'{RES}/probe_v3.pt')
ap.add_argument('--bs', type=int, default=32); ap.add_argument('--min_tokens', type=int, default=4)
ap.add_argument('--max_new', type=int, default=600); ap.add_argument('--tasks', default='')
ap.add_argument('--data', default=BBH); ap.add_argument('--cot', default=COT)
a = ap.parse_args()
dev = 'cuda'
tasks = a.tasks.split(',') if a.tasks else sorted(f[:-5] for f in os.listdir(a.data) if f.endswith('.json'))
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
t0 = time.time(); net = integ.attach_all(model, tok, integ.TOOLS_V2, veto=integ.VETO)
print(f'attached v2 ({time.time()-t0:.0f}s)', flush=True)
probe = torch.load(a.probe) if a.mode == 'gate' else None
pre = lambda t: open(f'{a.cot}/{t}.txt').read().split('-----', 1)[1].strip() + '\n\n'


def extract(text):
    m = re.findall(r'[Tt]he answer is\s*(.*?)(?:\.\s*$|\.\n|\n|$)', text.split('\n\nQ:')[0]); return m[-1].strip() if m else ''


def norm(x, task):
    x = x.strip().strip('.').strip()
    if re.fullmatch(r'\(?[A-R]\)?(\s.*)?', x) and task not in ('word_sorting', 'dyck_languages'): return '(' + re.match(r'\(?([A-R])', x).group(1) + ')'
    if task in ('multistep_arithmetic_two', 'object_counting'): m = re.search(r'-?\d+', x); return m.group() if m else x
    return x.lower()


per = {}; rows = []; t0 = time.time()
if os.path.exists(a.out + '.partial'):                       # resume: same run, same config; skip finished tasks
    old = json.load(open(a.out + '.partial'))
    if old.get('mode') == a.mode:
        rows = old['rows']
        for r in rows: per.setdefault(r['task'], []).append(r['ok'])
        print('resumed tasks:', sorted(per), flush=True)


def gen_retry(prompts):
    for k in range(40):
        try:
            return generate(model, tok, net, prompts, a.max_new, dev, mode=a.mode, probe=probe, min_tokens=a.min_tokens)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); print(f'OOM, retry {k+1} in 30s', flush=True); time.sleep(30)
    raise RuntimeError('OOM 40 times')


for t in tasks:
    if t in per: continue
    ex = json.load(open(f'{a.data}/{t}.json'))['examples']
    for b in range(0, len(ex), a.bs):
        ch = ex[b:b + a.bs]
        gens, stats = gen_retry([pre(t) + f"Q: {e['input']}\nA: Let's think step by step." for e in ch])
        for e, g, st in zip(ch, gens, stats):
            txt = tok.decode(g).replace('<|endoftext|>', ''); p = extract(txt); ok = norm(p, t) == norm(e['target'], t)
            per.setdefault(t, []).append(ok)
            rows.append({'task': t, 'gold': e['target'], 'pred': p, 'ok': ok, 'out': txt, 'ncall': st['ncall'], 'blocked': st['blocked']})
    json.dump({'mode': a.mode, 'max_new': a.max_new, 'partial': True, 'per_task': {k: 100 * sum(v) / len(v) for k, v in per.items()}, 'rows': rows}, open(a.out + '.partial', 'w'))
    rr = [r for r in rows if r['task'] == t]
    cr = 100 * sum('<tool_call>' in r['out'] for r in rr) / len(rr); blk = 100 * sum(r['blocked'] for r in rr) / len(rr)
    print(f'{t:40s} {100*sum(per[t])/len(per[t]):5.1f} call={cr:5.1f} blk={blk:5.1f} ncall={sum(r["ncall"] for r in rr)/len(rr):4.1f} | {time.time()-t0:.0f}s', flush=True)
acc = {t: 100 * sum(v) / len(v) for t, v in per.items()}
print(f'MEAN over {len(acc)} tasks: {sum(acc.values())/len(acc):.2f}')
json.dump({'mode': a.mode, 'max_new': a.max_new, 'per_task': acc, 'rows': rows}, open(a.out, 'w'), indent=1)
