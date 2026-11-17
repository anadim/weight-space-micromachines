"""BBH baselines without implants, in vLLM: same official 3-shot CoT prompts, greedy, same answer extraction as
scripts/run_bbh.py. Used for Qwen2.5-7B / 14B and for the vLLM Qwen3-0.6B-Base baseline (results/bbh/*_vllm.json).
python scripts/baselines/run_bbh_vllm.py MODEL OUT.json [--tasks a,b] [--tp 1] [--limit N] [--min_tokens 4]"""
import os, sys, re, json, time
from vllm import LLM, SamplingParams
A_ = sys.argv; mp, out = A_[1], A_[2]
arg = lambda k, d=None: A_[A_.index(k) + 1] if k in A_ else d
BBH = os.environ.get('BBH', 'data/bbh')
tasks = arg('--tasks').split(',') if arg('--tasks') else sorted(f[:-5] for f in os.listdir(f'{BBH}/data'))
limit = int(arg('--limit', 0))
pre = lambda t: open(f'{BBH}/cot/{t}.txt').read().split('-----', 1)[1].strip() + '\n\n'
def extract(text):
    m = re.findall(r'[Tt]he answer is\s*(.*?)(?:\.\s*$|\.\n|\n|$)', text.split('\n\nQ:')[0]); return m[-1].strip() if m else ''
def norm(a, task):
    a = a.strip().strip('.').strip()
    if re.fullmatch(r'\(?[A-R]\)?(\s.*)?', a) and task not in ('word_sorting', 'dyck_languages'): return '(' + re.match(r'\(?([A-R])', a).group(1) + ')'
    if task in ('multistep_arithmetic_two', 'object_counting'): m = re.search(r'-?\d+', a); return m.group() if m else a
    return a.lower()
llm = LLM(mp, tensor_parallel_size=int(arg('--tp', 1)), dtype='bfloat16', max_model_len=4096, gpu_memory_utilization=float(arg('--util', 0.88)), seed=0)
sp = SamplingParams(temperature=0, max_tokens=400, stop=['\n\nQ:'], min_tokens=int(arg('--min_tokens', 0)))
t0 = time.time(); per = {}; rows = []
for t in tasks:
    ex = json.load(open(f'{BBH}/data/{t}.json'))['examples'][:limit or None]
    outs = llm.generate([pre(t) + f"Q: {e['input']}\nA: Let's think step by step." for e in ex], sp, use_tqdm=False)
    for e, o in zip(ex, outs):
        txt = o.outputs[0].text; p = extract(txt); ok = norm(p, t) == norm(e['target'], t)
        per.setdefault(t, []).append(ok); rows.append({'task': t, 'gold': e['target'], 'pred': p, 'ok': ok, 'out': txt, 'finish': o.outputs[0].finish_reason, 'ntok': len(o.outputs[0].token_ids)})
    print(f'{t:40s} {100 * sum(per[t]) / len(per[t]):5.1f} (n={len(per[t])}) | {time.time() - t0:.0f}s', flush=True)
acc = {t: 100 * sum(v) / len(v) for t, v in per.items()}
print(f'MEAN {mp} over {len(acc)} tasks: {sum(acc.values()) / len(acc):.2f}')
json.dump({'model': mp, 'per_task': acc, 'mean': sum(acc.values()) / len(acc), 'rows': rows}, open(out, 'w'), indent=1)
