"""Qwen3-0.6B-Base generations for the detect-and-rethink machine (frozen, greedy, shots_v1 plain prompt).
For SPLIT in (train = 1500 train problems, seed 99 shuffle; test = 1319):
  base     : greedy solution
  rethink  : base + '\\n' + PHRASE, continued (last '####' counts; none = base stands)
  ocalc    : greedy with an exact oracle calculator: whenever the text ends in 'expr =' (>=1 operator), the exact value is
             appended (' v'), then generation continues
  ocalc_rt : ocalc + rethink (the oracle calculator stays on during the rethink)
Uses vLLM (the generation numbers in results/gate/ come from this vLLM run, not the HF loop of scripts/run_gsm.py).
python scripts/gate/gen_rethink.py OUT.json   (run from the repo root after data/prepare_gsm.py)
Optional env: SPLITS=test,train (default test,hard,train) and NO_OCALC=1 (skip the oracle-calculator generations; enough
for the probe of the rethink implant, which uses only 'base' and 'rethink')."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import json, re, sys, os, random
from decimal import Decimal, getcontext
from vllm import LLM, SamplingParams
from micromachines.calcfmt import strip_solution
getcontext().prec = 40
MP = 'Qwen/Qwen3-0.6B-Base'; OUT = sys.argv[1]; PH = "Hmm, wait. Let me rethink this.\n"
shots = json.load(open('data/gsm8k/shots_v1.json')); P = ''
for s in shots:
    body, fin = strip_solution(s['solution'] + f"\n#### {s['answer']}"); P += f"Question: {s['question']}\nAnswer: {body}\n#### {fin}\n\n"
tr = [json.loads(l) for l in open('data/gsm8k/train.jsonl')]; random.Random(99).shuffle(tr)
sq = {s['question'] for s in shots}; tr = [r for r in tr if r['question'] not in sq][:1500]
te = [json.loads(l) for l in open('data/gsm8k/test.jsonl')]
hd = [json.loads(l) for l in open('data/gsmhard/test.jsonl')]
llm = LLM(MP, dtype='bfloat16', max_model_len=4096, gpu_memory_utilization=float(os.environ.get('UTIL', '0.5')), enable_prefix_caching=True, seed=0,
          enforce_eager=os.environ.get('EAGER') == '1')
ans = lambda t: (lambda m: m[-1].replace(',', '') if m else None)(re.findall(r'####\s*\$?(-?[\d,]*\.?\d+)', t))
def ok(a, g):
    try: return a is not None and abs(float(a) - float(g)) < 1e-6 * max(1, abs(float(g)))
    except ValueError: return False
EXPR = re.compile(r'((?:\(?\$?\d[\d,]*(?:\.\d+)?\)? ?[-+*/x×] ?)+\(?\$?\d[\d,]*(?:\.\d+)?\)?) ?=$')
def calc(tail):
    m = EXPR.search(tail)
    if not m: return None
    e = m.group(1).replace('$', '').replace(',', '').replace('x', '*').replace('×', '*')
    try: v = eval(re.sub(r'(\d+(?:\.\d+)?)', r'Decimal("\1")', e))
    except Exception: return None
    v = v.quantize(Decimal('0.0001')).normalize() if v != v.to_integral() else v.to_integral()
    return format(v, 'f')
def run_calc(prompts, max_new=360, stop_end=('\n\n',)):
    """greedy with the oracle calculator; returns completions"""
    done = [None] * len(prompts); cur = [''] * len(prompts); live = list(range(len(prompts)))
    for _ in range(40):
        if not live: break
        O = llm.generate([prompts[k] + cur[k] for k in live], SamplingParams(temperature=0, max_tokens=max_new, stop=['='] + list(stop_end), include_stop_str_in_output=True), use_tqdm=False)
        nl = []
        for k, o in zip(live, O):
            t = o.outputs[0].text; cur[k] += t
            if t.endswith('=') and len(cur[k]) < 2500:
                v = calc(cur[k][-60:])
                if v is not None: cur[k] += ' ' + v
                nl.append(k)
            else: done[k] = cur[k]
        live = nl
    for k in live: done[k] = cur[k]
    return [d.split('\n\n')[0] if '\n\nQuestion' not in d else d.split('\n\nQuestion')[0] for d in done]
res = {}
SPL = os.environ.get('SPLITS', 'test,hard,train').split(','); NO_OCALC = os.environ.get('NO_OCALC') == '1'
for split, rows in [(k, v) for k, v in (('test', te), ('hard', hd), ('train', tr)) if k in SPL]:
    gold = [r['answer'].split('####')[1].strip().replace(',', '') for r in rows]
    pre = [P + f"Question: {r['question']}\nAnswer:" for r in rows]
    base = [o.outputs[0].text.rstrip() for o in llm.generate(pre, SamplingParams(temperature=0, max_tokens=320, stop=['\n\n']), use_tqdm=False)]
    O = llm.generate([p + b + '\n' + PH for p, b in zip(pre, base)], SamplingParams(temperature=0, max_tokens=360, stop=['\n\nQuestion', '\n\n\n']), use_tqdm=False)
    rt = [(b + '\n' + PH + o.outputs[0].text).rstrip() if ans(o.outputs[0].text) else b for b, o in zip(base, O)]
    R = {'q': [r['question'] for r in rows], 'gold': gold}; gens = [('base', base), ('rethink', rt)]
    if not NO_OCALC:
        oc = [t.rstrip() for t in run_calc(pre, stop_end=('\n\n',))]
        O2 = run_calc([p + c + '\n' + PH for p, c in zip(pre, oc)], stop_end=('\n\nQuestion', '\n\n\n'))
        ocr = [(c + '\n' + PH + t).rstrip() if ans(t) else c for c, t in zip(oc, O2)]
        gens += [('ocalc', oc), ('ocalc_rt', ocr)]
    for name, T in gens:
        R[name] = T; R[name + '_ok'] = [ok(ans(t), g) for t, g in zip(T, gold)]
        print(f'{split:5s} {name:9s} {100 * sum(R[name + "_ok"]) / len(rows):.2f}%', flush=True)
    res[split] = R
    json.dump(res, open(OUT, 'w'))
json.dump(res, open(OUT, 'w'))
