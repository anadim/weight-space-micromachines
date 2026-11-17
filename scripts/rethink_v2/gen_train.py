"""TRAIN-side generations for the v2 rethink implant (vLLM, used ONLY to make labelled training data; the evaluated arms
all run through the HF loop of scripts/rethink_v2/run_gsm_v2.py).  Prompt = exactly the token sequence of
scripts/run_gsm_circuit.py (shots_v1, plain).  For each problem: 1 greedy + K sampled solutions (stop at a blank line);
then for EVERY solution that writes '####': the 'rethink-at-####' continuation = solution tokens up to (not incl.) the
first '####'-token, + the phrase (with a '\\n' in front if the previous token does not end in '\\n'), continued greedy.
python scripts/rethink_v2/gen_train.py OUT.json --src data/gsm8k/train.jsonl [--n 0] [--K 3] [--temp 0.8]"""
import os, sys, json, re, argparse, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from transformers import AutoTokenizer
from micromachines.calcfmt import strip_solution
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--src', required=True)
ap.add_argument('--n', type=int, default=0); ap.add_argument('--K', type=int, default=3); ap.add_argument('--temp', type=float, default=0.8)
ap.add_argument('--max_new', type=int, default=600); ap.add_argument('--util', type=float, default=0.45)
a = ap.parse_args()
MP = 'Qwen/Qwen3-0.6B-Base'; PH = "Hmm, wait. Let me rethink this.\n"
tok = AutoTokenizer.from_pretrained(MP); T = lambda s: tok(s, add_special_tokens=False)['input_ids']
eos = tok.convert_tokens_to_ids('<|endoftext|>')
shots = json.load(open('data/gsm8k/shots_v1.json')); prompt = []
for s in shots:
    body, fin = strip_solution(s['solution'] + f"\n#### {s['answer']}"); prompt += T(f"Question: {s['question']}\nAnswer: {body}\n#### {fin}\n\n")
rows = [json.loads(l) for l in open(a.src)]
sq = {s['question'] for s in shots}; rows = [r for r in rows if r['question'] not in sq]
if a.n: rows = rows[:a.n]
txt = [tok.decode([i]) for i in range(len(tok))]
HASH = {i for i, t in enumerate(txt) if '####' in t}; ENDNL = [t.endswith('\n') for t in txt]
NUM = r'####\s*(-?[\d,]*\.?\d+)'
def pred(s): m = re.findall(NUM, s); return m[-1].replace(',', '') if m else None
def ok(p, g):
    try: return p is not None and abs(float(p) - float(g)) <= 1e-6 * max(1.0, abs(float(g)))
    except ValueError: return False
llm = LLM(MP, dtype='bfloat16', max_model_len=4096, gpu_memory_utilization=a.util, enable_prefix_caching=True, seed=0, enforce_eager=os.environ.get('EAGER') == '1')
def clean(ids):            # = the HF loop: stop at <|endoftext|> or a blank line; score text up to the first blank line
    ids = [i for i in ids if i != eos]
    return ids, tok.decode(ids).split('\n\n')[0]
P = [prompt + T(f"Question: {r['question']}\nAnswer:") for r in rows]
gold = [r['answer'].split('####')[1].strip().replace(',', '') for r in rows]
G = llm.generate([TokensPrompt(prompt_token_ids=p) for p in P], SamplingParams(temperature=0, max_tokens=a.max_new, stop=['\n\n']), use_tqdm=False)
S = llm.generate([TokensPrompt(prompt_token_ids=p) for p in P], SamplingParams(temperature=a.temp, n=a.K, max_tokens=a.max_new, stop=['\n\n'], seed=7), use_tqdm=False) if a.K else [None] * len(P)
sols = []
for k, (g, s) in enumerate(zip(G, S)):
    outs = [('greedy', g.outputs[0])] + ([('sample', o) for o in s.outputs] if s is not None else [])
    for kind, o in outs:
        ids, t = clean(list(o.token_ids)); p = pred(t)
        sols.append({'q': k, 'kind': kind, 'ids': ids, 'pred': p, 'ok': ok(p, gold[k])})
# rethink-at-#### continuations
jobs = []
for j, s in enumerate(sols):
    h = next((u for u, i in enumerate(s['ids']) if i in HASH), None); s['hash_pos'] = h
    if h is None: continue
    pre = s['ids'][:h]
    nl = ENDNL[pre[-1]] if pre else True
    add = T(('' if nl else '\n') + PH)
    s['rt_prefix_len'] = len(pre) + len(add); jobs.append((j, pre + add))
R = llm.generate([TokensPrompt(prompt_token_ids=P[sols[j]['q']] + x) for j, x in jobs], SamplingParams(temperature=0, max_tokens=a.max_new, stop=['\n\n']), use_tqdm=False)
for (j, x), o in zip(jobs, R):
    ids, _ = clean(list(o.outputs[0].token_ids)); full = x + ids
    t = tok.decode(full).split('\n\n')[0]; p = pred(t)
    sols[j]['rt_ids'] = full; sols[j]['rt_pred'] = p; sols[j]['rt_ok'] = ok(p, gold[sols[j]['q']])
# budget cap for the rethink@#### arm: if the rethink part has no '####' within CAP tokens, force ('\n' +) '####' there
HASH_ID = T('####')[0]
for CAP in (150, 300):
    jobs2 = []
    for j, s in enumerate(sols):
        if 'rt_ids' not in s: continue
        part = s['rt_ids'][s['rt_prefix_len']:]
        if any(i in HASH for i in part[:CAP]) or len(part) < CAP:
            s[f'cap{CAP}_ok'] = s['rt_ok']; s[f'cap{CAP}_len'] = len(s['rt_ids']); continue
        pre = s['rt_ids'][:s['rt_prefix_len'] + CAP]
        x = pre + ([] if ENDNL[pre[-1]] else T('\n')) + [HASH_ID]
        jobs2.append((j, x))
    R2 = llm.generate([TokensPrompt(prompt_token_ids=P[sols[j]['q']] + x) for j, x in jobs2], SamplingParams(temperature=0, max_tokens=24, stop=['\n']), use_tqdm=False)
    for (j, x), o in zip(jobs2, R2):
        t = tok.decode(x + list(o.outputs[0].token_ids)).split('\n\n')[0]; p = pred(t)
        sols[j][f'cap{CAP}_ok'] = ok(p, gold[sols[j]['q']]); sols[j][f'cap{CAP}_len'] = len(x) + len(o.outputs[0].token_ids)
# after-answer rethink (the v1 implant's intervention point): answer text + '\n' + phrase, continued; no new '####' = first answer stands
jobs3 = []
for j, s in enumerate(sols):
    if s['pred'] is None: continue
    base = tok.decode(s['ids']).split('\n\n')[0].rstrip()
    jobs3.append((j, T(base + '\n' + PH)))
R3 = llm.generate([TokensPrompt(prompt_token_ids=P[sols[j]['q']] + x) for j, x in jobs3], SamplingParams(temperature=0, max_tokens=a.max_new, stop=['\n\n']), use_tqdm=False)
for (j, x), o in zip(jobs3, R3):
    ids, t2 = clean(list(o.outputs[0].token_ids)); p = pred(t2)
    s = sols[j]; s['rta_len'] = len(x) + len(ids); s['rta_ok'] = ok(p, gold[s['q']]) if p is not None else s['ok']
json.dump({'src': a.src, 'n_problems': len(rows), 'gold': gold, 'K': a.K, 'temp': a.temp, 'sols': sols}, open(a.out, 'w'))
gr = [s for s in sols if s['kind'] == 'greedy']
f = lambda L, k: 100 * sum(s.get(k, s['ok']) if k != 'ok' else s['ok'] for s in L) / max(1, len(L))
print(f"greedy acc {f(gr, 'ok'):.2f}  greedy+rethink@#### {100*sum(s.get('rt_ok', s['ok']) for s in gr)/len(gr):.2f}  "
      f"after-answer {100*sum(s.get('rta_ok', s['ok']) for s in gr)/len(gr):.2f} cap150 {100*sum(s.get('cap150_ok', s['ok']) for s in gr)/len(gr):.2f} cap300 {100*sum(s.get('cap300_ok', s['ok']) for s in gr)/len(gr):.2f} "
      f"samples acc {f([s for s in sols if s['kind']=='sample'], 'ok'):.2f}  n_sols {len(sols)}")
