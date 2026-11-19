"""Score the NL-control checkpoints exactly like ../../ft100/scripts/score_ft100.py (official BBH norm for real BBH; team
harness 'ok' for own sets), at the eval budget used (A 1000 / B 700 new tokens) and re-scored at the ORIGINAL ft100 budget
(A 600 / B 400) by truncating each greedy output to that many tokens and re-extracting (greedy => prefix of the longer run).
Aggregates only. Writes out/scores_nl.json. Needs the full outputs (it re-tokenizes the text)."""
import json, re, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
from transformers import AutoTokenizer
R = os.environ.get('BFT_R', os.path.join(HERE, '..', 'work'))          # work dir of make_jobs_nl.py (out/*.json)
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')


def norm_off(x, task):   # verbatim official norm (score_ft100.py)
    x = x.strip().strip('.').strip()
    if re.fullmatch(r'\(?[A-R]\)?(\s.*)?', x) and task not in ('word_sorting', 'dyck_languages'): return '(' + re.match(r'\(?([A-R])', x).group(1) + ')'
    if task in ('multistep_arithmetic_two', 'object_counting'): m = re.search(r'-?\d+', x); return m.group() if m else x
    return x.lower()


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


BN = {'temporal': 'temporal_sequences', 'navigate': 'navigate', 'lies': 'web_of_lies'}


def trunc(out, k):
    ids = tok(out, add_special_tokens=False)['input_ids']
    return out if len(ids) <= k else tok.decode(ids[:k]), len(ids)


res = {}
for team, cap in [('A', 600), ('B', 400)]:
    for c in ['n32_s0', 'n100_s0', 'n100_s1', 'n300_s0']:
        p = f'{R}/out/{team}_nl_{c}.json'
        if not os.path.exists(p): continue
        rows = json.load(open(p))['rows']; agg = {}
        for r in rows:
            if team == 'A':
                t = r['task']; b = 'oos' if t.startswith('oos2_') else 'test' if re.fullmatch(r'(track|order)[357]', t) else 'bbh'
                task = t
            else:
                task, kind = r['set'].split('_', 1); b = kind; task = BN.get(task, task) if kind == 'bbh' else task
            short, ntok = trunc(r['out'], cap); ps = extract(short)
            if b == 'bbh':
                ok = norm_off(r['pred'], task) == norm_off(r['gold'], task); ok2 = norm_off(ps, task) == norm_off(r['gold'], task)
            else:
                ok = bool(r['ok'])
                ok2 = (normA(ps) == normA(r['gold'])) if team == 'A' else (normB(ps, task) == normB(r['gold'], task))
            s = agg.setdefault(b, {}).setdefault(task, [0, 0, 0, 0, 0])
            s[0] += 1; s[1] += ok; s[2] += ok2; s[3] += ntok > cap; s[4] += '<tool_call>' in r['out']
        out = {}
        for b, tt in agg.items():
            n = sum(v[0] for v in tt.values())
            out[b] = {'n': n, 'acc': round(100 * sum(v[1] for v in tt.values()) / n, 1), f'acc_at_{cap}': round(100 * sum(v[2] for v in tt.values()) / n, 1),
                      f'pct_over_{cap}': round(100 * sum(v[3] for v in tt.values()) / n, 1), 'call_pct': round(100 * sum(v[4] for v in tt.values()) / n, 1),
                      'per_task': {t: round(100 * v[1] / v[0], 1) for t, v in tt.items()}}
        res.setdefault(team, {})[c] = out
        print(team, c, {b: (v['acc'], v[f'acc_at_{cap}'], v[f'pct_over_{cap}'], v['n']) for b, v in out.items()}, flush=True)
json.dump(res, open(f'{R}/out/scores_nl.json', 'w'), indent=1)
