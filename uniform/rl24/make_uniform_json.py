"""[26] combined JSON for the 3-panel bar figure (GSM8K | IFEval | BBH), 4 rows each.
GSM8K is computed from the per-item eval rows (uniform/items.py: the reduced bundles in uniform/final_eval/items, or your own
full rows via FE_ROWS=uniform/final_eval/out/fe); IFEval / BBH are read from uniform/rl24/{ifeval,bbh}/rows.json (same schema: {"rows": [{"row", "label", "metric", "mean", "seeds", "ci95", "tokq"}], ...}).
usage: python3 uniform/rl24/make_uniform_json.py  ->  uniform/rl24/uniform.json"""
import json, os, random, sys
W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(W))
import items


def load(st, n):
    return {r['i']: r for r in items.load(st, n)}


def row(st, names):
    R = [load(st, n) for n in names]
    ids = sorted(R[0])
    per = [100 * sum(r[i]['ok'] for i in ids) / len(ids) for r in R]
    v = [sum(float(r[i]['ok']) for r in R) / len(R) for i in ids]
    random.seed(0)
    n = len(v)
    bs = sorted(100 * sum(v[random.randrange(n)] for _ in range(n)) / n for _ in range(2000))
    tok = sum(r[i].get('ntok', 0) for r in R for i in ids) / (len(R) * n)
    return dict(mean=round(sum(per) / len(per), 2), seeds=[round(p, 2) for p in per], ci95=[round(bs[50], 2), round(bs[1949], 2)],
                tokq=round(tok, 1))


GROWS = [
    ('1', 'host', ['baseH']),
    ('2', 'host + RL', ['P64_s0', 'P64_s1', 'P64_s2']),
    ('3', 'host + implants', ['r2_base']),
    ('4', 'host + implants + RL', ['r2_CW_s0', 'r2_CW_s1', 'r2_CW_s2']),
]
out = {'host': 'Qwen3-0.6B-Base (BBH rows: + §6.1 call-FT, [29])', 'decoding': 'T=1 seed 7, once per config',
       'rl': 'GRPO 300 u, 64 prompts x 8, lr 1e-6, T=1, train data only', 'panels': {}}
g = {'metric': 'accuracy (%)', 'sets': {}}
for st, lab in [('test', 'GSM8K test (1319)'), ('hard', 'GSM-Hard (1319)')]:
    g['sets'][st] = {'label': lab, 'rows': [dict(row=k, label=l, configs=n, **row(st, n)) for k, l, n in GROWS]}
out['panels']['gsm8k'] = g
for t in ('ifeval', 'bbh'):
    p = os.path.join(W, t, 'rows.json')
    out['panels'][t] = json.load(open(p)) if os.path.exists(p) else {'pending': True}
json.dump(out, open(os.path.join(W, 'uniform.json'), 'w'), indent=1)
print(json.dumps(out['panels']['gsm8k'], indent=None)[:1500])
