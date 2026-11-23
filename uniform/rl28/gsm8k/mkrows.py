"""[28] rows.json for GSM8K on Qwen3-1.7B-Base (schema of rl24/uniform.json panels; row CI = make_uniform_json.py method:
seed-averaged per item, 2000 item resamples, seed 0, 95% percentile). usage: python3 mkrows.py [SEEDS e.g. 0,1,2]"""
import json, os, random, sys
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import items   # uniform/items.py: reduced per-item rows (bundles) or your own full rows via $FE_ROWS
def _have(n): return all(items.have(st, n) for st in ('test', 'hard'))
S = sys.argv[1].split(',') if len(sys.argv) > 1 else [str(s) for s in range(3) if _have(f'r2_CW17_s{s}')]   # default: seeds present
SP = sys.argv[2].split(',') if len(sys.argv) > 2 else [str(s) for s in range(3) if _have(f'P17_s{s}')]


def _q32b(st):   # names mk32.py wrote to fe/lists/q32b_<st>.list (the original runs: the scH files of res/)
    p = os.path.join(H, 'fe', 'lists', f'q32b_{st}.list')
    return [l.split()[0] for l in open(p) if l.strip() and not l.startswith('#')] if os.path.exists(p) else []


def load(st, n):
    return {r['i']: r for r in items.load(st, n)}


def row(st, names):
    R = [load(st, n) for n in names]
    ids = sorted(R[0]); assert all(sorted(r) == ids for r in R)
    per = [100 * sum(r[i]['ok'] for i in ids) / len(ids) for r in R]
    v = [sum(float(r[i]['ok']) for r in R) / len(R) for i in ids]
    random.seed(0); n = len(v)
    bs = sorted(100 * sum(v[random.randrange(n)] for _ in range(n)) / n for _ in range(2000))
    tok = sum(r[i].get('ntok', 0) for r in R for i in ids) / (len(R) * n)
    return dict(mean=round(sum(per) / len(per), 2), seeds=[round(p, 2) for p in per], ci95=[round(bs[50], 2), round(bs[1949], 2)],
                tokq=round(tok, 1))


GROWS = [('1', 'host', ['base17H']), ('2', 'host + RL', ['P17_s%s' % s for s in SP]), ('3', 'host + implants', ['r2_17']),
         ('4', 'host + implants + RL', ['r2_CW17_s%s' % s for s in S])]
out = {'host': 'Qwen3-1.7B-Base', 'metric': 'accuracy (%)', 'sets': {},
       'notes': '[28] uniform 4-row protocol on Qwen3-1.7B-Base (same implants as 0.6B; readers/writers refit, verifier probe and stop '
                'row refit on 1.7B train/dev600 data; circuits unchanged). T=1 seed 7, once per config. RL: GRPO 300 u, 64 prompts x 8, '
                'lr 1e-6, T=1, train only. Row 3/4 = rung 2 (stk_r2q_17, SW_DMARG=40). See uniform/rl28/gsm8k/RESULTS.md.'}
for st, lab in [('test', 'GSM8K test (1319)'), ('hard', 'GSM-Hard (1319)')]:
    rows = []
    sch = {n.split('_', 1)[1]: n for n in _q32b(st) if items.have(st, n)}   # host -> scH<N>_<host> (the [32](b) configs)
    extra = [('2-MV', '[32](b) host + RL, N samples + majority (tok/q matched to row 4)', [sch.get(f'P17_s{s}', '-') for s in range(3)]),
             ('1-MV', '[32](b) host, N samples + majority (same N)', [sch.get('base17', '-')]),
             ('1-seeds', '[32](c) host, eval seeds 7/11/13', ['base17H', 'base17H_s11', 'base17H_s13']),
             ('3-seeds', '[32](c) host + implants, eval seeds 7/11/13', ['r2_17', 'r2_17_s11', 'r2_17_s13'])]
    for k, l, n in GROWS + extra:
        if n and all(items.have(st, x) for x in n):
            rows.append(dict(row=k, label=l, configs=n, **row(st, n)))
    out['sets'][st] = {'label': lab, 'rows': rows}
json.dump(out, open(os.path.join(H, 'rows.json'), 'w'), indent=1)
for st in out['sets']:
    for r in out['sets'][st]['rows']: print(st, r['row'], r['mean'], r['seeds'], r['ci95'], r['tokq'])
