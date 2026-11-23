"""[32](e) [28] 1.7B copy of uniform/final_eval/q32/bbh_pertask.py. BBH per-task sanity check: every task where row 4 (host + tools + gate + RL, 3 seeds) < row 1 (host, tools off) by > 3 points.
Paired item bootstrap per task (10,000 resamples; row 4 = per-item mean over the 3 seeds). Also prints row 3 and the RL-only row 2 per task.
usage: python3 uniform/rl28/bbh/w/pertask.py [EVAL_DIR]   (default uniform/rl28/bbh/res: reduced rows eval_<cfg>.json.gz; CPU)"""
import json, gzip, os, sys, random
from collections import defaultdict

W = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'res')
ROWS = {'0': ['base17_plain'], '1': ['ft17_t0_u0'], '2': ['F17_s0_u300', 'F17_s1_u300', 'F17_s2_u300'], '3': ['ft17_tools_gate'],
        '4': ['I17_s0_u300', 'I17_s1_u300', 'I17_s2_u300']}
TOOL = {'boolean_expressions', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'logical_deduction_three_objects',
        'multistep_arithmetic_two', 'navigate', 'object_counting', 'temporal_sequences', 'tracking_shuffled_objects_five_objects',
        'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects', 'web_of_lies'}


def load(cfgs):
    per = None
    for c in cfgs:
        p = os.path.join(W, f'eval_{c}.json')
        R = json.load(gzip.open(p + '.gz', 'rt') if os.path.exists(p + '.gz') else open(p))['rows']
        if per is None:
            per = [[r['task'], float(r['ok'])] for r in R]
        else:
            assert len(R) == len(per) and all(r['task'] == p[0] for r, p in zip(R, per))
            for r, p in zip(R, per):
                p[1] += float(r['ok'])
    return [(t, v / len(cfgs)) for t, v in per]


D = {k: load(v) for k, v in ROWS.items()}
by = defaultdict(list)
for i, (t, _) in enumerate(D['1']):
    by[t].append(i)
rng = random.Random(0)
out = []
for t, idx in sorted(by.items()):
    m = {k: 100 * sum(D[k][i][1] for i in idx) / len(idx) for k in D}
    d = [D['4'][i][1] - D['1'][i][1] for i in idx]
    bs = sorted(100 * sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(10000))
    out.append((t, len(idx), m, m['4'] - m['1'], bs[249], bs[9749]))
print('| task | n | tool? | 0 Base | 1 host | 2 +RL | 3 +tools | 4 +tools+RL | 4 − 1 [95% CI] |')
print('|---|---|---|---|---|---|---|---|---|')
for t, n, m, dd, lo, hi in out:
    flag = ' **< −3**' if dd < -3 else ''
    print(f"| {t} | {n} | {'tool' if t in TOOL else ''} | {m['0']:.1f} | {m['1']:.1f} | {m['2']:.1f} | {m['3']:.1f} | {m['4']:.1f} | "
          f"{dd:+.1f} [{lo:+.1f}, {hi:+.1f}]{flag} |")
neg = [o for o in out if o[3] < -3]
print(f"\ntasks with 4 − 1 < −3: {len(neg)} ({sum(o[0] in TOOL for o in neg)} tool); CI excludes 0 for: "
      + ', '.join(o[0] for o in neg if o[5] < 0))
