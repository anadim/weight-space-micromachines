"""Stratified subsample of a team's OWN training rows (rows copied verbatim, format unchanged).
Even split over tools, then evenly over that tool's styles; uniform random within each (tool, style) cell.
Team A: cells = (tool, style) of train3.jsonl.  Team B: tool interval={temporal, lab}, walk={navigate, hiker},
truth={lies, rumor}; style = main BBH-like task vs auxiliary setting. Error-recovery rows are kept if they fall in the sample.

python subsample.py A N SEED SRC/train3.jsonl OUT.jsonl
python subsample.py B N SEED SRC_DATA_DIR OUT_DIR      (writes OUT_DIR/{task}_train.jsonl, the layout ft_B.py reads)
"""
import json, random, sys, os, collections
team, n, seed, src, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
rng = random.Random(seed)


def split(n, keys):
    keys = list(keys); base, r = divmod(n, len(keys)); order = keys[:]; rng.shuffle(order)
    return {k: base + (1 if order.index(k) < r else 0) for k in keys}


if team == 'A':
    rows = [json.loads(l) for l in open(src)]
    cells = collections.defaultdict(list)
    for r in rows: cells[(r['tool'], r['style'])].append(r)
    tools = sorted({t for t, _ in cells}); pick = []
    for t, nt in split(n, tools).items():
        styles = sorted(s for tt, s in cells if tt == t)
        for s, ns in split(nt, styles).items(): pick += rng.sample(cells[(t, s)], ns)
    rng.shuffle(pick)
    with open(out, 'w') as f:
        for r in pick: f.write(json.dumps(r) + '\n')
    print(collections.Counter((r['tool'], r['style']) for r in pick))
else:
    T = {'interval': ['temporal', 'lab'], 'walk': ['navigate', 'hiker'], 'truth': ['lies', 'rumor']}
    os.makedirs(out, exist_ok=True); cnt = {}
    for t, nt in split(n, sorted(T)).items():
        for task, ns in split(nt, T[t]).items():
            rows = [json.loads(l) for l in open(f'{src}/{task}_train.jsonl')]
            pick = rng.sample(rows, ns)
            with open(f'{out}/{task}_train.jsonl', 'w') as f:
                for r in pick: f.write(json.dumps(r) + '\n')
            cnt[task] = (ns, sum(1 for r in pick if r.get('masked_prefix')))
    print(cnt, '(rows, of which error-recovery)')
