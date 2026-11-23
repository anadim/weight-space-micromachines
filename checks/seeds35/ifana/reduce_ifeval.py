"""Reduce one IFEval run dir (as written by ifeval_seed.sh or uniform/rl24/ifeval/rl_chain.sh) to the shipped layout:
    python3 ifana/reduce_ifeval.py SRC_DIR RUN [DEST_ROOT]     e.g.  python3 ifana/reduce_ifeval.py res_new/I_s3 I_s3
writes DEST_ROOT/RUN/ (default ../res/ifeval/RUN): eval_*.items.json (score json rows without resp_sha256), steps.jsonl
(config with path basenames, dev records, per-update scalars), eval_u300.summary.json (path basenames), run_facts.json."""
import json, os, sys, glob

HERE = os.path.dirname(os.path.abspath(__file__))
src, run = sys.argv[1], sys.argv[2]
o = os.path.join(sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, '..', 'res', 'ifeval'), run)
os.makedirs(o, exist_ok=True)
UK = ('u', 'reward', 'inst_strict', 'len', 'trunc', 'fired', 'roll_s', 'update_s', 'mem_gb')


def base(v):
    return os.path.basename(v.rstrip('/')) if isinstance(v, str) and v.startswith('/') else v


def rd(x):
    if isinstance(x, float): return round(x, 4)
    if isinstance(x, dict): return {k: rd(v) for k, v in x.items()}
    if isinstance(x, list): return [rd(v) for v in x]
    return x


for p in sorted(glob.glob(os.path.join(src, 'eval_*.score.json'))):
    if '.remote.' in p: continue
    d = json.load(open(p))
    h = {k: d[k] for k in ('source', 'source_sha256', 'scorer', 'n', 'command') if k in d}
    h['rows'] = [{k: r[k] for k in ('key', 'ids', 'strict', 'loose', 'finish', 'ntok', 'circ') if k in r} for r in d['rows']]
    json.dump(h, open(os.path.join(o, os.path.basename(p).replace('.score.json', '.items.json')), 'w'), separators=(',', ':'))
p = os.path.join(src, 'eval_u300.summary.json')
if os.path.exists(p):
    j = json.load(open(p))
    json.dump({k: base(v) for k, v in j.items()}, open(os.path.join(o, 'eval_u300.summary.json'), 'w'), indent=1)
p = os.path.join(src, 'steps.jsonl')
if os.path.exists(p):
    with open(os.path.join(o, 'steps.jsonl'), 'w') as g:
        for l in open(p):
            if not l.strip(): continue
            x = json.loads(l)
            if 'config' in x: x['config'] = {k: base(v) for k, v in x['config'].items()}
            elif 'roll_s' in x: x = {k: x[k] for k in UK if k in x}
            g.write(json.dumps(rd(x), separators=(',', ':')) + '\n')
f = {}
p = os.path.join(src, 'gpu.txt')
f['gpu'] = open(p).read().strip().split('\n')[-1].strip() if os.path.exists(p) else 'unknown'
for k in ('start', 'train_start', 'train_end', 'end'):
    p = os.path.join(src, k + '.txt')
    if os.path.exists(p): f[k] = open(p).read().strip()
json.dump(f, open(os.path.join(o, 'run_facts.json'), 'w'), indent=1)
print('wrote', o)
