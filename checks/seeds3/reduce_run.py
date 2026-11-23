"""Reduce one seeds3 run dir (run_seed.sh output, $WSM_DATA/runs/NAME) to the shipped res/ layout:
    python3 checks/seeds3/reduce_run.py RUN_DIR NAME [DEST]     DEST default: res/NAME (m4/m2) or res/rlctl/NAME (the rest)
steps.jsonl keeps every record except per-update detail (per-update lines are cut to the scalars below) and the duplicate
test records (they are in test_eval.json); absolute paths in config.json / test_eval.json become basenames. W3d runs: the
test eval is a second call writing to RUN_DIR/../TEST_NAME/test_eval.json (run_seed.sh), read from there."""
import json, os, sys

H = os.path.dirname(os.path.abspath(__file__))
src, name = sys.argv[1], sys.argv[2]
dst = sys.argv[3] if len(sys.argv) > 3 else os.path.join(H, 'res', *((name,) if name[:2] in ('m4', 'm2') else ('rlctl', name)))
os.makedirs(dst, exist_ok=True)
UK = ('u', 't', 'reward', 'fmt', 'r_gsm', 'r_scaled', 'kept_groups', 'len', 'writes', 'calls', 'n_veto', 'p_fire',
      'roll_s', 'upd_s', 'mem_gb', 'prompts')
DROP = ('mixed_test', 'test', 'resume_saved', 'resumed_from')


def rd(x):
    if isinstance(x, float): return round(x, 4)
    if isinstance(x, dict): return {k: rd(v) for k, v in x.items()}
    if isinstance(x, list): return [rd(v) for v in x]
    return x


def base(v):
    if isinstance(v, dict): return {k: base(w) for k, w in v.items()}
    if isinstance(v, list): return [base(w) for w in v]
    return os.path.basename(v.rstrip('/')) if isinstance(v, str) and v.startswith('/') else v


with open(os.path.join(dst, 'steps.jsonl'), 'w') as g:
    for l in open(os.path.join(src, 'steps.jsonl')):
        if not l.strip(): continue
        x = json.loads(l)
        if 'calls' in x and 'u' in x and 'roll_s' in x:
            x = {k: x[k] for k in UK + (() if 'upd_s' in x else ('train_s',)) if k in x}   # wiring runs: no upd_s
        elif any(k in x for k in DROP):
            continue
        g.write(json.dumps(base(rd(x)), separators=(',', ':')) + '\n')
c = json.load(open(os.path.join(src, 'config.json')))
json.dump(base(c), open(os.path.join(dst, 'config.json'), 'w'), indent=1)
te = os.path.join(src, 'test_eval.json')
if not os.path.exists(te): te = os.path.join(os.path.dirname(os.path.abspath(src).rstrip('/')), 'TEST_' + name, 'test_eval.json')
json.dump(base(json.load(open(te))), open(os.path.join(dst, 'test_eval.json'), 'w'), separators=(',', ':'))
print('wrote', dst)
