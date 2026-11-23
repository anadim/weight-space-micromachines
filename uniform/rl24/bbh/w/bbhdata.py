"""BBH look-alike data for rl24/bbh (task [24]/[26]): the teams' generators (envs_ref.py = the reference RL library's mixed-task envs.py,
unchanged) at FRESH seeds (train 200, dev 400; never 0-4 / 100-102), the 12 algorithmic tool tasks.
Differences to envs.py Env('bbhcall').rows (documented):
  * prompt = the BBH 3-shot CoT prompt of the MATCHING BBH task (track5 -> tracking_shuffled_objects_five_objects, ...)
    + "Q: ..\\nA: Let's think step by step." = exactly the BBH-27 harness format (envs.py uses the three_objects prompt for all n);
  * items whose input equals a BBH test input (any of the 27 task files; whitespace-normalized) are dropped (decontamination);
  * dev drops any item that is also in train; Team C generation has an attempt cap (small spaces).
python bbhdata.py OUTDIR BBH_DIR [n_train_per_task n_dev_per_task]
"""
import os, sys, json, random, glob
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('WSM_REPO_ROOT', os.environ.get('WSM_ROOT') or os.path.abspath(os.path.join(HERE, '..', '..', '..', '..')))   # repo root
os.environ.setdefault('WSM_BBH_GENERIC', os.path.join(os.environ['WSM_REPO_ROOT'], 'bbh_generic'))
sys.path.insert(0, HERE)
import envs_ref as E

G = E.G
BBH_TASK = {'track3': 'tracking_shuffled_objects_three_objects', 'track5': 'tracking_shuffled_objects_five_objects',
            'track7': 'tracking_shuffled_objects_seven_objects', 'order3': 'logical_deduction_three_objects',
            'order5': 'logical_deduction_five_objects', 'order7': 'logical_deduction_seven_objects',
            'temporal': 'temporal_sequences', 'navigate': 'navigate', 'lies': 'web_of_lies',
            'multistep_arithmetic_two': 'multistep_arithmetic_two', 'boolean_expressions': 'boolean_expressions',
            'object_counting': 'object_counting'}
TASKS = list(BBH_TASK)
COTDIR = {'track': 'teamA', 'order': 'teamA', 'temporal': 'teamB', 'navigate': 'teamB', 'lies': 'teamB'}


def cot_file(task):
    for d in ('teamA', 'teamB', 'teamC'):
        p = os.path.join(G, d, 'cot', BBH_TASK[task] + '.txt')
        if os.path.exists(p): return p
    raise FileNotFoundError(task)


def gen_C(seed, n, cap=200000):
    E._path(os.path.join(G, 'teamC')); import gen_tasks as GT
    rows = []
    for t in E.TASKS_C:
        banned = set()
        for f in glob.glob(os.path.join(G, 'teamC', 'data', '*', f'{t}.json')):
            banned |= {e['input'] for e in json.load(open(f))['examples']}
        r1 = random.Random(f'{t}-1')
        for _ in range(100):
            res = GT.TASKS[t](r1, 'test'); banned.add(res if t == 'multistep_arithmetic_two' else res[0])
        r = random.Random(f'{t}-{seed}'); seen = set(); k = 0; tries = 0
        while k < n and tries < cap:
            tries += 1
            res = GT.TASKS[t](r, 'test')
            inp, tgt = (res, str(eval(res[:-2]))) if t == 'multistep_arithmetic_two' else res
            if inp in seen or inp in banned: continue
            seen.add(inp); k += 1
            rows.append({'fam': 'C', 'task': t, 'q': inp, 'gold': tgt})
    return rows


def build(seed, n, bbh_inputs, exclude=frozenset()):
    E.SEED['X'] = seed
    rows = [dict(r) for r in E._gen_A('X', n)] + [dict(r) for r in E._gen_B('X', n)] + gen_C(seed, n)
    norm = lambda s: ' '.join(s.split())
    out, drop = [], {'bbh_test': 0, 'excluded': 0}
    for r in rows:
        if norm(r['q']) in bbh_inputs: drop['bbh_test'] += 1; continue
        if r['q'] in exclude: drop['excluded'] += 1; continue
        r['bbh_task'] = BBH_TASK[r['task']]
        r['prompt'] = E.cot(cot_file(r['task'])) + E.QA(r['q'])
        r['seed'] = seed
        out.append(r)
    return out, drop


if __name__ == '__main__':
    od, bbh = sys.argv[1], sys.argv[2]
    ntr = int(sys.argv[3]) if len(sys.argv) > 3 else 1000; ndv = int(sys.argv[4]) if len(sys.argv) > 4 else 34
    os.makedirs(od, exist_ok=True)
    bi = set()
    for f in glob.glob(os.path.join(bbh, '*.json')):
        bi |= {' '.join(e['input'].split()) for e in json.load(open(f))['examples']}
    print('bbh test inputs', len(bi))
    from collections import Counter
    tr, dtr = build(200, ntr, bi)
    dv, ddv = build(400, ndv + 10, bi, exclude={r['q'] for r in tr})
    by = {}
    for r in dv: by.setdefault(r['task'], []).append(r)
    dv = [r for t in TASKS for r in by.get(t, [])[:ndv]]
    for nm, rows, d in (('train', tr, dtr), ('dev', dv, ddv)):
        with open(os.path.join(od, f'{nm}.jsonl'), 'w') as f:
            for r in rows: f.write(json.dumps(r) + '\n')
        print(nm, len(rows), dict(Counter(r['task'] for r in rows)), 'dropped', d, flush=True)
    # cot copies vs the official BBH cot files (the BBH-27 harness reads BBH_COT)
    cdir = os.environ.get('BBH_COT')
    if cdir:
        for t in TASKS:
            a = E.cot(cot_file(t)); b = E.cot(os.path.join(cdir, BBH_TASK[t] + '.txt'))
            print('cot', t, 'identical' if a == b else 'DIFFERENT')
