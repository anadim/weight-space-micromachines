"""Collect call-form ladder test results from both hosts (rsync of *_test.out / *_basehost.out / steps.jsonl) into a table."""
import os, re, json, glob, subprocess, ast, collections
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.environ.get('CF_RESULTS', os.path.join(HERE, '..', 'results', 'wiring', 'callform'))   # released copies of the test logs + table.json
# optional: pull results from the training machines first. CF_REMOTES = space-separated rsync sources, e.g.
# "gpu-box-1:/path/to/rl/wiring/runs/cf/ gpu-box-2:/path/to/rl/wiring/runs/cf/" (unset: use the local cf_results/ only)
for src in os.environ.get('CF_REMOTES', '').split():
    subprocess.run(['rsync', '-aq', '--timeout=60', '--include=L300cf_*_test.out', '--include=L300cf_*_basehost.out', '--include=L300cf_*/',
                    '--include=steps.jsonl', '--exclude=*', src, D + '/'], timeout=300)
def tests(f):
    out = {}
    if not os.path.exists(f): return out
    for l in open(f):
        m = re.match(r'^(test|hard|dev) (\w+) (\{.*\})', l)
        if m: out[(m.group(1), m.group(2))] = ast.literal_eval(m.group(3))['acc']
    return out
rows = []
for d in sorted(glob.glob(os.path.join(D, 'L300cf_*'))):
    if not os.path.isdir(d): continue
    n = os.path.basename(d)
    if n.endswith(('_test', '_basehost')) or 'oom' in n or 'crash' in n: continue
    L = [json.loads(l) for l in open(os.path.join(d, 'steps.jsonl'))] if os.path.exists(os.path.join(d, 'steps.jsonl')) else []
    dev = {}
    for r in L:
        if 'dev' in r: dev[r['dev']['u']] = r['dev']['dev_acc']
    last_u = max([r['u'] for r in L if 'reward' in r] or [0])
    t = tests(os.path.join(D, n + '_test.out')); b = tests(os.path.join(D, n + '_basehost.out'))
    mx = [(r['module_exact']['u'], r['module_exact']['exact']) for r in L if 'module_exact' in r]
    rows.append(dict(run=n, u=last_u, done=any('DONE_RUN' in r for r in L), dev300=dev.get(300),
                     gsm_on=t.get(('test', 'full')), gsm_off=t.get(('test', 'hostonly')), gsm_hand=t.get(('test', 'cw_hand')),
                     hard_on=t.get(('hard', 'full')), hard_off=t.get(('hard', 'hostonly')), hard_hand=t.get(('hard', 'cw_hand')),
                     base_gsm_on=b.get(('test', 'full')), base_hard_on=b.get(('hard', 'full')), exact=mx[-1] if mx else None))
f = lambda x: '' if x is None else (f'{x:.1f}' if isinstance(x, float) else str(x))
cols = ['run', 'u', 'dev300', 'gsm_on', 'gsm_off', 'gsm_hand', 'hard_on', 'hard_off', 'hard_hand', 'base_gsm_on', 'base_hard_on']
print(' | '.join(cols)); [print(' | '.join(f(r[c]) for c in cols)) for r in rows]
json.dump(rows, open(os.path.join(D, 'table.json'), 'w'), indent=1)
