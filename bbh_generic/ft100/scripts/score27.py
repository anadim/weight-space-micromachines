"""All-27-task table, aggregates only (official norm of peek/scripts/score.py, re-applied to saved predictions).
python score27.py REF_LOG name=path.json ...   (REF_LOG = a base-run log with per-task lines, e.g. from scripts/run_bbh.py --machines none)"""
import json, re, sys


def norm(x, task):   # verbatim from peek/scripts/score.py
    x = x.strip().strip('.').strip()
    if re.fullmatch(r'\(?[A-R]\)?(\s.*)?', x) and task not in ('word_sorting', 'dyck_languages'): return '(' + re.match(r'\(?([A-R])', x).group(1) + ')'
    if task in ('multistep_arithmetic_two', 'object_counting'): m = re.search(r'-?\d+', x); return m.group() if m else x
    return x.lower()


ref = {}
for l in open(sys.argv[1]):
    m = re.match(r'(\w+)\s+([\d.]+) \(n=(\d+)\)', l)
    if m: ref[m.group(1)] = float(m.group(2))
cols = {'base_ref': {t: (v, None) for t, v in ref.items()}}
for arg in sys.argv[2:]:
    name, path = arg.split('=', 1)
    try: rows = json.load(open(path))['rows']
    except FileNotFoundError: continue
    s = {}
    for r in rows:
        x = s.setdefault(r['task'], [0, 0, 0]); x[0] += 1; x[1] += norm(r['pred'], r['task']) == norm(r['gold'], r['task']); x[2] += '<tool_call>' in r['out']
    cols[name] = {t: (100 * v[1] / v[0], 100 * v[2] / v[0]) for t, v in s.items()}
tasks = sorted(ref)
own = {'A': {t for t in tasks if t.startswith(('tracking', 'logical'))}, 'B': {'temporal_sequences', 'navigate', 'web_of_lies'},
       'C': {'multistep_arithmetic_two', 'boolean_expressions', 'object_counting'}}
own['all'] = own['A'] | own['B'] | own['C']
print(f"{'task':40s} " + ' '.join(f'{c[:22]:>22s}' for c in cols))
for t in tasks:
    tag = ''.join(k for k in 'ABC' if t in own[k])
    print(f'{t:37s} {tag:2s} ' + ' '.join(f"{(f'{cols[c][t][0]:5.1f}' + (f' (call {cols[c][t][1]:3.0f})' if cols[c][t][1] else '')) if t in cols[c] else '-':>22s}" for c in cols))
for lab, sel in [('MEAN 27 tasks', set(tasks)), ('mean, 12 tool tasks (A+B+C)', own['all']), ('mean, 15 no-tool tasks', set(tasks) - own['all']),
                 ('mean, A tasks (6)', own['A']), ('mean, B tasks (3)', own['B']), ('mean, C tasks (3)', own['C'])]:
    print(f'{lab:40s} ' + ' '.join(f"{(sum(cols[c][t][0] for t in sel) / len(sel)) if all(t in cols[c] for t in sel) else float('nan'):22.2f}" for c in cols))
