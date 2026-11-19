"""v1 vs v2 side by side (aggregates only for real BBH). python report_v2.py OUT_DIR
v1: sets_{base_notools,ft_notools,ft_tools}.json (+ var3_*), b27_{base_notools,ft_notools,ft_tools}[_T*].json
v2: sets_v2ft_{notools,tools}.json, b27_v2ft_{notools,tools}_P*.json   (base = v1 base, same model and harness)"""
import json, re, os, sys, glob, collections
O = sys.argv[1]; HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..')); from reduce_bbh import load_json     # reads plain or gzipped results
# The shipped results (../results/*.json.gz) are reduced: no generated text. Call rates then come from n_calls, and the
# composition per-tool analysis (which reads the text) is skipped; its full-output tables are in ../TABLES_v1_vs_v2.md.
hascall = lambda r: ('<tool_call>' in r['out']) if 'out' in r else r.get('n_calls', 0) > 0
pct = lambda v: f'{100 * sum(v) / len(v):5.1f}' if v else '  -  '


def L(f):
    for p in (f'{O}/{f}', f'{O}/{f}.gz'):
        if os.path.exists(p): return load_json(p)
    return None


def sets(name, extra=None):
    d = L(name)
    if d and extra and L(extra): d['rows'] += L(extra)['rows']
    return d


S = {'base': sets('sets_base_notools.json', 'var3_base_notools.json'), 'v1n': sets('sets_ft_notools.json', 'var3_ft_notools.json'),
     'v1t': sets('sets_ft_tools.json', 'var3_ft_tools.json'), 'v2n': sets('sets_v2ft_notools.json'), 'v2t': sets('sets_v2ft_tools.json')}
COLS = [('base', 'base'), ('v1n', 'v1 FT no tools'), ('v1t', 'v1 FT + tools'), ('v2n', 'v2 FT no tools'), ('v2t', 'v2 FT + tools')]
keys = []
for k in ('base', 'v1t', 'v2t'):
    for r in (S[k]['rows'] if S[k] else []):
        if (r['set'], r['task']) not in keys: keys.append((r['set'], r['task']))
print('## Team sets: v1 vs v2\n')
print('| set | task | n | ' + ' | '.join(n for _, n in COLS) + ' | call v1 | call v2 |'); print('|---|---|---|' + '---|' * (len(COLS) + 2))
rows_of = lambda k, key: [r for r in (S[k]['rows'] if S[k] else []) if (r['set'], r['task']) == key]
for key in keys:
    if key[0] == 'compose': continue
    cells = []
    for k, _ in COLS:
        rr = rows_of(k, key); c = pct([r['ok'] for r in rr])
        if key[0] == 'C_ood' and rr: c += f' / {pct([r["ok_lenient"] for r in rr])}'
        cells.append(c)
    print(f'| {key[0]} | {key[1]} | {len(rows_of("base", key)) or len(rows_of("v2t", key))} | ' + ' | '.join(cells) +
          f' | {pct([r["called"] for r in rows_of("v1t", key)])} | {pct([r["called"] for r in rows_of("v2t", key)])} |')
for s in ('A_test', 'A_oos2', 'A_var3', 'B_test', 'B_oos', 'C_test', 'C_ood'):
    cells = []
    for k, _ in COLS:
        by = {}
        for r in (S[k]['rows'] if S[k] else []):
            if r['set'] == s: by.setdefault(r['task'], []).append(r['ok'])
        cells.append(f'{sum(100 * sum(v) / len(v) for v in by.values()) / len(by):5.1f}' if by else '  -  ')
    tot = [f"{sum(r['ok'] for r in S[k]['rows'] if r['set'] == s)}/{sum(1 for r in S[k]['rows'] if r['set'] == s)}" if S[k] else '-' for k, _ in COLS]
    print(f'| **{s}** | task mean (raw) | | ' + ' | '.join(f'{c} ({t})' for c, t in zip(cells, tot)) + ' | | |')

# ---------------- composition ----------------
comp = {json.loads(l)['id']: json.loads(l) for l in open(f'{HERE}/../data/compose.jsonl')}
CALL = re.compile(r'<tool_call>(\w+)(.*?)</tool_call>(?:<tool_response>(.*?)</tool_response>)?', re.S)


def resp_ok(tool, resp, acc):
    if resp is None: return False
    resp = resp.strip()
    if tool == 'walk': resp = resp.split(' facing')[0]
    if tool in ('expr', 'calc'):
        try: return any(abs(float(resp) - float(x)) < 1e-6 for x in acc)
        except ValueError: return False
    return resp in acc


def passive_used(out, tool, acc):
    """v2: the C tools are passive. 'Used' = the host wrote an arithmetic expression (>= 1 binary operator between numbers) or a
    True/False formula (>= 1 of not/and/or) followed by '='. expr = integer-only left side, calc = left side with '.' or '/'.
    'Correct' = the value after '=' equals the gold intermediate value (with tools attached this value is the passive tool's write;
    in the no-tools arm it is the host's own)."""
    t = out.split('\n\nQ:')[0]
    if tool == 'bool':
        m = re.findall(r'((?:\(|\)|\s|True|False|not|and|or)+)=\s*(True|False)', t)
        m = [(l, v) for l, v in m if re.search(r'\b(not|and|or)\b', l) and re.search(r'True|False', l)]
        return bool(m), any(v in acc for _, v in m)
    m = re.findall(r'((?:[\d().]+\s*[-+*/×x]\s*)+[\d().]+)\s*=\s*(-?\d+(?:\.\d+)?)', t)
    m = [(l, v) for l, v in m if (('.' in l or '/' in l) if tool == 'calc' else not ('.' in l or '/' in l))]
    return bool(m), any(abs(float(v) - float(x)) < 1e-6 for _, v in m for x in acc)


print('\n## Composition test (45 frozen questions): v1 vs v2\n')
print('| arm | answer strict | lenient | every needed tool used | any A/B call |'); print('|---|---|---|---|---|')
PT = {}
for k, nm in COLS:
    if not S[k]: continue
    rr = [r for r in S[k]['rows'] if r['set'] == 'compose']; ev, anyc = [], []
    if rr and 'out' not in rr[0]:
        print(f'| {nm} | {pct([r["ok"] for r in rr])} | {pct([r["ok_lenient"] for r in rr])} | (needs full outputs) | {pct([hascall(r) for r in rr])} |'); continue
    for r in rr:
        q = comp[r['id']]; o = r['out'].split('\n\nQ:')[0]; calls = CALL.findall(o); names = {c[0] for c in calls}
        used = []
        for t in q['tools']:
            d = PT.setdefault((k, t), [0, 0, 0]); d[0] += 1
            if t in ('expr', 'bool', 'calc') and k.startswith('v2'):
                u, good = passive_used(o, t, q['accept'][t])
            else:
                mine = [c for c in calls if c[0] == t]; u, good = bool(mine), any(resp_ok(t, c[2], q['accept'][t]) for c in mine)
            d[1] += u; d[2] += good; used.append(u)
        ev.append(all(used)); anyc.append(bool(calls))
    print(f'| {nm} | {pct([r["ok"] for r in rr])} | {pct([r["ok_lenient"] for r in rr])} | {pct(ev)} | {pct(anyc)} |')
print('\nPer family, answer strict (v1 FT+tools / v2 FT+tools)')
print('| family | v1 | v2 |'); print('|---|---|---|')
for f in sorted({c['family'] for c in comp.values()}):
    print(f'| {f} | ' + ' | '.join(pct([r['ok'] for r in S[k]['rows'] if r['set'] == 'compose' and r['task'] == f]) if S[k] else '-' for k in ('v1t', 'v2t')) + ' |')
print('\nPer-tool use on composition (needed / used / correct value). v2 expr/bool/calc = passive: "used" = an `EXPR =` was written')
print('| tool | v1 FT+tools | v2 FT+tools | v2 FT no tools (host-written values) |'); print('|---|---|---|---|')
for t in ('track', 'order', 'interval', 'walk', 'truth', 'expr', 'bool', 'calc'):
    print(f'| {t} | ' + ' | '.join('/'.join(map(str, PT.get((k, t), [0, 0, 0]))) for k in ('v1t', 'v2t', 'v2n')) + ' |')

# ---------------- real BBH ----------------
TOOL_TASKS = ['tracking_shuffled_objects_three_objects', 'tracking_shuffled_objects_five_objects', 'tracking_shuffled_objects_seven_objects',
              'logical_deduction_three_objects', 'logical_deduction_five_objects', 'logical_deduction_seven_objects',
              'temporal_sequences', 'navigate', 'web_of_lies', 'multistep_arithmetic_two', 'boolean_expressions', 'object_counting']


def bbh(prefix):
    d = L(f'{prefix}.json')
    if d: return d
    parts = [load_json(f) for f in sorted(glob.glob(f'{O}/{prefix}_[TP]*.json') + glob.glob(f'{O}/{prefix}_[TP]*.json.gz'))]
    if not parts: return None
    return {'per_task': {k: v for p in parts for k, v in p['per_task'].items()}, 'rows': [r for p in parts for r in p['rows']]}


Bb = {'base': bbh('b27_base_notools'), 'v1n': bbh('b27_ft_notools'), 'v1t': bbh('b27_ft_tools'), 'v2n': bbh('b27_v2ft_notools'), 'v2t': bbh('b27_v2ft_tools')}
tasks = sorted(Bb['base']['per_task'])
print('\n## Real BBH, 27 tasks, one run per arm (official prompts and scoring; scores only)\n')
print('| task | tool task | ' + ' | '.join(n for _, n in COLS) + ' | call v1 | call v2 |'); print('|---|---|' + '---|' * (len(COLS) + 2))
cr = lambda k, t: pct([hascall(r) for r in Bb[k]['rows'] if r['task'] == t]) if Bb[k] else '  -  '
for t in tasks:
    cells = [f"{Bb[k]['per_task'][t]:5.1f}" if Bb[k] and t in Bb[k]['per_task'] else '  -  ' for k, _ in COLS]
    print(f'| {t} | {"yes" if t in TOOL_TASKS else ""} | ' + ' | '.join(cells) + f' | {cr("v1t", t)} | {cr("v2t", t)} |')
for lab, sel in (('27-task mean', tasks), ('12 tool-task mean', [t for t in tasks if t in TOOL_TASKS]), ('15 no-tool-task mean', [t for t in tasks if t not in TOOL_TASKS])):
    cells = [f"{sum(Bb[k]['per_task'][t] for t in sel) / len(sel):5.2f}" if Bb[k] and all(t in Bb[k]['per_task'] for t in sel) else '  -  ' for k, _ in COLS]
    calls = []
    for k in ('v1t', 'v2t'):
        rr = [r for r in (Bb[k]['rows'] if Bb[k] else []) if r['task'] in sel]; calls.append(pct([hascall(r) for r in rr]))
    print(f'| **{lab}** | | ' + ' | '.join(cells) + f' | {calls[0]} | {calls[1]} |')
for k in ('v1n', 'v2n'):
    if Bb[k]:
        rr = [r for r in Bb[k]['rows'] if r['task'] not in TOOL_TASKS]
        print(f'call rate on the 15 no-tool tasks, {k}: {pct([hascall(r) for r in rr])}')
for k in ('v1t', 'v2t'):
    if not Bb[k]: continue
    c = collections.Counter()
    for r in Bb[k]['rows']:
        if r['task'] in TOOL_TASKS: continue
        if 'out' not in r: c.update(r.get('tools', [])); continue          # reduced rows: names over the whole output
        for nm, _, _ in CALL.findall(r['out'].split('\n\nQ:')[0]): c[nm] += 1
    print(f'{k}: calls by name on the 15 no-tool tasks (top 8): ' + ', '.join(f'{a} {b}' for a, b in c.most_common(8)))
