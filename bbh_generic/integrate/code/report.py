"""Tables for the integration step (aggregates only for real BBH: per-task accuracy and call rate; no item is printed).
python report.py OUT_DIR > tables.md
  sets_{base_notools,ft_notools,ft_tools}.json   (eval_integ.py)
  b27_{base_notools,ft_notools,ft_tools}.json    (run27_integ.py)
"""
import json, re, os, sys
O = sys.argv[1]; HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = [('base_notools', 'frozen base'), ('ft_notools', 'integrated FT, no tools'), ('ft_tools', 'integrated FT + all 8 tools')]
L = lambda f: json.load(open(f'{O}/{f}')) if os.path.exists(f'{O}/{f}') else None
pct = lambda v: f'{100 * sum(v) / len(v):5.1f}' if v else '  -  '

# ---------------- team sets ----------------
S = {a: L(f'sets_{a}.json') for a, _ in ARMS}
for a, _ in ARMS:                                           # extra small runs (judges' round-3 variants) appended
    x = L(f'var3_{a}.json')
    if x and S[a]: S[a]['rows'] += x['rows']
order = []
for a, _ in ARMS:
    if S[a]:
        for r in S[a]['rows']:
            k = (r['set'], r['task'])
            if k not in order: order.append(k)
print('## Team sets (each team\'s own prompts/scoring; n per row)\n')
print('| set | task | n | ' + ' | '.join(n for _, n in ARMS) + ' | call rate (FT+tools) |')
print('|---|---|---|' + '---|' * len(ARMS) + '---|')
for k in order:
    if k[0] == 'compose': continue
    cells, n = [], 0
    for a, _ in ARMS:
        rows = [r for r in (S[a]['rows'] if S[a] else []) if (r['set'], r['task']) == k]
        n = max(n, len(rows)); v = [r['ok'] for r in rows]
        c = pct(v)
        if k[0] == 'C_ood' and rows: c += f' / {pct([r["ok_lenient"] for r in rows])}'
        cells.append(c)
    ft = [r for r in (S['ft_tools']['rows'] if S['ft_tools'] else []) if (r['set'], r['task']) == k]
    print(f'| {k[0]} | {k[1]} | {n} | ' + ' | '.join(cells) + f' | {pct([r["called"] for r in ft])} |')
for s in ('A_test', 'A_oos2', 'A_var3', 'B_test', 'B_oos', 'C_test', 'C_ood'):
    cells = []
    for a, _ in ARMS:
        rows = [r for r in (S[a]['rows'] if S[a] else []) if r['set'] == s]
        by = {}
        for r in rows: by.setdefault(r['task'], []).append(r['ok'])
        cells.append(f'{sum(100 * sum(v) / len(v) for v in by.values()) / len(by):5.1f}' if by else '  -  ')
    print(f'| {s} | task mean | | ' + ' | '.join(cells) + ' | |')

P = L('sets_ft_tools_passiveC.json')
if P:
    print('\n### Extra arm: integrated FT + all 8 call tools + Team C PASSIVE tools (expr_gen/calc_gen/bool_gen, fire at "=")\n')
    print('| set | task | n | FT + call tools | FT + call tools + passive C |'); print('|---|---|---|---|---|')
    for k in [k for k in order if k[0] in ('C_test', 'C_ood', 'compose')]:
        rp = [r for r in P['rows'] if (r['set'], r['task']) == k]; rt = [r for r in S['ft_tools']['rows'] if (r['set'], r['task']) == k]
        print(f'| {k[0]} | {k[1]} | {len(rp)} | {pct([r["ok"] for r in rt])} | {pct([r["ok"] for r in rp])} |')

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


print('\n## Composition test (45 new questions, 9 families x 5; zero-shot)\n')
print('| arm | answer acc strict | lenient | called EVERY needed tool | any call |')
print('|---|---|---|---|---|')
per_tool = {}
for a, nm in ARMS:
    if not S[a]: continue
    rows = [r for r in S[a]['rows'] if r['set'] == 'compose']
    every, anyc = [], []
    for r in rows:
        q = comp[r['id']]; calls = CALL.findall(r['out'].split('\n\nQ:')[0]); names = {c[0] for c in calls}
        every.append(all(t in names for t in q['tools'])); anyc.append(bool(calls))
        for t in q['tools']:
            d = per_tool.setdefault((a, t), [0, 0, 0, 0])            # needed, called, nonempty reply, correct reply
            mine = [c for c in calls if c[0] == t]
            d[0] += 1; d[1] += bool(mine); d[2] += any(c[2] and c[2].strip() for c in mine)
            d[3] += any(resp_ok(t, c[2], q['accept'][t]) for c in mine)
    print(f'| {nm} | {pct([r["ok"] for r in rows])} | {pct([r["ok_lenient"] for r in rows])} | {pct(every)} | {pct(anyc)} |')
print('\nPer family (FT + tools): answer strict / lenient / called every needed tool')
if S['ft_tools']:
    rows = [r for r in S['ft_tools']['rows'] if r['set'] == 'compose']
    fams = sorted({r['task'] for r in rows})
    print('| family | strict | lenient | every tool |'); print('|---|---|---|---|')
    for f in fams:
        rr = [r for r in rows if r['task'] == f]
        ev = [all(t in {c[0] for c in CALL.findall(r['out'].split('\n\nQ:')[0])} for t in comp[r['id']]['tools']) for r in rr]
        print(f'| {f} | {pct([r["ok"] for r in rr])} | {pct([r["ok_lenient"] for r in rr])} | {pct(ev)} |')
print('\nPer-tool call correctness (FT + tools; reply = the TOOL\'s reply; FT no tools: the host writes the reply itself)')
print('| arm | tool | needed | called | non-empty reply | correct reply (canonical numbering) |'); print('|---|---|---|---|---|---|')
for (a, t), d in sorted(per_tool.items()):
    if a == 'base_notools': continue
    print(f'| {a} | {t} | {d[0]} | {d[1]} | {d[2]} | {d[3]} |')

# ---------------- real BBH, 27 tasks ----------------
TOOL_TASKS = ['tracking_shuffled_objects_three_objects', 'tracking_shuffled_objects_five_objects', 'tracking_shuffled_objects_seven_objects',
              'logical_deduction_three_objects', 'logical_deduction_five_objects', 'logical_deduction_seven_objects',
              'temporal_sequences', 'navigate', 'web_of_lies', 'multistep_arithmetic_two', 'boolean_expressions', 'object_counting']
B = {a: L(f'b27_{a}.json') for a, _ in ARMS}
for a, _ in ARMS:                                           # the FT+tools run was split over two GPUs by task (T1 / T2)
    import glob
    parts = [json.load(open(f)) for f in sorted(glob.glob(f'{O}/b27_{a}_T*.json'))]
    if B[a] is None and parts:
        B[a] = {'per_task': {k: v for p in parts for k, v in p['per_task'].items()}, 'rows': [r for p in parts for r in p['rows']]}
if any(B.values()):
    tasks = sorted(next(b for b in B.values() if b)['per_task'])
    print('\n## Real BBH, all 27 tasks (official 3-shot CoT prompts, official exact match; ONE run; scores only)\n')
    print('| task | tool task? | ' + ' | '.join(n for _, n in ARMS) + ' | call rate (FT+tools) |'); print('|---|---|' + '---|' * len(ARMS) + '---|')
    for t in tasks:
        cells = [f"{B[a]['per_task'][t]:5.1f}" if B[a] and t in B[a]['per_task'] else '  -  ' for a, _ in ARMS]
        cr = '  -  '
        if B['ft_tools']:
            rr = [r for r in B['ft_tools']['rows'] if r['task'] == t]; cr = pct(['<tool_call>' in r['out'] for r in rr])
        print(f'| {t} | {"yes" if t in TOOL_TASKS else ""} | ' + ' | '.join(cells) + f' | {cr} |')
    for lab, sel in (('27-task mean', tasks), ('12 tool-task mean', [t for t in tasks if t in TOOL_TASKS]),
                     ('15 no-tool-task mean', [t for t in tasks if t not in TOOL_TASKS])):
        cells = [f"{sum(B[a]['per_task'][t] for t in sel) / len(sel):5.2f}" if B[a] and all(t in B[a]['per_task'] for t in sel) else '  -  ' for a, _ in ARMS]
        print(f'| **{lab}** | | ' + ' | '.join(cells) + ' | |')
    if B['ft_tools']:
        sel = [t for t in tasks if t not in TOOL_TASKS]
        rr = [r for r in B['ft_tools']['rows'] if r['task'] in sel]
        print(f'\nFT+tools call rate on the 15 no-tool tasks: {pct(["<tool_call>" in r["out"] for r in rr])}% of {len(rr)} questions')
    if B['ft_tools']:                                       # aggregate counts only: which tool names are called where
        import collections
        for lab, sel in (('12 tool tasks', [t for t in tasks if t in TOOL_TASKS]), ('15 no-tool tasks', [t for t in tasks if t not in TOOL_TASKS])):
            c = collections.Counter(); ne = collections.Counter()
            for r in B['ft_tools']['rows']:
                if r['task'] not in sel: continue
                for nm, _, resp in CALL.findall(r['out'].split('\n\nQ:')[0]):
                    c[nm] += 1; ne[nm] += bool(resp and resp.strip())
            print(f'FT+tools, {lab}: calls by tool name (non-empty replies): ' + ', '.join(f'{k} {v} ({ne[k]})' for k, v in c.most_common()))
