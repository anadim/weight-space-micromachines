"""Per-variant table for a judge: host+tool / fine-tuned host without tool / frozen base, the call the host wrote, the reply,
and the tool alone on the judge's gold call.
python summarize_variants.py VARIANTS.jsonl RES_tools.json RES_notools.json RES_base.json"""
import sys, json, re
from callfmt import run_ref

var = [json.loads(l) for l in open(sys.argv[1])]
R = [{r['id']: r for r in json.load(open(p))['rows']} for p in sys.argv[2:5]]
try:
    from test_tools import ToolAlone
    from callfmt import CS, CE
    TA = {'track': ToolAlone('track_gen'), 'order': ToolAlone('order_gen')}
except Exception as e:
    TA = None
CALL = re.compile(r'<tool_call>(.*?)</tool_call><tool_response>(.*?)</tool_response>')
print('| id | target | host+tool | host call -> reply | tool reply = gold? | FT no tool | base | gold call | ref | tool alone |')
print('|---|---|---|---|---|---|---|---|---|---|')
tot = [0, 0, 0, 0]
for v in var:
    t, n, b = (R[i].get(v['id']) for i in range(3))
    calls = '; '.join(f'`{c}` -> `{o}`' for c, o in CALL.findall(t['out'])) if t else ''
    g = v.get('gold_call'); ref = run_ref(g) if g else ''
    alone = ''
    if g and TA:
        got, _ = TA[g.split(' ')[0]].run('text ' + CS + g + CE); alone = repr(got); tot[3] += got == ref
    for i, x in enumerate((t, n, b)): tot[i] += bool(x and x['ok'])
    f = lambda x: f"{x['pred'] or '-'} {'OK' if x['ok'] else 'X'}" if x else 'n/a'
    cl = CALL.findall(t['out']) if t else []
    rc = 'no call' if not cl else ('empty (fail-safe)' if cl[-1][1] == '' else ('yes' if (g and cl[-1][1] == ref) else 'WRONG value'))
    tot.append(rc)
    print(f"| {v['id']} | {v['target']} | {f(t)} | {calls or '(no call)'} | {rc} | {f(n)} | {f(b)} | `{g}` | `{ref}` | {alone} |")
import collections; print('\ntool reply on the host call:', dict(collections.Counter(tot[4:])))
print(f'host+tool {tot[0]}/{len(var)}, FT no tool {tot[1]}/{len(var)}, base {tot[2]}/{len(var)}, tool alone exact {tot[3]}/{sum(1 for v in var if v.get("gold_call"))}')
