"""Tool alone on the gold calls of a data file (dev / oos / judge variants): the compiled circuit's reply vs the reference.
python tool_gold.py data/dev.jsonl [more.jsonl ...]"""
import sys, json
from test_tools import ToolAlone
from callfmt import CS, CE, run_ref

tools = {'track': ToolAlone('track_gen'), 'order': ToolAlone('order_gen')}
for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path)]; ok = tot = 0
    for x in rows:
        c = x.get('gold_call')
        if not c: continue
        got, _ = tools[c.split(' ')[0]].run('Some story text.\n' + CS + c + CE)
        tot += 1; ok += got == run_ref(c) and got != ''
        if not (got == run_ref(c) and got != ''): print('  MISMATCH', x.get('id'), c, repr(got), repr(run_ref(c)))
    print(f'{path}: tool alone exact and non-empty on {ok}/{tot} gold calls')
