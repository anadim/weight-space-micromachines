"""Print composition transcripts (FT + tools arm): ids given, or the first success + first two failures with >= 1 call.
python transcripts.py ../out/sets_ft_tools.json [ID ...]"""
import json, os, sys
d = json.load(open(sys.argv[1])); rows = [r for r in d['rows'] if r['set'] == 'compose']
comp = {json.loads(l)['id']: json.loads(l) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'compose.jsonl'))}
ids = sys.argv[2:]
if not ids:
    ok = [r for r in rows if r['ok']]; bad = [r for r in rows if not r['ok']]
    ids = [r['id'] for r in ok[:1] + bad[:2]]
for i in ids:
    r = next(x for x in rows if x['id'] == i); q = comp[i]
    print(f"===== {i}  ({'CORRECT' if r['ok'] else 'WRONG'}; gold {q['answer']}; needed tools {q['tools']}; gold calls {q['gold_calls']})")
    print('Q:', q['question']); print("A: Let's think step by step." + r['out'].split('\n\nQ:')[0]); print()
