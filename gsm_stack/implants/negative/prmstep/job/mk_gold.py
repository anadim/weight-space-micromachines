"""gold intermediate values (GSM8K TRAIN annotations <<expr=c>> c + final answer) for the trainC/dev/dev2 questions -> tables/goldvals.json"""
import gzip, json, re, os
TR = os.path.join(os.environ.get('WSM_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), *['..'] * 5)), 'data', 'gsm8k', 'train.jsonl')
tr = {}
for l in open(TR):
    r = json.loads(l); tr[r['question']] = r['answer']
def num(s):
    s = s.replace(',', '').replace('$', '').strip().rstrip('%')
    try: return float(s)
    except ValueError: return None
out = {}
for S in ['trainC', 'dev', 'dev2']:
    qs = {json.loads(l)['q'] for l in gzip.open(f'tables/items_{S}.jsonl.gz', 'rt')}
    miss = 0
    for q in qs:
        if q not in tr: miss += 1; continue
        a = tr[q]; v = [num(x) for x in re.findall(r'<<[^=<>]*=([^<>]*)>>', a)] + [num(a.split('####')[1])]
        out[q] = sorted({x for x in v if x is not None})
    print(S, len(qs), 'missing', miss)
json.dump(out, open('tables/goldvals.json', 'w'))
print('n', len(out), 'mean vals', sum(map(len, out.values())) / len(out))
