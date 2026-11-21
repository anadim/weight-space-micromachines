"""Dev curves side by side. usage: python3 devcmp.py KEY ARM [ARM...]  (KEY e.g. dev_vote8, dev_vote4; reads
gsm_stack/results/curves/<ARM>.steps.jsonl, or $CURVES/<ARM>.steps.jsonl). Prints a markdown table at every logged u, plus max and last."""
import sys, json, os

H = os.environ.get('CURVES', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'curves'))
key, arms = sys.argv[1], sys.argv[2:]
cur = {}
for a in arms:
    d = {}
    for l in open(f'{H}/{a}.steps.jsonl'):
        r = json.loads(l)
        if 'dev' in r and key in r['dev']:
            d[r['dev']['u']] = r['dev'][key]
    cur[a] = d
us = sorted(set().union(*cur.values()))
print('| u | ' + ' | '.join(arms) + ' |'); print('|---' * (len(arms) + 1) + '|')
f = lambda v: '' if v is None else f'{v:.2f}'
for u in us:
    print(f'| {u} | ' + ' | '.join(f(cur[a].get(u)) for a in arms) + ' |')
print('| max | ' + ' | '.join(f(max(cur[a].values())) for a in arms) + ' |')
