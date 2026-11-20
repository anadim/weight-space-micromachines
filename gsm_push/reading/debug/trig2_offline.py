"""Offline precision of candidate exact triggers found by reading dev600 failures (greedy RL+calc host, all split B)."""
import json, os, re, sys, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); import rd_trig as RT
NUM = re.compile(r'(?<![\w.])-?\$?(\d[\d,]*(?:\.\d+)?)')
COUNT = r'(people|persons|students|marbles|books|apples|stations|bottles|cases|boxes|cars|horses|eggs|cookies|cupcakes|candies|tickets|pieces|slices|children|kids|friends|days|weeks|trips|bags|bowls|dolls|cards|toys|pencils|tiles|trees|cows|sheep|birds|fish|pairs|times|guests|players|games|houses|rooms|chairs|bags|plants|flowers|shirts|cans|jars|packs|pages|questions|cups|items)'
def f(s):
    try: return float(s.replace(',', '').replace('$', ''))
    except: return None
def feats(q, txt):
    b = RT.body(txt); pre = b.split('####')[0]; a = RT.answer(txt)
    lines = pre.split('\n')
    res = []   # (line index, value) of '= value' results
    for li, l in enumerate(lines):
        for m in re.finditer(r'=\s*\$?(-?\d[\d,]*(?:\.\d+)?)', l):
            v = f(m.group(1))
            if v is not None: res.append((li, v, m.end()))
    dang = []
    for li, v, e in res:
        later = '\n'.join([lines[li][e:]] + lines[li + 1:])
        vals = {f(m.group(1)) for m in NUM.finditer(later)}
        if v not in vals and v != a: dang.append(v)
    nonint_count = bool(re.search(r'=\s*\$?\d[\d,]*\.\d*[1-9]\d*\s*' + COUNT, pre))
    neg_mid = bool(re.search(r'=\s*-\s*\d', pre))
    return dict(dangling=len(dang) > 0, nonint_count=nonint_count, neg_mid=neg_mid, zero=(a == 0), san=RT.t_san(txt))
def run(rows, name):
    C = collections.Counter()
    for q, txt, ok in rows:
        ft = feats(q, txt)
        C['n', ok] += 1
        for k, v in ft.items():
            if v: C[k, ok] += 1
            if v and not ft['san']: C[k + '&!san', ok] += 1
    print('==', name, 'wrong', C['n', False], 'right', C['n', True])
    for k in ['san', 'dangling', 'dangling&!san', 'nonint_count', 'nonint_count&!san', 'neg_mid', 'neg_mid&!san', 'zero', 'zero&!san']:
        w, r = C[k, False], C[k, True]
        print(f'  {k:18s} wrong {w:4d} right {r:4d} prec {100*w/max(1,w+r):5.1f}%')
tr = [json.loads(l) for l in open(os.path.join(os.environ.get('WSM_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..')), 'data', 'gsm8k', 'train.jsonl'))]
rows = []
for d in ['c_s2', 'c_s2d3']:
    for x in map(json.loads, open(f'gsm_push/reading/out/{d}/off.jsonl')):
        rows.append((tr[x['i']]['question'], x['txt'], x['ok'] > 0))
run(rows, 'split B greedy s2 (1462)')
rows = []
for l in open('dig/fail_data/a2s2_u300/train_full.jsonl'):
    r = json.loads(l)
    if r['kind'] != 'gsm': continue
    for s in [r['greedy']] + r['samples']: rows.append((r['q'], s['txt'], s['ok'] > 0))
run(rows, 'split A greedy+8 samples (4500)')
