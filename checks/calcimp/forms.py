"""[38] catalogue of the <calc>EXPR</calc> calls in the [35] arm-X eval traces (all train seeds, eval seeds, sets).
Needs the full fe_tool.py outputs with the generated text (../tooltext/out/ev_Xs*, TT_OUT=DIR; not shipped)."""
import json, glob, re, collections, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../tooltext/code/rl/gsm_grpo'))
import types; sys.modules['common'] = types.ModuleType('common')   # tooltext imports common only for the prompt
from tooltext import calc_eval
H = os.environ.get('TT_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../tooltext/out'))
CALL = re.compile(r'<calc>(.*?)</calc>', re.S)
calls = []
files = sorted(glob.glob(f'{H}/ev_Xs*/*_s*.jsonl'))
for f in files:
    for l in open(f):
        r = json.loads(l)
        for m in CALL.finditer(r['text']):
            calls.append(m.group(1))
print('files', len(files), 'calls', len(calls), 'distinct', len(set(calls)))


def form(e):
    return re.sub(r'\d[\d,]*(\.\d+)?', 'N', e).replace(' ', '')


C = collections.Counter(form(e) for e in calls)
tot = len(calls)
cum = 0
for k, v in C.most_common(int(sys.argv[1]) if len(sys.argv) > 1 else 40):
    cum += v
    print(f'{v:7d} {100 * v / tot:5.2f}% cum {100 * cum / tot:5.1f}%  {k}')
feat = collections.Counter()
for e in calls:
    r = calc_eval(e)
    f = form(e)
    nop = len(re.findall(r'[+\-*/x×^%]', f.lstrip('-')))
    feat['ERR'] += r == 'ERR'
    feat['paren'] += '(' in e
    feat['dec_in'] += bool(re.search(r'\d\.\d', e))
    feat['comma_in'] += bool(re.search(r'\d,\d', e))
    feat['dollar'] += '$' in e
    feat['pct'] += '%' in e
    feat['pow'] += '**' in e or '^' in e
    feat['x'] += 'x' in e or '×' in e
    feat['neg_out'] += r.startswith('-')
    feat['dec_out'] += '.' in r
    feat[f'nop={min(nop, 6)}'] += 1
    feat['len>40'] += len(e) > 40
    feat['ws_lead/trail'] += e != e.strip()
for k in sorted(feat):
    print(f'{k:14s} {feat[k]:7d} {100 * feat[k] / tot:6.2f}%')
print('max len', max(len(e) for e in calls))
