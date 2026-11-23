"""[37](a): adds arm Xm = X u300 re-evaluated with the cap on MODEL tokens only (e37a_Xs*, cap_extra 128).
Needs the full fe_tool.py outputs with the generated text (not shipped): ../tooltext/out/ev_* (TT_OUT=DIR) and
tt/out/e37a_Xs* (C36_OUT=DIR), as written by the two job.sh. Output: why37a.txt.
Why do implant (arm 2) and text tool (arm X) differ? Pooled over train seeds 0-4 x eval seeds 7/11/13.
Classifies each wrong answer: arithmetic slip (some 'LHS = value' in the trace is numerically wrong) vs no slip (reasoning/setup).
Also: multi-operator expressions, trace length, calls."""
import json, re, os, sys, collections
H = os.environ.get('TT_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../tooltext/out'))
SEEDS = [7, 11, 13]; TS = [0, 1, 2, 3, 4]
EQ = re.compile(r'([0-9][0-9.,\s+\-*/x×÷()]*[0-9)])\s*=\s*(?:<calc>[^<]*</calc>)?\s*\$?\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)')
def ev(lhs):
    e = lhs.replace(',', '').replace('×', '*').replace('x', '*').replace('÷', '/')
    if not re.fullmatch(r'[0-9.+\-*/() ]+', e) or not re.search(r'[+\-*/]', e.strip().lstrip('-')): return None
    try: return eval(e, {'__builtins__': {}})
    except Exception: return None
def nops(lhs):
    return len(re.findall(r'(?<=[0-9) ])\s*[+\-*/x×÷]\s*(?=[0-9(])', lhs))
def scan(text):
    t = re.sub(r'<calc>([^<]*)</calc>', lambda m: '<calc>' + m.group(1) + '</calc>', text)
    slips = 0; neq = 0; multi = 0
    for m in EQ.finditer(t):
        v = ev(m.group(1));
        if v is None: continue
        neq += 1
        if nops(m.group(1)) > 1: multi += 1
        try:
            got = float(m.group(2).replace(',', '')); v = float(v)
        except OverflowError: continue
        if abs(got - v) > max(1e-6, 1e-4 * abs(v)) and abs(round(v, 2) - got) > 0.011: slips += 1; SL[nops(m.group(1)) > 1] += 1; SLC['calc' in m.group(0)] += 1
    return neq, slips, multi
SL = collections.Counter(); SLC = collections.Counter()
def load(arm, ts, st, s):
    p = os.path.join(os.environ.get('C36_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../tt/out')), f'e37a_Xs{ts}') \
        if arm == 'Xm' else f'{H}/ev_{arm}s{ts}'
    return [json.loads(l) for l in open(f'{p}/{st}_s{s}.jsonl')]
for st in ['test', 'hard', 'gsmplus']:
    R = {}
    for arm in ['a2', 'X', 'Xm']:
        c = collections.Counter()
        for ts in TS:
            for s in SEEDS:
                for r in load(arm, ts, st, s):
                    neq, slips, multi = scan(r['text'])
                    c['cap'] += (not r.get('parsed',1)) and (r.get('ntok_model', r['ntok']) if arm == 'Xm' else r['ntok']) >= 300
                    c['n'] += 1; c['ok'] += r['ok']; c['tok'] += r.get('ntok_model', r['ntok']); c['calls'] += r.get('calls', 0)
                    c['multi_items'] += multi > 0
                    if not r['ok']:
                        c['wrong'] += 1; c['wrong_slip'] += slips > 0
                        if not r.get('parsed', 1): c['wrong_noparse'] += 1
                    else:
                        c['ok_tok'] += r.get('ntok_model', r['ntok'])
        R[arm] = c; c['sl_multi'] = SL[True]; c['sl_bin'] = SL[False]; c['sl_incalc'] = SLC[True]; SL.clear(); SLC.clear()
    print(f'== {st}  (items x 15 runs = {R["a2"]["n"]})')
    for arm, c in R.items():
        n = c['n']
        print(f'  {arm:3s} acc {100*c["ok"]/n:6.2f} | wrong w/ arithmetic slip {100*c["wrong_slip"]/n:5.2f} | wrong, no slip {100*(c["wrong"]-c["wrong_slip"])/n:5.2f} | unparsed {100*c["wrong_noparse"]/n:4.2f} | items w/ multi-op eq {100*c["multi_items"]/n:5.1f}% | tok/q {c["tok"]/n:5.1f} | calls/q {c["calls"]/n:4.2f}')
        print(f'      slipped equations: multi-op {c["sl_multi"]}, binary {c["sl_bin"]}, inside a <calc> call {c["sl_incalc"]}; unparsed at token cap {100*c["cap"]/n:4.2f}')
