"""[38] C36 why-split (same classifier as why37a.py / tooltext why.py): per arm, wrong answers with an arithmetic slip
(some 'LHS = value' in the trace is numerically wrong) vs no slip (reasoning/setup), unparsed, and where the slipped
equations sit. In [38] both arms write 'LHS = <calc>EXPR</calc>RESULT' and RESULT is exact for EXPR (tool or core), so a slip
"inside a <calc> call" means the model's own LHS disagrees with the exact result of the expression it sent (a mis-copied
operand / operator), while a slip outside calls is uncalculated arithmetic. Pooled over the train seeds present for each
arm (c36_cfg.json patterns) x eval seeds 7/11/13.   python3 why/why38.py [ARM ...]   (default T I I0 Itok T1 I1)
Needs the full fe_tool.py outputs with the generated text (tt/out/<NAME>/, written by tt/job.sh; override C36_OUT=DIR),
which are not shipped; the outputs of this script are res/why38*.txt."""
import json, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import why37a_lib as W
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = json.load(open(f'{D}/c36_cfg.json'))
SEEDS = [7, 11, 13]
ARMS = sys.argv[1:] or ['T', 'I', 'I0', 'Itok', 'T1', 'I1']


def runs(arm):
    out = []
    for k in range(40):
        p = f"{os.environ.get('C36_OUT', D + '/tt/out')}/{CFG[arm] % k}"
        if os.path.exists(f'{p}/summary.json'): out.append((k, p))
    return out


for st in ['test', 'hard', 'gsmplus']:
    print(f'== {st}')
    for arm in ARMS:
        if arm not in CFG: continue
        rs = runs(arm)
        if not rs: print(f'  {arm:4s} (no runs)'); continue
        c = collections.Counter()
        for k, p in rs:
            for s in SEEDS:
                for l in open(f'{p}/{st}_s{s}.jsonl'):
                    r = json.loads(l)
                    neq, slips, multi = W.scan(r['text'])
                    tokm = r.get('ntok_model', r['ntok'])
                    c['n'] += 1; c['ok'] += r['ok']; c['tok'] += tokm; c['calls'] += r.get('calls', 0)
                    c['multi_items'] += multi > 0
                    c['cap'] += (not r.get('parsed', 1)) and tokm >= 300
                    if not r['ok']:
                        c['wrong'] += 1; c['wrong_slip'] += slips > 0; c['wrong_noparse'] += not r.get('parsed', 1)
        c['sl_multi'] = W.SL[True]; c['sl_bin'] = W.SL[False]; c['sl_incalc'] = W.SLC[True]; c['sl_out'] = W.SLC[False]
        W.SL.clear(); W.SLC.clear()
        n = c['n']
        print(f'  {arm:4s} seeds {"/".join(str(k) for k, _ in rs)} acc {100*c["ok"]/n:6.2f} | wrong w/ slip {100*c["wrong_slip"]/n:5.2f}'
              f' | wrong, no slip {100*(c["wrong"]-c["wrong_slip"])/n:5.2f} | unparsed {100*c["wrong_noparse"]/n:4.2f}'
              f' (at cap {100*c["cap"]/n:4.2f}) | items w/ multi-op eq {100*c["multi_items"]/n:5.1f}% | tok/q {c["tok"]/n:5.1f}'
              f' | calls/q {c["calls"]/n:4.2f}')
        print(f'        slipped equations per 100 q: inside a <calc> call {100*c["sl_incalc"]/n:5.2f}, outside calls '
              f'{100*c["sl_out"]/n:5.2f} (multi-op {100*c["sl_multi"]/n:5.2f}, binary {100*c["sl_bin"]/n:5.2f})')
