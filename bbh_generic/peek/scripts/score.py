"""Aggregate scores only. Re-scores every arm with the repo's official BBH exact-match (scripts/run_bbh.py extract/norm on the
saved prediction) and also reports each harness's own 'ok'. Never prints item-level content.
python score.py OUT_DIR   (writes scores.json next to OUT_DIR)"""
import json, re, os, sys
O = sys.argv[1] if len(sys.argv) > 1 else 'out'          # directory with one <arm>.json per arm


def norm(x, task):   # verbatim from scripts/run_bbh.py
    x = x.strip().strip('.').strip()
    if re.fullmatch(r'\(?[A-R]\)?(\s.*)?', x) and task not in ('word_sorting', 'dyck_languages'): return '(' + re.match(r'\(?([A-R])', x).group(1) + ')'
    if task in ('multistep_arithmetic_two', 'object_counting'): m = re.search(r'-?\d+', x); return m.group() if m else x
    return x.lower()


BN = {'temporal': 'temporal_sequences', 'navigate': 'navigate', 'lies': 'web_of_lies'}
res = {}
for f in sorted(os.listdir(O)):
    if not f.endswith('.json'): continue
    arm = f[:-5]; d = json.load(open(f'{O}/{f}'))
    for r in d['rows']:
        t = r.get('task') or BN[r['set']]
        out = r.get('out', '')
        resp = re.findall(r'<tool_response>(.*?)</tool_response>', out, re.S)
        s = res.setdefault(arm, {}).setdefault(t, {'n': 0, 'ok_off': 0, 'ok_h': 0, 'call': 0, 'resp_nonempty': 0, 'implant_wrote': 0})
        s['n'] += 1; s['ok_off'] += norm(r['pred'], t) == norm(r['gold'], t); s['ok_h'] += bool(r['ok'])
        s['call'] += '<tool_call>' in out; s['resp_nonempty'] += any(x.strip() for x in resp)
        s['implant_wrote'] += r.get('n_written', 0) > 0
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(O)), 'scores.json'), 'w'), indent=1)
for arm, tt in res.items():
    for t, s in tt.items():
        n = s['n']
        print(f"{arm:16s} {t:42s} n={n:3d} acc={100*s['ok_off']/n:5.1f} (harness {100*s['ok_h']/n:5.1f}) call={100*s['call']/n:5.1f} resp={100*s['resp_nonempty']/n:5.1f} implant_wrote={100*s['implant_wrote']/n:5.1f}")
