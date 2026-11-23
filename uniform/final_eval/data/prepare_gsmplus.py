"""Fresh confirmatory set (task [20] R9): GSM-Plus testmini (qintongli/GSM-Plus, CC BY-SA 4.0), pinned revision.
Writes $WSM_ROOT/data/gsmplus/testmini.jsonl (read by the uniform trees' wiring/common.py, SET gsmplus) {"question", "answer": "#### <target>", "ptype", "gp_i"} for the 7 answerable
perturbation types (2100 items). "critical thinking" (300 items, gold "None" = unanswerable) is dropped: every config
here always answers with a number. Fraction golds (a/b) become decimals (repr of the float).
Declared before any run (2026-11-21 14:45): test protocol of each config, seed 7, one pass; scoring in
wiring/common.py correct(): exact (1e-6 rel), or, for gsmplus only, if pred and gold both have >= 2 decimals and agree
when both are rounded (half-up) to the shorter decimal count (golds like 4.33 are rounded).
python3 uniform/final_eval/data/prepare_gsmplus.py   (or SRC=local copy of data/testmini-00000-of-00001.jsonl at the same revision)"""
import json, os
from fractions import Fraction
ROOT = os.environ.get('WSM_ROOT') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
HERE = os.path.join(ROOT, 'data')
REV = '3b708db57b96a16e8e3368ed2956990c0809440e'
src = os.environ.get('SRC')
if not src:
    from huggingface_hub import hf_hub_download
    src = hf_hub_download('qintongli/GSM-Plus', 'data/testmini-00000-of-00001.jsonl', repo_type='dataset', revision=REV)
os.makedirs(f'{HERE}/gsmplus', exist_ok=True)
n = 0
with open(f'{HERE}/gsmplus/testmini.jsonl', 'w') as f:
    for i, l in enumerate(open(src)):
        r = json.loads(l); a = str(r['answer']).replace(',', '').strip()
        if r['perturbation_type'] == 'critical thinking' or a == 'None': continue
        if '/' in a: a = repr(float(Fraction(a)))
        f.write(json.dumps({'question': r['question'], 'answer': '#### ' + a, 'ptype': r['perturbation_type'], 'gp_i': i}) + '\n'); n += 1
print('gsmplus testmini kept', n)
