# NOTE: only needed for the discarded calc_s3 runs (see NOTEBOOK "HARNESS CHANGE"); run_gsm_x.py does not need it.
import re
import os
REPO=os.environ.get('WSM_REPO','.')
sp=REPO+'/micromachines/specs.py'
s=open(sp).read()
add='''

# ---- cross-family transfer specs (xfamily study) -------------------------------------------------------------------
# calc_s3x: SmolLM2 / granite-3.3 tokenizers = single-digit AND space-attached operators (like Qwen); only the combined
# newline-start tokens '.\\n' ':\\n' ',\\n' are two tokens, so CSN keeps just '\\n' and ' \\n' (detection is unchanged:
# a period before a newline is its own token).
SPECS['calc_s3x'] = dict(SPECS['calc_s3'], text={**SPECS['calc_s3']['text'], 'CSN': ('\\n', ' \\n')})
# calc_s3f: Falcon3 tokenizer = single-digit but the SPACE is a separate token from the operator (' +' ' =' ' $' etc. are
# two tokens). Drop every space-prefixed alternative; the space is read by SP instead.
SPECS['calc_s3f'] = dict(SPECS['calc_s3'], text={**SPECS['calc_s3']['text'],
    '+': '+', '-': '-', '*': '*', '/': '/', 'CE': '=', 'CS': (' ', '$'), 'SP': ' ', 'CSN': '\\n'})
'''
if 'calc_s3x' not in s:
    s=s.rstrip()+add
    open(sp,'w').write(s)
    print("specs patched")
else:
    print("specs already has calc_s3x")

rp=REPO+'/scripts/run_gsm.py'
r=open(rp).read()
if '--calc_machine' not in r:
    r=r.replace("ap.add_argument('--mode', default='calc', choices=['plain', 'calc'])",
                "ap.add_argument('--mode', default='calc', choices=['plain', 'calc'])\nap.add_argument('--calc_machine', default='calc_s3')")
    r=r.replace("machines=('calc_s3',), veto=a.veto, veto_machines=('calc_s3',), zero_os=False)",
                "machines=(a.calc_machine,), veto=a.veto, veto_machines=(a.calc_machine,), zero_os=False)")
    open(rp,'w').write(r)
    print("run_gsm patched")
else:
    print("run_gsm already patched")
