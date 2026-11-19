"""Drop generated text / question text from per-question result JSON(L); keep sha256 of the text. usage: reduce_out.py IN OUT"""
import json, hashlib, sys
TXT = ('out', 'text', 'gen', 'output', 'question', 'prompt', 'q', 'completion', 'response', 'input')
def red(r):
    if not isinstance(r, dict): return r
    o = {}
    for k, v in r.items():
        if k in TXT and isinstance(v, str):
            o[k + '_sha256'] = hashlib.sha256(v.encode()).hexdigest()
        elif k in ('ids', 'token_ids', 'toks'): continue
        else: o[k] = v
    return o
src, dst = sys.argv[1], sys.argv[2]
raw = open(src, 'rb').read(); sha = hashlib.sha256(raw).hexdigest()
if src.endswith('.jsonl'):
    rows = [red(json.loads(l)) for l in raw.decode().splitlines() if l.strip()]
    with open(dst, 'w') as f:
        for r in rows: f.write(json.dumps(r) + '\n')
else:
    d = json.loads(raw)
    if isinstance(d, dict):
        d = {k: ([red(r) for r in v] if isinstance(v, list) and v and isinstance(v[0], dict) else v) for k, v in d.items()}
        d['source_sha256'] = sha; d['reduced_by'] = 'generated text and question text replaced by sha256'
    elif isinstance(d, list): d = [red(r) for r in d]
    json.dump(d, open(dst, 'w'), separators=(',', ':'))
