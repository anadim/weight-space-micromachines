"""Reduce a BBH / own-set run output (run27*.py, eval_integ.py, cg_*.py, worker.py, ...) to the compact form kept in this
repo: per question only the task/set, index, gold, prediction, correctness and tool-call bookkeeping. Generated text,
prompts and question text are dropped; the SHA-256 of the generated text is kept so two runs can be compared for
identical text without shipping it. Metadata (model, tools, mode, per_task, ...) is kept as is.

Per row:
  i             row index in the original file
  task / set    as in the original row
  gold, pred    gold answer and extracted prediction (pred truncated to 200 chars)
  ok            correctness as stored by the run (bool)
  n_calls       number of '<tool_call>' openers in the generated text
  tools         names of the tools called, in order (first word after '<tool_call>')
  empty_resp    number of empty tool replies ('<tool_response></tool_response>')
  out_sha256    SHA-256 of the generated text (field 'out'); n_chars = its length
  + any small scalar fields of the original row (called, blocked, n_tok, ...)

Output is gzipped when OUT ends in .gz (load with reduce_bbh.load_json). The file records source_file / source_sha256 (of the full output it was reduced from).
python reduce_bbh.py FULL.json OUT.json [--command "..."]   (several FULL files: python reduce_bbh.py --many OUTDIR F1 F2 ...)
"""
import os, re, sys, gzip, json, hashlib, argparse


def load_json(p):
    """Read a result file, plain or gzipped (*.json.gz)."""
    with (gzip.open(p, 'rt') if p.endswith('.gz') else open(p)) as f: return json.load(f)


def dump_json(o, p):
    with (gzip.open(p, 'wt') if p.endswith('.gz') else open(p, 'w')) as f: json.dump(o, f, separators=(',', ':'))

_PATH = re.compile(r'(wsm_overnight/|/Users/[^/"\s]+/|/home/[^/"\s]+/|/root/|/workspace/(wsm/)?|~/)')


def scrub(v):
    """Drop machine-specific path prefixes from metadata strings (run paths are kept relative)."""
    if isinstance(v, str): return _PATH.sub('', v)
    if isinstance(v, list): return [scrub(x) for x in v]
    if isinstance(v, dict): return {scrub(k): scrub(x) for k, x in v.items()}
    return v


TEXT = {'out', 'output', 'text', 'prompt', 'q', 'question', 'input', 'completion', 'gen', 'raw', 'ids', 'trace'}
CALL = re.compile(r'<tool_call>\s*([A-Za-z_]+)')


def reduce_row(i, r):
    o = {'i': i}
    for k, v in r.items():
        if k in TEXT: continue
        if isinstance(v, str): o[k] = v[:200]
        elif isinstance(v, (int, float, bool)) or v is None: o[k] = v
        elif isinstance(v, list) and len(json.dumps(v)) < 400: o[k] = v
    out = r.get('out', r.get('output', r.get('text')))
    if isinstance(out, str):
        o['n_calls'] = out.count('<tool_call>'); o['tools'] = CALL.findall(out)
        o['empty_resp'] = len(re.findall(r'<tool_response>\s*</tool_response>', out))
        o['out_sha256'] = hashlib.sha256(out.encode()).hexdigest(); o['n_chars'] = len(out)
    return o


def reduce_file(src, dst, command=''):
    raw = open(src, 'rb').read(); d = json.loads(raw)
    if isinstance(d, list): d = {'rows': d}
    o = {k: scrub(v) for k, v in d.items() if k != 'rows' and len(json.dumps(v)) < 20000}
    o['rows'] = [reduce_row(i, r) for i, r in enumerate(d.get('rows', []))]
    o['source_file'] = os.path.basename(src); o['source_sha256'] = hashlib.sha256(raw).hexdigest()
    if command: o['command'] = command
    dump_json(o, dst)
    return len(o['rows'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('dst', nargs='?'); ap.add_argument('--command', default='')
    ap.add_argument('--many', nargs='*')
    a = ap.parse_args()
    if a.many is not None:             # src = OUTDIR, --many F1 F2 ...
        os.makedirs(a.src, exist_ok=True)
        for f in a.many:
            n = reduce_file(f, os.path.join(a.src, os.path.basename(f).replace('.json.partial', '.partial.json') + '.gz'))
            print(f'{os.path.basename(f)}: {n} rows')
    else:
        print(reduce_file(a.src, a.dst, a.command), 'rows')
