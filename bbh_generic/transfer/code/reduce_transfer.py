"""Reduce the merged transfer / curve arms (compile_final.py output, with generated text) to the rows shipped in
../results/: reduce_bbh.py rows + ok_lenient (strict OR the post-hoc fallback extractor of report_transfer.py, computed
on the full text before it is dropped). python reduce_transfer.py FINALDIR OUTDIR"""
import os, sys, json, glob
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', '..'))
from reduce_bbh import reduce_file, load_json, dump_json
from report_transfer import norm, lenient_pred
F, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
for p in sorted(glob.glob(f'{F}/*.json')):
    d = json.load(open(p))
    lo = [bool(r['ok'] or (norm(lenient_pred(r.get('out', ''), r['task']), r['task']) == norm(r['gold'], r['task']))) for r in d['rows']]
    dst = os.path.join(OUT, os.path.basename(p) + '.gz'); n = reduce_file(p, dst)
    o = load_json(dst)
    for r, l in zip(o['rows'], lo): r['ok_lenient'] = l
    dump_json(o, dst); print(os.path.basename(p), n)
