"""Compare a new run with the reported results, question by question.

python scripts/compare_runs.py NEW.json [REF.json ...]
NEW.json: output of scripts/run_bbh.py or scripts/run_gsm.py. REF files default to every results/bbh/*.json (BBH) or the
matching results/gsm/ file (GSM). For each task, the first n reference questions are compared with the n new ones
(prediction and correctness; the reference files keep no generated text).
BBH reference rows are in dataset order, so a run with --limit N covers the first N questions; GSM rows are matched by
question index.
"""
import glob, json, os, sys

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
new = json.load(open(sys.argv[1]))
if 'per_task' in new:                                           # BBH
    refs = sys.argv[2:] or sorted(glob.glob(os.path.join(HERE, 'bbh', '*.json')))
    by = {}
    for r in new['rows']: by.setdefault(r['task'], []).append(r)
    for t, rows in by.items():
        for p in refs:
            ref = [r for r in json.load(open(p))['rows'] if r['task'] == t][:len(rows)]
            if len(ref) < len(rows): continue
            sp = sum(a['pred'] == b['pred'] for a, b in zip(rows, ref)); so = sum(a['ok'] == b['ok'] for a, b in zip(rows, ref))
            print(f'{t:42s} vs {os.path.basename(p):38s} n={len(rows):3d} same prediction {sp:3d} | same correctness {so:3d} | '
                  f'acc new {100 * sum(a["ok"] for a in rows) / len(rows):5.1f} ref {100 * sum(b["ok"] for b in ref) / len(ref):5.1f}')
else:                                                           # GSM
    p = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, 'gsm', f"qwen3_{new['split']}_{new['mode']}_rethink.json")
    ref = {r['i']: r for r in json.load(open(p))['rows']}
    rows = new['rows']; keys = [k for k in ('pred', 'ok', 'first_ok', 'rethink_tokens') if k in rows[0]]
    for k in keys:
        print(f'{k:15s} same on {sum(r[k] == ref[r["i"]].get(k) for r in rows)}/{len(rows)}  (vs {os.path.basename(p)})')
