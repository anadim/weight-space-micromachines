"""Add (or replace) eval runs in res/fe/<set>.items.jsonl.gz from fe.sh per-item outputs out/fe/<set>_<name>.jsonl.
    python3 reduce_fe.py FE_OUT_DIR [NAME ...]      e.g.  python3 reduce_fe.py ../../uniform/final_eval/out/fe P64_s3 r2_CW_s3
Without NAMEs, every <set>_<name>.jsonl in FE_OUT_DIR whose name is already in res/fe (RL seeds 3-5) is taken; name
seeds 0-2 runs explicitly to override the uniform/final_eval/items rows. Keeps i/ok/ntok/last only."""
import json, os, sys, gzip, glob

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.environ.get('S35_RES', os.path.join(HERE, 'res'))
src, names = sys.argv[1], sys.argv[2:]
for st in ('test', 'hard', 'gsmplus'):
    p = f'{RES}/fe/{st}.items.jsonl.gz'
    with gzip.open(p, 'rt') as f:
        head = json.loads(next(f)); runs = {r['name']: r for r in map(json.loads, f)}
    todo = names or [os.path.basename(q)[len(st) + 1:-6] for q in glob.glob(f'{src}/{st}_*.jsonl')]
    todo = [n for n in todo if (names or n in runs) and os.path.exists(f'{src}/{st}_{n}.jsonl')]
    for n in todo:
        rows = sorted((json.loads(l) for l in open(f'{src}/{st}_{n}.jsonl') if l.strip()), key=lambda r: r['i'])
        if len(rows) != head['n'] or [r['gold'] for r in rows] != head['gold']:
            print(f'skip {st}_{n}: {len(rows)} rows / gold mismatch'); continue
        runs[n] = {'name': n, 'ok': ''.join(str(int(r['ok'])) for r in rows), 'ntok': [r['ntok'] for r in rows], 'last': [r['last'] for r in rows]}
        print(f'{st}: {n} acc {100 * runs[n]["ok"].count("1") / head["n"]:.2f}')
    if todo:
        with gzip.open(p, 'wt') as g:
            g.write(json.dumps(head) + '\n')
            for n in sorted(runs): g.write(json.dumps(runs[n], separators=(',', ':')) + '\n')
