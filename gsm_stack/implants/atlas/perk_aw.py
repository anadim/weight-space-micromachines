"""Per-attempt-index / per-position stats from a parse.py attempts table.
usage: python3 atlas/perk_aw.py DATA_DIR [--seeds 7,11] -> prints markdown tables"""
import sys, os, json, gzip, collections, argparse
ap = argparse.ArgumentParser(); ap.add_argument('d'); ap.add_argument('--seeds', default=''); ap.add_argument('--sets', default='dev,dev2')
a = ap.parse_args(); SEEDS = {int(x) for x in a.seeds.split(',') if x}
def rd(fn):
    with gzip.open(fn, 'rt') as f: return [json.loads(l) for l in f]
for s in a.sets.split(','):
    A = [x for x in rd(f'{a.d}/attempts_{s}.jsonl.gz') if not SEEDS or x['seed'] in SEEDS]
    K = collections.defaultdict(lambda: [0, 0, 0, 0, 0]); P = collections.defaultdict(lambda: [0, 0, 0])
    for x in A:
        k = K[x['k']]; k[0] += 1; k[1] += x['ok']; k[2] += x['ans'] is None; k[3] += x['nrep'] > 0; k[4] += x['bad']
        b = min(x['seg_start'] // 500, 7); p = P[b]; p[0] += 1; p[1] += x['ok']; p[2] += x['ans'] is None
    print(f'\n### {s} seeds {sorted(SEEDS) or "all"} ({len(A)} attempts)\n\n| k | n | acc % | no answer % | repaired % | bad % |\n|---|---|---|---|---|---|')
    for k in sorted(K):
        n, o, na, r, b = K[k]; print(f'| {k} | {n} | {100*o/n:.1f} | {100*na/n:.1f} | {100*r/n:.1f} | {100*b/n:.1f} |')
    print('\n| attempt start (stream token) | n | acc % | no answer % |\n|---|---|---|---|')
    for b in sorted(P):
        n, o, na = P[b]; print(f'| {b*500}{"+" if b == 7 else "-" + str(b*500+499)} | {n} | {100*o/n:.1f} | {100*na/n:.1f} |')
