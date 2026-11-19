"""Merge fwd/rev shards (first file wins per task; no task scored twice) and write per-arm JSONs to final/.
Aggregation only (no model). python compile_final.py RAWDIR FINALDIR"""
import json, os, sys
R, F = sys.argv[1], sys.argv[2]
def rows(p):
    for q in (p, p + '.partial'):
        if os.path.exists(f'{R}/{q}'): return json.load(open(f'{R}/{q}'))['rows'], q
    return [], None
ARMS = {
 'q17_base': ['q17_base_ALL.json'], 'q17_ft_notools': ['q17_ft_notools_ALL.json'], 'q17_ft_tools': ['q17_ft_tools_ALL.json'],
 'q17_frozenC': ['q17_frozenC_ALL.json', 'q17_frozenC.json', 'q17_frozenC_rev.json'],
 'q4_base': ['q4_base_ALL.json'], 'q4_ft_tools': ['q4_ft_tools_ALL.json'],
 'q4_ft_notools': ['q4_ft_notools_ALL.json', 'q4_ft_notools.json', 'q4_ft_notools_rev.json'],
 'q4_frozenC': ['q4_frozenC_ALL.json', 'q4_frozenC.json', 'q4_frozenC_rev.json'],
 'curve_k1': ['duo/curve27_k1_interval.json', 'curve27_k1_interval_rev.json'], 'curve_k2': ['duo/curve27_k2_track.json'],
 'curve_k3': ['curve27_k3_order.json', 'duo/curve27_k3_order_rev.json'],
 'curve_k4': ['curve27_k4_truth.json', 'curve27_k4_truth_mid.json', 'duo/curve27_k4_truth_rev.json'],
 'curve_k5': ['curve27_k5_walk.json', 'curve27_k5_walk_mid.json', 'duo/curve27_k5_walk_rev.json']}
for a, fs in ARMS.items():
    out, seen, src = [], set(), {}
    for f in fs:
        rs, q = rows(f)
        if q is None: continue
        ts = {r['task'] for r in rs} - seen
        out += [r for r in rs if r['task'] in ts]; seen |= ts; src[q] = len(ts)
        if f.endswith('_ALL.json') and rs: break
    # completeness: a task is complete if it has its full item count (partials only hold finished tasks)
    json.dump({'arm': a, 'sources': src, 'n_tasks': len(seen), 'rows': out}, open(f'{F}/{a}.json', 'w'))
    print(f'{a:16s} tasks={len(seen):2d} sources={src}')
