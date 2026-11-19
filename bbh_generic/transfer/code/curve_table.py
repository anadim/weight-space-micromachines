import json, sys, os, gzip
F = sys.argv[1]
T12 = {'boolean_expressions','logical_deduction_five_objects','logical_deduction_seven_objects','logical_deduction_three_objects','multistep_arithmetic_two','navigate','object_counting','temporal_sequences','tracking_shuffled_objects_five_objects','tracking_shuffled_objects_seven_objects','tracking_shuffled_objects_three_objects','web_of_lies'}
def st(p):
    if not os.path.exists(p): p += '.gz'
    with (gzip.open(p, 'rt') if p.endswith('.gz') else open(p)) as f: rows = json.load(f)['rows']; per = {}; call = {}
    for r in rows: per.setdefault(r['task'], []).append(r['ok']); call.setdefault(r['task'], []).append(('<tool_call>' in r['out']) if 'out' in r else r['n_calls'] > 0)
    acc = {t: 100*sum(v)/len(v) for t, v in per.items()}
    m = lambda s: (sum(s)/len(s), len(s)) if s else (float('nan'), 0)
    n15 = [t for t in per if t not in T12]
    c15 = 100*sum(sum(call[t]) for t in n15)/sum(len(call[t]) for t in n15) if n15 else float('nan')
    return m(list(acc.values())), m([acc[t] for t in acc if t in T12]), m([acc[t] for t in acc if t not in T12]), c15, acc
rows = [('k0', '(none) = v2 FT no tools', 'q06_ft_notools'), ('k1', '+ interval', 'curve_k1'), ('k2', '+ track', 'curve_k2'), ('k3', '+ order', 'curve_k3'),
        ('k4', '+ truth', 'curve_k4'), ('k5', '+ walk', 'curve_k5'), ('k6', '+ Team C passive (expr/calc/bool) = v2 FT + all 8 tools', 'q06_ft_tools')]
print('| k | tool added | 27-mean (n tasks) | 12-mean (n) | 15-mean (n) | no-tool call rate % |'); print('|---|---|---|---|---|---|')
allacc = {}
for k, lab, f in rows:
    a, b, c, cr, acc = st(f'{F}/{f}.json'); allacc[k] = acc
    fl = '' if a[1] == 27 else ' **INCOMPLETE**'
    print(f'| {k} | {lab} | {a[0]:.2f} ({a[1]}){fl} | {b[0]:.2f} ({b[1]}) | {c[0]:.2f} ({c[1]}) | {cr:.1f} |')
print('\nPer-task (strict %):\n| task | tool |' + ''.join(f' {k} |' for k, _, _ in rows)); print('|---|---|' + '---|'*len(rows))
for t in sorted(allacc['k0']):
    print(f"| {t} | {'Y' if t in T12 else ''} |" + ''.join(f" {allacc[k][t]:.1f} |" if t in allacc[k] else ' - |' for k, _, _ in rows))
