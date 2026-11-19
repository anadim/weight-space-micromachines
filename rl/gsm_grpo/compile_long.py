"""300-update comparison on the H100 from fetched files: fetched/long_h100/final/*.json (final_eval) and
fetched/long_h100/steps/<run>.jsonl (dev curves). Paired bootstrap (questions; seeds averaged per question).
python compile_long.py"""
import json, glob, os, random
F = os.environ.get('RL_RESULTS_H100', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'gsm_grpo', 'H100'))
fin = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(f'{F}/final/*.json')}
f = lambda v: '—' if v is None else f'{v:.1f}'
print('| model (H100, u300) | GSM8K on / off | GSM-Hard on / off | GSM-Hard 201+ on / off | rethink frac (test on) |')
print('|---|---|---|---|---|')
for k in sorted(fin):
    d = fin[k]
    g = lambda s, m, key='acc': d.get(f'{s}_{m}', {}).get(key)
    print(f"| {k} | {f(g('test','on'))} / {f(g('test','host'))} | {f(g('hard','on'))} / {f(g('hard','host'))} | "
          f"{f(g('hard','on','acc_201plus'))} / {f(g('hard','host','acc_201plus'))} | {f(100 * g('test','on','rethink_frac')) if g('test','on','rethink_frac') is not None else '—'} |")
random.seed(0)


def items(n, s, m):
    return fin[n][f'{s}_{m}']['per_item']


def ci(d, B=2000):
    n = len(d); m = sum(d) / n; bs = sorted(sum(random.choice(d) for _ in range(n)) / n for _ in range(B))
    return 100 * m, 100 * bs[int(.025 * B)], 100 * bs[int(.975 * B)]


avg = lambda a, b: [(x + y) / 2 for x, y in zip(a, b)]
print()
for s in ('test', 'hard'):
    try:
        a1 = avg(items('H_a1_s0_u300', s, 'host'), items('H_a1_s1_u300', s, 'host'))
        a2 = avg(items('H_a2_s0_u300', s, 'on'), items('H_a2_s1_u300', s, 'on'))
        a2off = avg(items('H_a2_s0_u300', s, 'host'), items('H_a2_s1_u300', s, 'host'))
        a4 = avg(items('H_a4_s0_u300', s, 'on'), items('H_a4_s1_u300', s, 'on'))
        a4off = avg(items('H_a4_s0_u300', s, 'host'), items('H_a4_s1_u300', s, 'host'))
    except KeyError as e:
        print('missing', e); continue
    for name, x, y in [('arm2(on) - arm1', a2, a1), ('arm2(off) - arm1', a2off, a1), ('arm4(on) - arm2(on)', a4, a2),
                       ('arm4(on) - arm1', a4, a1), ('arm4(off) - arm2(off)', a4off, a2off)]:
        m, lo, hi = ci([p - q for p, q in zip(x, y)])
        print(f'{s:5s} {name:24s} {m:+6.2f} pp  95% CI [{lo:+.2f}, {hi:+.2f}]')
