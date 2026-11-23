"""(d) §5.2 scrambled control on GSM8K: LIB (correct implant library) vs LIBscr (every implant but the calculator scrambled),
300 updates each, RL seeds 0 / 1 / 2. Prints the pair table, the paired item bootstrap of LIB − LIBscr (s1, s2, pooled; 2000
resamples, numpy-free copy of rl/library/analysis.py's) and the "big numbers" dev check (scaled half, implants on → off).
    cd checks/seeds3 && python3 libpair.py       (CPU, about 5 s)
Inputs: res/rlctl/<run>/{test_eval.json,steps.jsonl} (s1, s2 on B200); seed 0 (H100) and the original LIB seed 1 (A40) from
$WSM_ROOT/rl/results/library/figdata/{LIB_s0,LIBscr_s0,LIB_s1}_steps.jsonl. S3_RES overrides res/."""
import json, os, random

H = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(os.environ.get('S3_RES', os.path.join(H, 'res')), 'rlctl')
FD = os.path.join(os.environ.get('WSM_ROOT', os.path.join(H, '..', '..')), 'rl', 'results', 'library', 'figdata')
L = {n: json.load(open(f'{D}/{n}/test_eval.json')) for n in ('LIB_s1', 'LIBscr_s1', 'LIB_s2', 'LIBscr_s2')}


def boot_diff(A, B, nb=2000, seed=0):   # A, B: lists of per-item 0/1 lists (runs); item bootstrap shared across runs
    n = len(A[0]); rng = random.Random(seed)
    pa = [sum(r[j] for r in A) / len(A) for j in range(n)]; pb = [sum(r[j] for r in B) / len(B) for j in range(n)]
    d = [pa[j] - pb[j] for j in range(n)]
    ms = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(nb))
    return 100 * sum(d) / n, 100 * ms[int(0.025 * nb)], 100 * ms[int(0.975 * nb) - 1]


def steps(run):
    p = f'{FD}/{run}_steps.jsonl' if run.endswith('_s0') or run == 'LIB_s1_A40' else f'{D}/{run}/steps.jsonl'
    if run == 'LIB_s1_A40': p = f'{FD}/LIB_s1_steps.jsonl'
    X = [json.loads(l) for l in open(p)]
    upd = {x['u']: x for x in X if 'calls' in x and 'roll_s' in x}
    dev = [x['dev'] for x in X if 'dev' in x]
    off = [x['dev_toolsoff'] for x in X if 'dev_toolsoff' in x]
    test = [x['test'] for x in X if 'test' in x]
    return upd, dev, off, (test[-1] if test else L.get(run))


print('| pair | LIB GSM8K / Hard (calc-only) | LIBscr GSM8K / Hard (calc-only) | LIB − LIBscr | LIBscr test calls/q; train calls/q u1 → u25 → u300; dev calc fire u300 |')
print('|---|---|---|---|---|')
M = []
for sd, gpu in (('s0', 'H100'), ('s1', 'B200'), ('s2', 'B200')):
    _, _, _, a = steps(f'LIB_{sd}'); u, dv, _, b = steps(f'LIBscr_{sd}')
    acc = lambda t, k: t[k]['acc']
    M.append([acc(a, 'test_full'), acc(a, 'hard_full'), acc(b, 'test_full'), acc(b, 'hard_full')])
    print(f'| {sd} ({gpu}) | {acc(a, "test_full"):.2f} / {acc(a, "hard_full"):.2f} ({acc(a, "test_calconly"):.2f} / {acc(a, "hard_calconly"):.2f}) '
          f'| {acc(b, "test_full"):.2f} / {acc(b, "hard_full"):.2f} ({acc(b, "test_calconly"):.2f} / {acc(b, "hard_calconly"):.2f}) '
          f'| {M[-1][0] - M[-1][2]:+.2f} / {M[-1][1] - M[-1][3]:+.2f} '
          f'| {b["test_full"]["calls"]:.2f}; {u[1]["calls"]:.2f} → {u[25]["calls"]:.2f} → {u[300]["calls"]:.2f}; {100 * dv[-1]["dev_t_calc"]["fire"]:.0f}% |')
m = [sum(r[i] for r in M) / 3 for i in range(4)]
print(f'| mean (3) | {m[0]:.2f} / {m[1]:.2f} | {m[2]:.2f} / {m[3]:.2f} | {m[0] - m[2]:+.2f} / {m[1] - m[3]:+.2f} | |')
print('\npaired item bootstrap, LIB − LIBscr (pp, 95% CI):')
for k in ('test_full', 'hard_full'):
    for sd in ('s1', 's2'):
        print(k, sd, '%+.2f [%+.2f, %+.2f]' % boot_diff([L[f'LIB_{sd}'][k]['per_item']], [L[f'LIBscr_{sd}'][k]['per_item']]))
    print(k, 'pooled s1+s2', '%+.2f [%+.2f, %+.2f]' % boot_diff([L['LIB_s1'][k]['per_item'], L['LIB_s2'][k]['per_item']], [L['LIBscr_s1'][k]['per_item'], L['LIBscr_s2'][k]['per_item']]))
print('\nbig numbers (dev u300, scaled half), implants on → all off:')
on = []; offv = []
for run, lab in (('LIB_s0', 's0 (H100)'), ('LIB_s1_A40', 's1 (A40, original)'), ('LIB_s1', 's1 (B200)'), ('LIB_s2', 's2 (B200)')):
    _, dv, off, _ = steps(run)
    on.append(dv[-1]['dev_scaled']); offv.append(off[-1]['devtoolsoff_scaled'])
    print(f'- LIB {lab}: {on[-1]:.2f} → {offv[-1]:.2f}')
print(f'- mean ({len(on)} runs): {sum(on) / len(on):.2f} → {sum(offv) / len(offv):.2f}')
