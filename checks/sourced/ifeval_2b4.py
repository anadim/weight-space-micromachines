"""(e) §5.1 IFEval (Qwen3-0.6B-Base, rl24): RL then attach (row 2b) vs RL with the implant attached (row 4), paired.
Per prompt: seed-averaged prompt-strict indicator (seeds 0-2, as analyze.py); delta 2b - 4 with a 95% paired item
bootstrap (10k resamples of the 541 prompts, seed 2 = analyze.py's pairs). Also per-seed (s_k vs s_k) paired deltas,
and fixed/broken counts on the seed-majority (>=2 of 3) indicator. python3 checks/sourced/ifeval_2b4.py
(input: uniform/rl24/ifeval/res/{P,I}_s*/eval_u300.*.score.json.gz, shipped gzipped; plain .json also read)"""
import gzip, json, os, random
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.environ.get('WSM_ROOT') or os.path.join(HERE, '..', '..')
O = f'{ROOT}/uniform/rl24/ifeval/res'; B = 10000
_op = lambda p: gzip.open(p + '.gz', 'rt') if os.path.exists(p + '.gz') else open(p)
L = lambda d, f: {r['key']: r for r in json.load(_op(f'{O}/{d}/{f}.score.json'))['rows']}
R2b = [L(f'P_s{s}', 'eval_u300.imp') for s in range(3)]; R4 = [L(f'I_s{s}', 'eval_u300.imp') for s in range(3)]
R2 = [L(f'P_s{s}', 'eval_u300.plain') for s in range(3)]
K = sorted(R2b[0]); n = len(K); assert all(sorted(r) == K for r in R2b + R4 + R2)
ps = lambda R: [[float(all(r[k]['strict'])) for k in K] for r in R]
tq = lambda R: sum(float(r[k]['ntok'] or 0) for r in R for k in K) / (n * len(R))
def boot(x, y, seed):
    rng = random.Random(seed); d = [a - b for a, b in zip(x, y)]; out = []
    for _ in range(B):
        out.append(sum(d[rng.randrange(n)] for _ in range(n)) / n)
    out.sort(); return 100 * sum(d) / n, 100 * out[int(.025 * B)], 100 * out[int(.975 * B) - 1]
avg = lambda M: [sum(c) / len(c) for c in zip(*M)]
A, Bq, C = ps(R2b), ps(R4), ps(R2)
print(f'n = {n} prompts; tok/q 2b {tq(R2b):.0f}, 4 {tq(R4):.0f}, 2 {tq(R2):.0f}\n')
print('| comparison | Δ prompt-strict (pp) [95% CI] |\n|---|---|')
for lab, x, y in [('2b − 4, seed-averaged', avg(A), avg(Bq)), ('2b − 2, seed-averaged (check vs ledger +9.12)', avg(A), avg(C)),
                  ('4 − 2, seed-averaged (check vs ledger +4.81)', avg(Bq), avg(C))]:
    d, lo, hi = boot(x, y, 2); print(f'| {lab} | {d:+.2f} [{lo:+.2f}, {hi:+.2f}] |')
for s in range(3):
    d, lo, hi = boot(A[s], Bq[s], 2); print(f'| 2b − 4, seed {s} vs seed {s} | {d:+.2f} [{lo:+.2f}, {hi:+.2f}] |')
m = lambda M: [int(sum(c) >= 2) for c in zip(*M)]
a, b = m(A), m(Bq)
print(f'\nseed-majority prompt-strict: 2b {100*sum(a)/n:.2f}, 4 {100*sum(b)/n:.2f}; 2b right & 4 wrong {sum(x>y for x,y in zip(a,b))}, '
      f'4 right & 2b wrong {sum(y>x for x,y in zip(a,b))}')
print('per seed 2b / 4:', ' | '.join(f'{100*sum(A[s])/n:.2f} / {100*sum(Bq[s])/n:.2f}' for s in range(3)))
