"""(c) §4.4 compliant S2F vs the no-implant host row H (a2_u300_s2 = 'u0', GSM8K test n=1319, T=1 seed 7).
Paired item bootstrap (items aligned on 'i'), 10,000 resamples, 95% percentile CI; exact two-sided McNemar p.
Sanity row: the old non-compliant S2F (post-sampling swap hook, test_u0_B, 72.25) vs H must reproduce RL_RESULTS §4's +4.55 [2.58, 6.44].
python3 checks/sourced/s2f_vs_h.py   (input: res/fe/test_*.items.jsonl, reduced fe_hook rows, shipped)"""
import json, os, random
from math import comb
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res', 'fe')
B = 10000


def load(tag): return {r['i']: float(r['ok']) for r in map(json.loads, open(f'{D}/test_{tag}.items.jsonl'))}


def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n)


def boot(d, seed=0):
    rng, n = random.Random(seed), len(d)
    bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return 100 * bs[int(0.025 * B)], 100 * bs[int(0.975 * B) - 1]


H = load('lad_H')
print('| config (file) | acc | Δ vs H (67.70) [95% CI] | fixed / broken | McNemar p |')
print('|---|---|---|---|---|')
for lab, tag in (('S2F compiled logit-write trigger (S2Fw_u0, miss3 j23342)', 'S2Fw_u0'),
                 ('S2F residual-write trigger (S2Frw_u0, mw_test after mwchk j23562)', 'S2Frw_u0'),
                 ('old S2F, post-sampling swap hook — NOT compliant (u0_B, sanity)', 'u0_B')):
    X = load(tag)
    it = sorted(set(H) & set(X))
    d = [X[i] - H[i] for i in it]
    lo, hi = boot(d)
    b, c = sum(v == 1 for v in d), sum(v == -1 for v in d)
    print(f'| {lab} | {100 * sum(X[i] for i in it) / len(it):.2f} | {100 * sum(d) / len(d):+.2f} [{lo:+.2f}, {hi:+.2f}] | {b} / {c} | {mcnemar(b, c):.2g} |')
print(f'\nn = {len(it)} items; H = test_lad_H {100 * sum(H.values()) / len(H):.2f}')
