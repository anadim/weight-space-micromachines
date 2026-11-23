"""(d) §5.2 '<calc> calls fade 80% -> 25%; no '=' implants: 88%; corrupted '=' implant: 95-98%'.
Reproduces rl/figures/v3/make_new_v3.py calc_share()/rl7(): share (%) of GSM rollout rows containing <calc> per RL update
(calc_share_*.jsonl 'u': {update: [n_gsm_rows, n_calc, n_eq]}, made by calc_share_remote.py); start = mean of per-update
shares over u1-5, end = mean over u41-50 (unweighted by row count, as in the figure). Also reports the row-pooled end share.
python3 checks/sourced/calc_fade.py   (input: rl/figures/v3/data, shipped)"""
import json, glob, os, statistics as S
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.environ.get('WSM_ROOT') or os.path.join(HERE, '..', '..')
D = f'{ROOT}/rl/figures/v3/data'
R = {}
for f in sorted(glob.glob(f'{D}/calc_share_*.jsonl')):
    for l in open(f):
        d = json.loads(l); R[d['run']] = ({int(u): v for u, v in d['u'].items()}, os.path.basename(f))
conds = [('exact "=" implant (control)', ['causal_i_control', 'causal_i_control_s1']),
         ('no "=" implants', ['causal_ii_noPassive', 'causal_ii_noPassive_s1']),
         ('"=" implant 25% corrupted', ['compete_c9_s0', 'compete_c9_s1', 'compete_c9_s2']),
         ('"=" implant 55% corrupted', ['compete_c1_s0', 'compete_c1_s1', 'compete_c1_s2']),
         ('"=" implant 87% corrupted', ['compete_c3_s0', 'compete_c3_s1', 'compete_c3_s2'])]
share = lambda U, us: S.mean(100 * U[u][1] / U[u][0] for u in us)
pooled = lambda U, us: 100 * sum(U[u][1] for u in us) / sum(U[u][0] for u in us)
print('| condition | run (file) | last u | u1–5 | u41–50 | u41–50 pooled (n rows) |')
print('|---|---|---|---|---|---|')
summ = []
for lab, runs in conds:
    st, en = [], []
    for r in runs:
        U, f = R[r]
        a, b = share(U, range(1, 6)), share(U, range(41, 51))
        st.append(a); en.append(b)
        print(f'| {lab} | {r} ({f}) | {max(U)} | {a:.1f} | {b:.1f} | {pooled(U, range(41, 51)):.1f} ({sum(U[u][0] for u in range(41, 51))}) |')
    summ.append((lab, S.mean(st), S.mean(en), min(en), max(en)))
print('\n| condition | mean u1–5 | mean u41–50 | seed range u41–50 |\n|---|---|---|---|')
for lab, a, b, lo, hi in summ:
    print(f'| {lab} | {a:.1f} | {b:.1f} | {lo:.1f}–{hi:.1f} |')
