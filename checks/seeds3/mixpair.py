"""(e) mixed task, 3 seeds: all implants (m1m = m1 with --cw_mask 1) vs plain RL from the worked-solution SFT host (aNL) and
from the tagged SFT host (m3, seeds 1-2); (f) distilled links W3d, 3 seeds: GSM-Hard links on vs host only.
    cd checks/seeds3 && python3 mixpair.py > mixpair.md       (CPU, numpy, about 2 min)
Inputs: res/rlctl/{aNL,m1m,m3,W3d}_s*/{test_eval.json,steps.jsonl} (B200 runs); the original one-seed points from
$WSM_ROOT/rl/results/library/figdata/mix_m1_mixLIB_test.json and rl/results/wiring/figdata/W3d_s{0,1}.jsonl. S3_RES overrides res/.
CIs: percentile item bootstrap (10000 resamples, numpy default_rng(0), one generator for the whole script). A seed mean
resamples the same item indices for every seed (the CI is item-sampling uncertainty of the mean; seed SD is separate);
paired differences resample items jointly for both arms. The run analysis that produced these tables also bootstrapped two
sections that are not part of this check (28 draws of 1319 items between (e) and (f)); --burn 28 (default) skips the same
draws so that the (f) CIs match it exactly; --burn 0 gives equally valid CIs that differ in the last digit."""
import argparse, json, os
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument('--B', type=int, default=10000); ap.add_argument('--burn', type=int, default=28)
a = ap.parse_args()
D = os.path.join(os.environ.get('S3_RES', os.path.join(H, 'res')), 'rlctl')
RES = os.path.join(os.environ.get('WSM_ROOT', os.path.join(H, '..', '..')), 'rl', 'results')
RNG = np.random.default_rng(0)
P = print


def jload(p):
    try: return json.load(open(p))
    except Exception: return None


def runs(arm):
    return sorted(r for r in os.listdir(D) if r.startswith(arm + '_s') and os.path.isdir(os.path.join(D, r)))


def steps(r):
    return [json.loads(l) for l in open(os.path.join(D, r, 'steps.jsonl'))]


def boot(mats, f):
    n = mats[0].shape[1]
    idx = RNG.integers(0, n, size=(a.B, n))
    return np.percentile(np.array([f([m[:, i] for m in mats]) for i in idx]), [2.5, 97.5])


def ci_mean(m):
    m = np.asarray(m, float); lo, hi = boot([m], lambda x: x[0].mean())
    return 100 * m.mean(), 100 * lo, 100 * hi


def ci_diff(m1, m2):
    m1, m2 = np.asarray(m1, float), np.asarray(m2, float)
    lo, hi = boot([m1, m2], lambda x: x[0].mean() - x[1].mean())
    return 100 * (m1.mean() - m2.mean()), 100 * lo, 100 * hi


fmt = lambda t: f'{t[0]:.1f} [{t[1]:.1f}, {t[2]:.1f}]'
fmtd = lambda t: f'{t[0]:+.1f} [{t[1]:+.1f}, {t[2]:+.1f}]'
sd = lambda xs: f'{np.std(xs, ddof=1):.1f}' if len(xs) > 1 else '-'


def dev_curve(st, key='dev', acc='dev_acc'):
    return [(d[key]['u'], d[key].get(acc)) for d in st if key in d and isinstance(d[key], dict) and 'u' in d[key]]


def upd_s(st):
    xs = [d['upd_s'] for d in st if 'upd_s' in d] or [d['roll_s'] + d['train_s'] for d in st if 'roll_s' in d and 'train_s' in d]
    return f'{np.median(xs):.0f}' if xs else '-'


# ---------------------------------------------------------------- (e) mixed task: 400 stratified test items, u150
ORIG = {'m1 (all implants, cw_mask 0, s0)': dict(full=68.25, gsm=50.9, bbh=87.6, combo=36.1, ifeval=81.5, toolsoff=36.8, hostonly=31.8),
        'm3 (plain RL, tagged host, s0)': dict(full=36.3, gsm=31.8, bbh=46.0, combo=14.8, ifeval=41.3)}
_m1 = jload(f'{RES}/library/figdata/mix_m1_mixLIB_test.json')
if _m1:
    ORIG['m1 (all implants, cw_mask 0, s0)'].update(full=_m1['full']['test_acc'], toolsoff=_m1['toolsoff']['test_acc'], hostonly=_m1['hostonly']['test_acc'])
TYPES = dict(g='gsm', b='bbh', c='combo', i='ifeval')
MIX = {arm: {r: jload(os.path.join(D, r, 'test_eval.json')) for r in runs(arm)} for arm in ('aNL', 'm1m', 'm3')}
LAB = dict(aNL='aNL: plain RL from the worked-solution SFT host', m1m='m1m: all implants (m1, --cw_mask 1)',
           m3='m3: plain RL from the tagged SFT host')
P('# (e) mixed task and (f) distilled links, 3 seeds (95% item-bootstrap CIs)\n')
P('## (e) mixed task (GSM / BBH / combo / IFEval; test 400, stratified; u150; B200)\n')
P('| arm | seeds | full | gsm | bbh | combo | ifeval | tools-off | host-only | seed SD | gen tok | s/upd |')
P('|---|---|---|---|---|---|---|---|---|---|---|---|')
for arm, R in MIX.items():
    names = sorted(R); per = np.array([R[r]['full']['test_per_item'] for r in names])
    ty = R[names[0]]['full']['test_item_types']; assert all(R[r]['full']['test_item_types'] == ty for r in names)
    cells = [fmt(ci_mean(per))]
    for k in TYPES:
        j = [i for i, c in enumerate(ty) if c == k]; cells.append(fmt(ci_mean(per[:, j])))
    for md in ('toolsoff', 'hostonly'):
        ok = all(isinstance(R[r].get(md), dict) and R[r][md].get('test_per_item') for r in names)
        cells.append(fmt(ci_mean([R[r][md]['test_per_item'] for r in names])) if ok else '-')
    P(f'| {LAB[arm]} | {",".join(n.split("_s")[-1] for n in names)} | ' + ' | '.join(cells) +
      f' | {sd([100 * np.mean(x) for x in per])} | {np.mean([R[r]["full"]["test_len"] for r in names]):.0f} | {"/".join(upd_s(steps(n)) for n in names)} |')
for k, v in ORIG.items():
    P(f'| original {k} (point) | 0 | {v["full"]:.1f} | {v["gsm"]:.1f} | {v["bbh"]:.1f} | {v["combo"]:.1f} | {v["ifeval"]:.1f} | '
      + ' | '.join(f'{v[x]:.1f}' if x in v else '-' for x in ('toolsoff', 'hostonly')) + ' | - | - | - |')
P('\nPaired differences (same 400 items; seed means):\n')
P('| comparison | full | gsm | bbh | combo | ifeval |'); P('|---|---|---|---|---|---|')
for x, y in (('aNL', 'm3'), ('m1m', 'm3'), ('m1m', 'aNL')):
    px = np.array([MIX[x][r]['full']['test_per_item'] for r in sorted(MIX[x])])
    py = np.array([MIX[y][r]['full']['test_per_item'] for r in sorted(MIX[y])])
    tx = MIX[x][sorted(MIX[x])[0]]['full']['test_item_types']
    assert tx == MIX[y][sorted(MIX[y])[0]]['full']['test_item_types'] and px.shape[1] == py.shape[1]
    cells = [fmtd(ci_diff(px, py))]
    for k in TYPES:
        j = [i for i, c in enumerate(tx) if c == k]; cells.append(fmtd(ci_diff(px[:, j], py[:, j])))
    P(f'| {x} − {y} | ' + ' | '.join(cells) + ' |')
R = MIX['m1m']; names = sorted(R); full = np.array([R[r]['full']['test_per_item'] for r in names])
P('\nm1m leave-one-out on test (full minus the run with that implant detached, seed mean):\n')
P('| implant | full − LOO |'); P('|---|---|')
for k in R[names[0]]:
    if not k.startswith('loo_'): continue
    if not all(isinstance(R[r].get(k), dict) for r in names): P(f'| {k[4:]} | identical by construction (some seed) |'); continue
    P(f'| {k[4:]} | {fmtd(ci_diff(full, [R[r][k]["test_per_item"] for r in names]))} |')
for k in ('trained_host_orig_tools', 'orig_host_trained_tools'):
    if all(isinstance(R[r].get(k), dict) for r in names):
        P(f'| swap: {k} (acc) | {fmt(ci_mean([R[r][k]["test_per_item"] for r in names]))} |')
P('\nDev curves (dev_mixed 400; u: acc):\n')
for arm in MIX:
    for r in sorted(MIX[arm]):
        P(f'- {r}: ' + ', '.join(f'{u}: {v:.1f}' for u, v in dev_curve(steps(r)) if v is not None))
P('- original m1 dev: 0: 44.5, 25: 49.8, 50: 53.5, 100: 64.2, 150: 65.5')

# ---------------------------------------------------------------- (f) W3d: distilled links, RL 300 u, GSM8K test / GSM-Hard
for _ in range(a.burn): RNG.integers(0, 1319, size=(a.B, 1319))
P(f'\n## (f) W3d: links distilled from the hand wiring, then RL 300 u (B200; RNG burn {a.burn})\n')
P('| seed | dev u300 on / links reset | dev links effect | GSM8K test full | test host-only | test full − host-only | GSM-Hard full | Hard host-only | Hard full − host-only | s/upd |')
P('|---|---|---|---|---|---|---|---|---|---|')
eff = []
for r in runs('W3d'):
    st = steps(r); dv = dict(dev_curve(st)); rs = dict(dev_curve(st, 'dev_reset_curve', 'devreset_acc'))
    t = jload(os.path.join(D, r, 'test_eval.json'))
    c = {k: fmt(ci_mean([t[k]['per_item']])) for k in ('test_full', 'test_hostonly', 'hard_full', 'hard_hostonly')}
    d = {k: ci_diff([t[k + '_full']['per_item']], [t[k + '_hostonly']['per_item']]) for k in ('test', 'hard')}
    eff.append(d['hard'][0])
    P(f'| {r.split("_")[-1]} | {dv[300]:.1f} / {rs[300]:.1f} | {dv[300] - rs[300]:+.1f} | {c["test_full"]} | {c["test_hostonly"]} | {fmtd(d["test"])} | '
      f'{c["hard_full"]} | {c["hard_hostonly"]} | {fmtd(d["hard"])} | {upd_s(st)} |')
P(f'\nGSM-Hard links on − host only, mean over {len(eff)} seeds: {np.mean(eff):+.1f} (seed SD {np.std(eff, ddof=1):.1f})')
P('\nDev curves (dev 800; u: links on / links reset):\n')
for r in runs('W3d'):
    st = steps(r); dv = dict(dev_curve(st)); rs = dict(dev_curve(st, 'dev_reset_curve', 'devreset_acc'))
    P(f'- {r}: ' + ', '.join(f'{u}: {dv[u]:.1f}/{rs[u]:.1f}' for u in sorted(dv) if u in rs))
P('- original W3d dev u300 links effect (dev − reset): s0 +11.3, s1 +11.5 (NOTEBOOK); s2 +10.6 at u250 (OOM at u251)')
for s in (0, 1):
    p = f'{RES}/wiring/figdata/W3d_s{s}.jsonl'
    if os.path.exists(p):
        st = [json.loads(l) for l in open(p)]; dv = dict(dev_curve(st)); rs = dict(dev_curve(st, 'dev_reset_curve', 'devreset_acc'))
        if 300 in dv and 300 in rs: P(f'  - recomputed from figdata W3d_s{s}: {dv[300]:.1f} − {rs[300]:.1f} = {dv[300] - rs[300]:+.1f}')
