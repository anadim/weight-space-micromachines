"""lane seqver (copy of attnver/repick_av.py): offline re-pick through the rung-4 BLD writer with an attention increment delta added to the per-slot blend.
python3.11 repick_av.py --sc job/out/sc_TAG.npz [--cfgs CFG,...] [--l2 1e-3,1e-2] [--save CFG@MODE@L2]
Per slot b' = max(sn,-12) + max(lg,-12) + delta  (delta = 'inc' model output; alone-scores are used as delta * scale 0 -> skip).
Eval rows = the rung-4 rows (implants/sidenet/aw/out/sn_{dev,dev2}, 4 seeds; in-stack R_sn 'sn' per slot) -> delta = 0 with the
rung-4 theta reproduces the rung-4 pick exactly (checked). trainC = tc_trainC rows with the g4 OOF sn (as the rung-4 theta fit).
Modes: X0 = rung-4 theta unchanged (zero extra parameters), BLD = fresh theta fit on trainC (OOF delta; as the rung-4 theta),
5-fold item CV on trainC; trainC CV f/b are vs the rung-4 writer (theta fit in-sample on trainC, so slightly optimistic for r4).
Fixes/breaks vs the rung-4 row (its in-stack pick), summed over the 4 seeds; one-sided sign test."""
import argparse, gzip, json, os, sys
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser('gsm_stack/implants/vfrefit/off'))
import refit as RF
from refit import sig, fit, acc_row, signp

ROWS = {'trainC': os.path.expanduser('gsm_stack/implants/vfrefit/aw/out/tc_trainC'),
        'dev': os.path.expanduser('gsm_stack/implants/sidenet/aw/out/sn_dev'),
        'dev2': os.path.expanduser('gsm_stack/implants/sidenet/aw/out/sn_dev2')}
TAB = os.path.expanduser('gsm_stack/implants/sidenet/tables/attempts_{}.jsonl.gz')
V5 = ('lg', 'sane', 'nn', 'zt', 'qc')
STAT_NOLG = [f'{a}_{b}' for a in ('mean', 'max', 'min') for b in V5 if b != 'lg']
BLD = ['c8', 'lc', 'ssb'] + STAT_NOLG + ['mean_b', 'max_b', 'min_b']


class St(RF.St):
    def __init__(self, r, sn, de):
        super().__init__(r)
        if not len(self.rep): return
        n = self.n
        M = np.array([[self.keep[j] and tuple(r['sym'][j]) == tuple(r['sym'][self.rep[q]]) for j in range(n)]
                      for q in range(len(self.rep))], float)
        cnt = M.sum(1); f = self.f
        lgc = np.maximum(np.array(r['lg'][:n], np.float64), -12)
        b = np.maximum(np.array(sn[:n], np.float64), -12) + lgc + np.array(de[:n], np.float64)
        f['ssb'] = M @ sig(b / 2); f['mean_b'] = (M @ b) / cnt; f['max_b'] = np.where(M > 0, b[None], -1e9).max(1)
        f['min_b'] = np.where(M > 0, b[None], 1e9).min(1)


def table_map(S, vec, shuf=False):
    rows_t = [json.loads(l) for l in gzip.open(TAB.format(S), 'rt')]
    assert len(vec) == len(rows_t), (S, len(vec), len(rows_t))
    if shuf and os.environ.get('AV_SHUF'):                       # noise control: permute delta among the attempts of the same item
        vec = np.array(vec, np.float64).copy(); rng = np.random.default_rng(int(os.environ['AV_SHUF']))
        ix = np.array([r['idx'] for r in rows_t])
        for it in np.unique(ix):
            j = np.flatnonzero(ix == it); vec[j] = vec[rng.permutation(j)]
    return {(r['seed'], r['idx'], r['k']): float(x) for r, x in zip(rows_t, vec)}


def load_set(S, de_vec, sn_tc, scale):
    dm = table_map(S, de_vec, True) if de_vec is not None else {}
    sm = table_map(S, sn_tc) if S == 'trainC' else None
    D = {}
    for sd, rows in RF.load(ROWS[S]).items():
        out = []
        for r in rows:
            n = len(r['lg'])
            de = [scale * dm.get((sd, r['idx'], k), 0.0) for k in range(n)]
            sn = [sm.get((sd, r['idx'], k), -30.0) for k in range(n)] if sm is not None else r['sn']
            s = St(r, sn, de); s.r4ok = bool(r['ok']) if S != 'trainC' else None
            out.append(s)
        D[sd] = out
    return D


def pick(s, names, th, c0):
    if not len(s.rep): return -1
    sc = s.G(names) @ th + c0
    return int(s.rep[int(np.argmax(sc - 1e-9 * s.rep))])


def evaluate(D, th, c0, tag, allt=None, cvth=None):
    line = tag
    if cvth is not None:
        cv = np.array([(acc_row(s, pick(s, BLD, *cvth[s.idx % 5])), b0) for s, b0 in zip(allt, BASE_TC)])
        fx = int((cv[:, 0] & ~cv[:, 1]).sum()); br = int((~cv[:, 0] & cv[:, 1]).sum())
        line += f' | trainC CV {100 * cv[:, 0].mean():.2f} ({100 * (cv[:, 0].mean() - cv[:, 1].mean()):+.2f} vs r4-theta, {fx}/{br})'
    tot = [0, 0]
    for S in ('dev', 'dev2'):
        accs, fxs, brs, per = [], 0, 0, []
        for sd, v in sorted(D[S].items()):
            new = np.array([acc_row(s, pick(s, BLD, th, c0)) for s in v]); old = np.array([s.r4ok for s in v])
            accs.append(100 * new.mean()); fxs += int((new & ~old).sum()); brs += int((~new & old).sum()); per.append(f'{100 * new.mean():.2f}')
        old4 = np.mean([100 * np.mean([s.r4ok for s in v]) for v in D[S].values()])
        line += f' | {S} {np.mean(accs):.2f} ({np.mean(accs) - old4:+.2f}, {fxs}/{brs}; {"/".join(per)})'
        tot[0] += fxs; tot[1] += brs
    return line + f' | pooled {tot[0]}/{tot[1]} p={signp(*tot):.3f}', tot


def main():
    global th0, c00, BASE_TC
    ap = argparse.ArgumentParser()
    ap.add_argument('--sc', required=True)
    ap.add_argument('--cfgs', default='')
    ap.add_argument('--l2', default='1e-3,1e-2')
    ap.add_argument('--scales', default='1')
    ap.add_argument('--save', default='')
    a = ap.parse_args()
    Wt = json.load(open(os.path.expanduser('gsm_stack/implants/sidenet/vfsn_BLD_0.001.json')))
    assert Wt['feats'] == BLD
    th0 = np.array(Wt['th'], np.float64); c00 = float(Wt['c0'])
    sn_tc = np.load(os.path.join(H, 'job/tables/sc_g4.npz'))['c4:HM:lin:pair|trainC']
    sc = dict(np.load(a.sc))
    cfgs = a.cfgs.split(',') if a.cfgs else sorted({k.split('|')[0] for k in sc if ':inc:' in k})
    # control: delta = 0, rung-4 theta -> must reproduce the rung-4 picks exactly
    D0 = {S: load_set(S, None, sn_tc, 0) for S in ('dev', 'dev2')}
    for S in ('dev', 'dev2'):
        mm = sum(pick(s, BLD, th0, c00) != s.pick for v in D0[S].values() for s in v)
        print(f'control {S}: delta=0 rung-4 theta pick mismatches vs rung-4 rows = {mm}', flush=True); assert mm == 0
    T0 = load_set('trainC', None, sn_tc, 0); allt0 = [s for v in T0.values() for s in v]
    BASE_TC = [acc_row(s, pick(s, BLD, th0, c00)) for s in allt0]          # rung-4 writer on trainC (in-sample theta)
    th_c, c_c = fit(allt0, BLD, 1e-3)
    print('control: delta=0 fresh trainC fit l2 1e-3 vs rung-4 theta: max |dtheta| %.2e' % np.abs(th_c - th0).max(), flush=True)
    for cfg in cfgs:
        for scale in map(float, a.scales.split(',')):
            D = {S: load_set(S, sc[f'{cfg}|{S}'], sn_tc, scale) for S in ('trainC', 'dev', 'dev2')}
            allt = [s for v in D['trainC'].values() for s in v]
            print(f'## {cfg} scale {scale:g}', flush=True)
            print(evaluate(D, th0, c00, 'X0 (r4 theta)')[0], flush=True)
            for l2 in map(float, a.l2.split(',')):
                cvth = {k: fit([s for s in allt if s.idx % 5 != k], BLD, l2) for k in range(5)}
                th, c0 = fit(allt, BLD, l2)
                print(evaluate(D, th, c0, f'BLD l2={l2:g}', allt, cvth)[0], flush=True)
                if a.save == f'{cfg}@{scale:g}@{l2:g}':
                    json.dump(dict(th=th.tolist(), c0=float(c0), feats=BLD, l2=l2, cfg=cfg, scale=scale),
                              open(os.path.join(H, f'vfsq_{l2:g}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
