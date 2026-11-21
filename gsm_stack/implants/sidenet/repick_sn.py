"""lane sidenet: offline re-pick on the rung-3 avf rows with the side-net score as a per-slot register feature.
python3.11 repick_sn.py --sc job/out/sc_TAG.npz [--cfgs CFG,...] [--modes ...]
Writer family = ClusterLin over answer groups (as aw_vf.vf_vote / vfrefit/off/refit.py), per-slot Fv extended with s = side-net
score; group features = the 18 base features (+ side-net stats, per mode). theta fitted on trainC rows (tc_trainC, 4 seeds) using
the side-net's trainC OUT-OF-FOLD scores; trainC CV = item-grouped 5 folds (idx % 5, the same folds as the side-net);
dev/dev2 = full-trainC fit, re-picked on the avf rows (same streams => pure re-pick) with the fold-ensemble side-net scores.
Paired fixes/breaks vs the row's in-stream VF pick, summed over the 4 seeds; one-sided sign test."""
import argparse, gzip, json, os, sys
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser(os.environ.get('WSM_ROOT', '.') + '/gsm_stack/implants/vfrefit/off'))
import refit as RF
from refit import sig, fit, acc_row, signp

ROWS = {'trainC': os.path.expanduser(os.environ.get('WSM_ROOT', '.') + '/gsm_stack/implants/vfrefit/aw/out/tc_trainC'),
        'dev': os.path.expanduser(os.environ.get('WSM_ROOT', '.') + '/gsm_stack/implants/ladr3/aw/out/avf_dev'),
        'dev2': os.path.expanduser(os.environ.get('WSM_ROOT', '.') + '/gsm_stack/implants/ladr3/aw/out/avf_dev2')}
V5 = ('lg', 'sane', 'nn', 'zt', 'qc')


class St(RF.St):
    def __init__(self, r, sn):
        super().__init__(r)
        if not len(self.rep): return
        n = self.n; s = np.maximum(np.array(sn[:n], np.float64), -12)
        M = np.array([[self.keep[j] and tuple(r['sym'][j]) == tuple(r['sym'][self.rep[q]]) for j in range(n)]
                      for q in range(len(self.rep))], float)
        cnt = M.sum(1); f = self.f
        f['ssn'] = M @ sig(s); f['mean_sn'] = (M @ s) / cnt
        f['max_sn'] = np.where(M > 0, s[None], -1e9).max(1); f['min_sn'] = np.where(M > 0, s[None], 1e9).min(1)
        # replace-lg variants: lg -> sn everywhere (sum sigmoid, mean/max/min)
        f['ssig_sn'] = f['ssn']
        lgc = np.maximum(np.array(r['lg'][:n], np.float64), -12); b = s + lgc
        f['ssb'] = M @ sig(b / 2); f['mean_b'] = (M @ b) / cnt; f['max_b'] = np.where(M > 0, b[None], -1e9).max(1)
        f['min_b'] = np.where(M > 0, b[None], 1e9).min(1)


STAT = [f'{a}_{b}' for a in ('mean', 'max', 'min') for b in V5]
STAT_NOLG = [f'{a}_{b}' for a in ('mean', 'max', 'min') for b in V5 if b != 'lg']
SN4 = ['ssn', 'mean_sn', 'max_sn', 'min_sn']
MODES = {
    'B18': ['c8', 'lc', 'ssig'] + STAT,                                   # control: fresh refit, no side-net
    'AUG': ['c8', 'lc', 'ssig'] + STAT + SN4,                             # augment: + side-net group stats
    'REP': ['c8', 'lc', 'ssig_sn'] + STAT_NOLG + ['mean_sn', 'max_sn', 'min_sn'],   # replace lg by the side-net score
    'BLD': ['c8', 'lc', 'ssb'] + STAT_NOLG + ['mean_b', 'max_b', 'min_b'],          # lg + sn blend in place of lg
    'P:AUG': ['c8', 'lc', 'ssig'] + STAT + SN4,                           # ridge toward the current theta (sn coefs toward 0)
}


def load_set(S, sc, cfg):
    gid = {}
    import gzip as gz
    rows_t = [json.loads(l) for l in gz.open(os.path.expanduser(os.environ.get('WSM_ROOT', '.') + f'/gsm_stack/implants/sidenet/tables/attempts_{S}.jsonl.gz'), 'rt')]
    v = sc[f'{cfg}|{S}']; assert len(v) == len(rows_t)
    for r, x in zip(rows_t, v): gid[(r['seed'], r['idx'], r['k'])] = float(x)
    D = {}
    for sd, rows in RF.load(ROWS[S]).items():
        out = []
        for r in rows:
            sn = [gid.get((sd, r['idx'], k), -30.0) for k in range(len(r['lg']))]
            out.append(St(r, sn))
        D[sd] = out
    return D


def pick(s, names, th, c0):
    if not len(s.rep): return -1
    sc = s.G(names) @ th + c0
    return int(s.rep[int(np.argmax(sc - 1e-9 * s.rep))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sc', required=True)
    ap.add_argument('--cfgs', default='')
    ap.add_argument('--modes', default='B18,AUG,REP,BLD,P:AUG')
    ap.add_argument('--l2', default='1e-3,1e-2')
    ap.add_argument('--save', default='')                     # CFG@MODE@L2 -> json weights
    a = ap.parse_args()
    import torch
    W = torch.load(os.path.expanduser(os.environ.get('WSM_ROOT', '.') + '/gsm_stack/implants/ladr3/aw/stk/vf_w.pt'))
    th0 = np.array(W['th'], np.float64); c00 = float(W['c0'])
    sc = dict(np.load(a.sc))
    cfgs = a.cfgs.split(',') if a.cfgs else sorted({k.split('|')[0] for k in sc})
    for cfg in cfgs:
        D = {S: load_set(S, sc, cfg) for S in ('trainC', 'dev', 'dev2')}
        allt = [s for v in D['trainC'].values() for s in v]
        for S in ('dev', 'dev2'):
            mm = sum(pick(s, MODES['B18'], th0, c00) != s.pick for v in D[S].values() for s in v)
            assert mm == 0, (S, mm)
        print(f'## {cfg}', flush=True)
        for mode in a.modes.split(','):
            names = MODES[mode]
            pri = np.concatenate([th0, np.zeros(len(names) - 18)]) if mode.startswith('P:') else None
            for l2 in map(float, a.l2.split(',')):
                cv = []
                for k in range(5):
                    th, c0 = fit([s for s in allt if s.idx % 5 != k], names, l2, pri)
                    cv += [(acc_row(s, pick(s, names, th, c0)), acc_row(s, s.pick)) for s in allt if s.idx % 5 == k]
                cv = np.array(cv); fx = int((cv[:, 0] & ~cv[:, 1]).sum()); br = int((~cv[:, 0] & cv[:, 1]).sum())
                th, c0 = fit(allt, names, l2, pri)
                line = f'{mode:6s} l2={l2:g} trainC CV {100 * cv[:, 0].mean():.2f} ({100 * (cv[:, 0].mean() - cv[:, 1].mean()):+.2f}, {fx}/{br})'
                tot = [0, 0]
                for S in ('dev', 'dev2'):
                    accs, fxs, brs, per = [], 0, 0, []
                    for sd, v in sorted(D[S].items()):
                        new = np.array([acc_row(s, pick(s, names, th, c0)) for s in v]); old = np.array([acc_row(s, s.pick) for s in v])
                        accs.append(100 * new.mean()); fxs += int((new & ~old).sum()); brs += int((~new & old).sum())
                        per.append(f'{100 * new.mean():.2f}')
                    old4 = np.mean([100 * np.mean([acc_row(s, s.pick) for s in v]) for v in D[S].values()])
                    line += f' | {S} {np.mean(accs):.2f} ({np.mean(accs) - old4:+.2f}, {fxs}/{brs}; {"/".join(per)})'
                    tot[0] += fxs; tot[1] += brs
                line += f' | pooled {tot[0]}/{tot[1]} p={signp(*tot):.3f}'
                print(line, flush=True)
                if a.save == f'{cfg}@{mode}@{l2:g}':
                    out = dict(th=th.tolist(), c0=float(c0), feats=names, l2=l2, cfg=cfg, mode=mode)
                    json.dump(out, open(os.path.join(H, f'vfsn_{mode.replace(":", "")}_{l2:g}.json'), 'w'), indent=1)
                    print('saved', out['feats'])


if __name__ == '__main__':
    main()
