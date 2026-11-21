"""lane vfrefit: offline refit of the rung-3 final writer (answer-group ClusterLin) on trainC rows of the rung-3 stack.
python3.11 refit.py [--cands ...] : fit on aw/out/tc_trainC (4 seeds), item-grouped 5-fold CV on trainC, then full-trainC fit ->
re-pick on the dev/dev2 avf rows (implants/ladr3/aw/out/avf_*: same streams => pure re-pick), paired fixes/breaks vs the current
VF pick (row 'pick', = vf_w.pt). Accuracy of a row = correctness of the picked slot's register value (= the written '#### X');
rows without a kept slot keep the row's own ok (identical across writers).
Per slot (as aw_vf.vf_vote): keep = ok & (sane | no sane slot); Fv = [max(lg,-12), sane, nn/10, z_nt (within row, used slots), qc].
Per answer group (all-symbol-equal kept slots) feature library (all register-computable, see FEATS)."""
import argparse, glob, gzip, json, os, re, sys
import numpy as np
from math import comb
from scipy.optimize import minimize

H = os.path.dirname(os.path.abspath(__file__))
AW = os.path.join(H, '..', 'aw')
REF = os.path.expanduser(os.environ.get('WSM_ROOT', '.') + '/gsm_stack/implants/ladr3/aw/out')
sig = lambda x: 1 / (1 + np.exp(-x))


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, '*.jsonl.gz'))):
        sd = int(re.search(r'_seed(\d+)', f).group(1))
        out[sd] = [json.loads(l) for l in gzip.open(f, 'rt')]
    return out


class St:
    """one row (stream): per used slot registers + per-group feature dict"""
    def __init__(self, r, sub=None):
        if sub is not None:                                  # augmentation: sub-stream of the committed slots (order kept)
            r = dict(r); r['natt'] = len(sub); r['pick'] = r['pick0'] = -2
            for k in ('rok', 'bad', 'val', 'lg', 'nn', 'qc', 'nt', 'sym'): r[k] = [r[k][i] for i in sub]
        n = r['natt']; self.idx = r['idx']; self.n = n; self.rowok = r['ok']; self.pick = r['pick']; self.pick0 = r['pick0']
        g = float(r['gold'])
        ok = np.array(r['rok'][:n], bool); bad = np.array(r['bad'][:n], bool); sane = ok & ~bad
        self.keep = ok & (sane | ~sane.any())
        val = np.array(r['val'][:n]); self.cor = ok & (np.abs(val - g) <= 1e-6 * max(1.0, abs(g)))
        lg = np.array(r['lg'][:n], np.float64); nn = np.array(r['nn'][:n]); qc = np.array(r['qc'][:n]); nt = np.array(r['nt'][:n])
        sy = [tuple(x) for x in r['sym'][:n]]
        zt = (nt - nt.mean()) / (nt.std() + 1e-3) if n > 1 else np.zeros(n)
        zn = (nn - nn.mean()) / (nn.std() + 1e-3) if n > 1 else np.zeros(n)
        self.Fv = np.stack([np.maximum(lg, -12), sane.astype(float), nn / 10, zt, qc], 1)
        keys = []; rep = []
        for i in range(n):
            if self.keep[i] and sy[i] not in keys: keys.append(sy[i]); rep.append(i)
        self.rep = np.array(rep, int); K = len(rep)
        if K == 0: return
        M = np.array([[self.keep[j] and sy[j] == k for j in range(n)] for k in keys], float)     # (K, n)
        self.y = self.cor[self.rep]
        cnt = M.sum(1); nk = self.keep.sum(); s = sig(self.Fv[:, 0])
        ssig = M @ s; Fv = self.Fv
        big = np.where(M[:, :, None] > 0, Fv[None], -1e9).max(1); small = np.where(M[:, :, None] > 0, Fv[None], 1e9).min(1)
        sw = ssig / max((s * self.keep).sum(), 1e-9)
        isb = np.zeros(K); isb[int(np.argmax(ssig - 1e-9 * np.arange(K)))] = 1     # base (pick0) group: max sum sigmoid(lg), earliest
        zq = (qc - qc.mean()) / (qc.std() + 1e-3) if n > 1 else np.zeros(n)
        f = dict(c8=cnt / 8, lc=np.log(cnt), ssig=ssig, sh=cnt / n, shk=cnt / nk, lsh=np.log(cnt / n), pw=sw, isb=isb,
                 ismax=(cnt == cnt.max()).astype(float), n8sh=(n / 8) * cnt / n,
                 sumlg=M @ Fv[:, 0], zn_mean=(M @ zn) / cnt, zn_max=np.where(M > 0, zn[None], -1e9).max(1),
                 zq_mean=(M @ zq) / cnt,
                 c8e=cnt / n, lce=np.log(8 * cnt / n), ssige=ssig * 8 / n,                 # 8-slot-equivalent count (rescale only)
                 c8m=cnt / max(n, 8), lcm=np.log(8 * cnt / max(n, 8)), ssigm=ssig * 8 / max(n, 8))   # rescale only when n > 8
        for j, nm in enumerate(('lg', 'sane', 'nn', 'zt', 'qc')):
            f['mean_' + nm] = (M @ Fv[:, j]) / cnt; f['max_' + nm] = big[:, j]; f['min_' + nm] = small[:, j]
        self.f = f

    def G(self, names):
        return np.stack([self.f[k] for k in names], 1)


V5 = ('lg', 'sane', 'nn', 'zt', 'qc')
STAT = [f'{a}_{b}' for a in ('mean', 'max', 'min') for b in V5]
BASE18 = ['c8', 'lc', 'ssig'] + STAT                    # = aw_vf.vf_vote order (cnt/8, log cnt, sum sig, mean5, max5, min5)
CANDS = {
    'B18': BASE18,                                       # same features, refit
    'SH18': ['sh', 'lc', 'ssig'] + STAT,                 # count/8 -> share of used slots
    'SHK18': ['shk', 'lc', 'ssig'] + STAT,               # share of kept slots
    'SH_LSH': ['sh', 'lsh', 'ssig'] + STAT,              # share + log share
    'SH_PW': ['sh', 'lc', 'ssig', 'pw'] + STAT,          # + probe-weight share
    'SH_PW_LSH': ['sh', 'lsh', 'pw'] + STAT,             # all normalised: share, log share, probe share
    'SH_PW_ISB': ['sh', 'lc', 'ssig', 'pw', 'isb'] + STAT,
    'C8_SH_PW': ['c8', 'sh', 'lc', 'ssig', 'pw'] + STAT,
    'SH_PW_Z': ['sh', 'lc', 'ssig', 'pw', 'zn_mean', 'zn_max', 'zq_mean'] + STAT,
    'SH_PW_SUMLG': ['sh', 'lc', 'ssig', 'pw', 'sumlg'] + STAT,
    'SH_PW_MEAN': ['sh', 'lsh', 'pw'] + [f'mean_{b}' for b in V5] + [f'max_{b}' for b in ('lg', 'nn', 'zt')],
    'E8': ['c8e', 'lce', 'ssige'] + STAT,                # 8-slot-equivalent count/log count/sum sig (always)
    'M8': ['c8m', 'lcm', 'ssigm'] + STAT,                # same, only when n > 8
    'M8_PW': ['c8m', 'lcm', 'ssigm'] + STAT + ['pw'],
    'B18_PW': BASE18 + ['pw'],
    'B18_SH': BASE18 + ['sh'],
}
# mode per candidate name prefix: 'X0:' = old theta, no fit (zero-parameter rescale); 'P:' = ridge toward the old theta
# (first 18 coefficients prior = vf_w.pt th, extra features prior 0); plain = ridge toward 0 (fresh fit)


def fit(streams, names, l2, prior=None):
    Gs = [s.G(names) for s in streams if len(s.rep) > 1 and s.y.any()]
    Ys = [s.y for s in streams if len(s.rep) > 1 and s.y.any()]
    X = np.concatenate(Gs); mu = X.mean(0); sd = X.std(0) + 1e-6
    Z = (X - mu) / sd; Y = np.concatenate(Ys).astype(float)
    pz = np.zeros(len(names)) if prior is None else prior * sd        # ridge toward the prior theta (standardized space)
    seg = np.repeat(np.arange(len(Gs)), [len(x) for x in Gs]); S = len(Gs)

    def fg(th):
        z = Z @ th
        m = np.full(S, -1e18); np.maximum.at(m, seg, z)
        e = np.exp(z - m[seg]); den = np.bincount(seg, e, S); p = e / den[seg]
        L = -(np.log(np.bincount(seg, p * Y, S) + 1e-300)).mean() + l2 * ((th - pz) ** 2).sum()
        q = p * Y / (np.bincount(seg, p * Y, S)[seg] + 1e-300)          # posterior over gold clusters (single gold => Y)
        return L, Z.T @ (p - q) / S + 2 * l2 * (th - pz)
    r = minimize(fg, pz.copy(), jac=True, method='L-BFGS-B', options=dict(maxiter=3000))
    return r.x / sd, -(mu * r.x / sd).sum()


def pick(s, names, th, c0):
    if not len(s.rep): return -1
    sc = s.G(names) @ th + c0
    return int(s.rep[int(np.argmax(sc - 1e-9 * s.rep))])


def acc_row(s, p):
    return bool(s.cor[p]) if p >= 0 else bool(s.rowok)


def signp(fx, br):
    n = fx + br
    return sum(comb(n, j) for j in range(fx, n + 1)) / 2 ** n if n else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tc', default=os.path.join(AW, 'out', 'tc_trainC'))
    ap.add_argument('--cands', default=','.join(CANDS))
    ap.add_argument('--l2', default='1e-4,1e-3,1e-2')
    ap.add_argument('--save', default='')             # CAND:L2 -> write weights file off/vf_w_<CAND>.pt-ready json
    ap.add_argument('--sets', default='dev,dev2')
    ap.add_argument('--aug', type=int, default=0)       # random sub-streams per trainC row (size U[ceil(n/2), n]) added to the fit set
    a = ap.parse_args()
    import torch
    W = torch.load(os.path.join(AW, 'stk', 'vf_w.pt'))
    th0 = np.array(W['th'], np.float64); c00 = float(W['c0'])
    TC = {sd: [St(r) for r in rows] for sd, rows in load(a.tc).items()}
    EV = {S: {sd: [St(r) for r in rows] for sd, rows in load(os.path.join(REF, f'avf_{S}')).items()} for S in a.sets.split(',')}
    allt = [s for v in TC.values() for s in v]
    aug = []
    if a.aug:
        rng = np.random.default_rng(0)
        for v in load(a.tc).values():
            for r in v:
                n = r['natt']
                if n < 4: continue
                for _ in range(a.aug):
                    m = int(rng.integers((n + 1) // 2, n + 1)); aug.append(St(r, sorted(rng.choice(n, m, replace=False).tolist())))
        print('aug sub-streams', len(aug))
    print(f'trainC seeds {sorted(TC)} rows {len(allt)} informative {sum(len(s.rep) > 1 and s.y.any() for s in allt)}; '
          f'natt mean {np.mean([s.n for s in allt]):.2f}; oracle {100 * np.mean([s.cor.any() or s.rowok for s in allt]):.2f}')
    # sanity: current theta reproduces the in-stream pick on the eval rows
    for S, D in EV.items():
        mm = sum(pick(s, BASE18, th0, c00) != s.pick for v in D.values() for s in v)
        okm = sum(acc_row(s, s.pick) != bool(s.rowok) for v in D.values() for s in v)
        print(f'{S}: current-theta re-pick mismatches vs row pick {mm}; register-acc vs row ok mismatches {okm}')
    a0 = 100 * np.mean([acc_row(s, pick(s, BASE18, th0, c00)) for s in allt])
    p0 = 100 * np.mean([acc_row(s, s.pick0) for s in allt])
    print(f'trainC acc: current VF theta {a0:.2f}  base voter pick0 {p0:.2f}  row ok {100 * np.mean([s.rowok for s in allt]):.2f}')
    fold = lambda s: s.idx % 5
    res = []
    for cn in a.cands.split(','):
        mode, base = (cn.split(':') + [None])[:2] if ':' in cn else ('', cn)
        names = CANDS[base]
        pri = np.concatenate([th0, np.zeros(len(names) - 18)]) if mode in ('P', 'X0') else None
        def fitm(ss, l2):
            if mode == 'X0': return pri, c00
            return fit(ss, names, l2, pri)
        for l2 in (map(float, a.l2.split(',')) if mode != 'X0' else [0.0]):
            cv = []
            for k in range(5):
                th, c0 = fitm([s for s in allt + aug if fold(s) != k], l2)
                cv += [(acc_row(s, pick(s, names, th, c0)), acc_row(s, pick(s, BASE18, th0, c00))) for s in allt if fold(s) == k]
            cv = np.array(cv); fx = int((cv[:, 0] & ~cv[:, 1]).sum()); br = int((~cv[:, 0] & cv[:, 1]).sum())
            th, c0 = fitm(allt + aug, l2)
            line = f'{cn:12s} l2={l2:g}  trainC CV {100 * cv[:, 0].mean():.2f} ({100 * (cv[:, 0].mean() - cv[:, 1].mean()):+.2f}, f/b {fx}/{br})'
            ev = {}
            for S, D in EV.items():
                accs, fxs, brs = [], 0, 0
                for sd, v in sorted(D.items()):
                    new = np.array([acc_row(s, pick(s, names, th, c0)) for s in v]); old = np.array([acc_row(s, s.pick) for s in v])
                    accs.append(100 * new.mean()); fxs += int((new & ~old).sum()); brs += int((~new & old).sum())
                    ev[(S, sd)] = 100 * (new.mean() - old.mean())
                old4 = np.mean([100 * np.mean([acc_row(s, s.pick) for s in v]) for v in D.values()])
                line += f' | {S} {np.mean(accs):.2f} ({np.mean(accs) - old4:+.2f}, {fxs}/{brs})'
            print(line, flush=True)
            res.append((cn, l2, cv[:, 0].mean(), th, c0))
            if a.save and a.save == f'{cn}:{l2:g}':
                out = dict(th=th.tolist(), c0=float(c0), feats=names, l2=l2, src='vfrefit trainC (tc_trainC seeds ' + ','.join(map(str, sorted(TC))) + ')')
                json.dump(out, open(os.path.join(H, f'vf_w_{cn}_{l2:g}.json'), 'w'), indent=1)
                print('saved', out)
    b = max(res, key=lambda x: x[2])
    print(f'best by trainC CV: {b[0]} l2={b[1]:g} CV {100 * b[2]:.2f}')


if __name__ == '__main__':
    main()
