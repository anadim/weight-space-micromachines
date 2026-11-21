"""Offline study of cheap voter features (lane voter-feat).  Train on trainC only; evaluate dev / dev2.

python3.11 implants/vfeat/offline.py  -> offline.log / offline.json
Base = the stack voter sigmoid(lg) (exact port in voter/vote.py).  Learned voters: listwise exp-weight (voter/learn.py) fit on
trainC streams (2 seed streams + 8 bootstrap 8-subsets per item), and 1-parameter families chosen by trainC vote acc.
"""
import os, sys, json, pickle, collections
import numpy as np
H = os.path.expanduser('gsm_stack/implants'); HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f'{H}/voter')
import vote as V
from learn import Listwise

VF = ['qcov_d', 'qcov_w', 'nmiss_d', 'nmiss_w', 'nq_w', 'nnum', 'fk30', 'fk50', 'fk77', 'fk90', 'ans_last_eq', 'tfk3', 'tfk5']
ZF = ['ntok', 'fk50', 'fk77', 'nmiss_w', 'lg', 'nnum']          # within-stream z-scored versions (pr_z_*)
NAMES = ['lg', 'sane', 'ntok', 'nsteps', 'lp_n1', 'lp_mean', 'qcov', 'ans_in_body', 'calc_bad'] + ['pr_' + f for f in VF] + ['pr_z_' + f for f in ZF]
SEEDS2 = (7, 11)


def attach(by, S):
    vf = pickle.load(open(f'{HERE}/cache/vf_{S}.pkl', 'rb'))
    for L in by.values():
        for a in L:
            f = vf[a['gid']]
            for k in VF: a['pr_' + k] = float(f.get(k, np.nan))


def zfill(x):
    """within-stream z of selected features over the stream's committed attempts (vote-time register op)"""
    if not len(x.S): return
    for f in ZF:
        src = x.F[:, NAMES.index(f if f in ('ntok', 'lg') else 'pr_' + f)]
        x.F[:, NAMES.index('pr_z_' + f)] = (src - src.mean()) / (src.std() + 1e-3) if len(src) > 1 else 0.0


def make(S, boot, seed=0):
    by, items = V.load_set(S); attach(by, S)
    ss = V.seed_streams(by)
    st = V.build(ss, NAMES)
    for x, (idx, sd, _) in zip(st, ss):
        if items and not any(a['voted'] for a in by[idx] if a['seed'] == sd):
            x.fixed = items[idx]['vote_ok'][str(sd)]
        x.seed = sd
    bt = V.build(V.random_subsets(by, boot, seed=seed), NAMES) if boot else []
    for x in st + bt: zfill(x)
    return by, st, bt


def auroc(y, s):
    ok = ~np.isnan(s); y, s = y[ok], s[ok]
    if y.sum() == 0 or y.sum() == len(y): return np.nan
    o = np.argsort(s, kind='mergesort'); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r); r = (sums / cnt)[inv]
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def within_auroc(idx, y, s):
    """pairs (right, wrong) inside the same item"""
    num = den = 0.0
    for i in np.unique(idx):
        m = (idx == i) & ~np.isnan(s)
        a, b = s[m & (y == 1)], s[m & (y == 0)]
        if len(a) and len(b):
            d = a[:, None] - b[None, :]
            num += (d > 0).sum() + 0.5 * (d == 0).sum(); den += d.size
    return num / den if den else np.nan


def proj(x, c):
    y = object.__new__(V.Stream); y.__dict__ = dict(x.__dict__); y.F = x.F[:, c] if len(x.S) else np.zeros((0, len(c))); return y


sig = lambda z: 1 / (1 + np.exp(-z))


def accv(streams, wfn):
    return np.array([x.correct(x.pick(wfn(x) if len(x.S) else np.zeros(0))) for x in streams], float)


def main():
    LOG = open(f'{HERE}/offline.log', 'w')

    def P(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n')
    D = {}
    for S in ('trainC', 'dev', 'dev2'):
        _, st, bt = make(S, 8 if S == 'trainC' else 0, seed=1); D[S] = (st, bt)
    out = dict(auroc={}, voters={})
    # ---------------- AUROC: committed slots of seed streams, label = slot answer correct
    P('## AUROC (committed attempts; pooled / within-item over right-wrong pairs of the same item; >0.5 = higher feature -> right)')
    cols = ['lg', 'ntok', 'nsteps', 'lp_n1', 'qcov', 'ans_in_body'] + ['pr_' + f for f in VF] + ['pr_z_' + f for f in ZF]
    P(f'{"feature":16s} ' + ' '.join(f'{S:>15s}' for S in D))
    for c in cols:
        j = NAMES.index(c); row = {}
        for S, (st, _) in D.items():
            F = np.concatenate([x.F[:, j] for x in st if len(x.S)]); y = np.concatenate([x.ok for x in st if len(x.S)])
            idx = np.concatenate([[x.idx] * len(x.S) for x in st if len(x.S)])
            row[S] = (auroc(y, F), within_auroc(idx, y, F))
        out['auroc'][c] = row
        P(f'{c:16s} ' + ' '.join(f'  {row[S][0]:.3f} / {row[S][1]:.3f}' for S in D))
    # mixed-attempt subsets with true forks: correlation of proxy vs true close-fork rate
    for S in ('dev', 'dev2'):
        st = D[S][0]
        F = np.concatenate([x.F for x in st if len(x.S)])
        t = F[:, NAMES.index('pr_tfk3')]; m = ~np.isnan(t)
        P(S, 'corr(true tfk3, proxy):', {f: round(float(np.corrcoef(t[m], F[m, NAMES.index('pr_' + f)])[0, 1]), 3) for f in ('fk30', 'fk50', 'fk77', 'fk90')},
          'mean tfk3 right/wrong', round(float(t[m][np.concatenate([x.ok for x in st if len(x.S)])[m] == 1].mean()), 2),
          round(float(t[m][np.concatenate([x.ok for x in st if len(x.S)])[m] == 0].mean()), 2))
    # ---------------- voters
    TR = D['trainC'][0] + D['trainC'][1]
    ref = {S: accv(D[S][0], lambda x: sig(x.F[:, 0])) for S in D}
    P('\n## vote acc (seed streams). base = stack sigmoid(lg).  cols: acc4 = mean of 4 seeds, acc2 = seeds 7,11, d = delta vs base, f/b fixes/breaks (4 seeds)')
    P(f'{"voter":58s} | {"trainC":>6s} | {"dev acc4":>8s} {"d4":>6s} {"acc2":>6s} {"d2":>6s} {"f":>3s} {"b":>3s} | {"dev2 acc4":>9s} {"d4":>6s} {"acc2":>6s} {"d2":>6s} {"f":>3s} {"b":>3s}')

    def report(name, wfn, extra=None):
        r = {}
        for S in D:
            a = accv(D[S][0], wfn); b = ref[S]; sd = np.array([x.seed for x in D[S][0]]); m2 = np.isin(sd, SEEDS2)
            r[S] = dict(acc4=100 * a.mean(), d4=100 * (a - b).mean(), acc2=100 * a[m2].mean() if m2.any() else np.nan,
                        d2=100 * (a - b)[m2].mean() if m2.any() else np.nan,
                        fix=int(((a == 1) & (b == 0)).sum()), brk=int(((a == 0) & (b == 1)).sum()),
                        fix2=int(((a == 1) & (b == 0) & m2).sum()), brk2=int(((a == 0) & (b == 1) & m2).sum()))
        if extra: r['w'] = extra
        out['voters'][name] = r
        P(f'{name:58s} | {r["trainC"]["acc4"]:6.2f} | ' + ' | '.join(
            f'{r[S]["acc4"]:8.2f} {r[S]["d4"]:+6.2f} {r[S]["acc2"]:6.2f} {r[S]["d2"]:+6.2f} {r[S]["fix"]:3d} {r[S]["brk"]:3d}' for S in ('dev', 'dev2')))
        return r

    report('base: sigmoid(lg)', lambda x: sig(x.F[:, 0]))
    report('count', lambda x: np.ones(len(x.S)))

    def lw(ns, l2=1e-3):
        c = [NAMES.index(n) for n in ns]; m = Listwise(l2, kind='exp').fit([proj(x, c) for x in TR])
        return report('listwise exp ' + '+'.join(n.replace('pr_', '') for n in ns), lambda x: m.w(proj(x, c)),
                      dict(th=dict(zip(ns, m.th.round(4).tolist())), b0=float(m.b0)))
    base = ['lg', 'sane']
    lw(base)
    for f in ['qcov', 'pr_qcov_w', 'pr_nmiss_d', 'pr_nmiss_w', 'pr_fk30', 'pr_fk50', 'pr_fk77', 'pr_fk90', 'pr_ans_last_eq',
              'ans_in_body', 'pr_nnum', 'ntok', 'pr_z_ntok', 'pr_z_fk50', 'pr_z_fk77', 'pr_z_nmiss_w', 'pr_z_lg', 'lp_n1']:
        lw(base + [f])
    lw(base + ['pr_z_nnum']); lw(base + ['pr_nnum', 'pr_z_nnum']); lw(base + ['pr_nnum', 'pr_z_ntok', 'pr_qcov_w'])
    lw(base + ['pr_nnum', 'pr_z_nnum', 'pr_z_ntok', 'ntok', 'pr_qcov_w', 'pr_nmiss_w'])
    lw(base + ['pr_nmiss_w', 'pr_qcov_w'])
    lw(base + ['pr_qcov_w', 'pr_fk77'])
    lw(base + ['pr_qcov_w', 'pr_fk77', 'pr_z_ntok'])
    lw(base + ['pr_qcov_w', 'pr_nmiss_w', 'pr_fk50', 'pr_fk77', 'pr_z_ntok', 'pr_z_fk77', 'pr_ans_last_eq'])
    lw(base + ['pr_qcov_w', 'pr_nmiss_w', 'pr_fk50', 'pr_fk77', 'pr_z_ntok', 'pr_z_fk77', 'pr_ans_last_eq', 'ntok', 'nsteps', 'lp_n1', 'ans_in_body', 'calc_bad'])
    # 1-parameter families, chosen on trainC
    P('\n## 1-param families w = sigmoid(lg) * exp(-a * feat)  (a chosen by trainC acc; all rows shown)')
    for f in ('pr_nmiss_w', 'pr_nmiss_d', 'pr_fk77', 'pr_fk50', 'pr_z_ntok', 'pr_nnum', 'pr_z_nnum'):
        j = NAMES.index(f)
        for a in ((0.1, 0.25, 0.5, 1.0, 2.0) if f in ('pr_nmiss_w', 'pr_nmiss_d', 'pr_fk77', 'pr_fk50') else (-0.1, -0.25, -0.5, -1.0)):
            report(f'sig(lg)*exp(-{a}*{f.replace("pr_", "")})', lambda x, j=j, a=a: sig(x.F[:, 0]) * np.exp(-a * x.F[:, j]))
    json.dump(out, open(f'{HERE}/offline.json', 'w'), indent=1, default=float)


main()
