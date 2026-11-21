"""optional: step-error detector line scores (implants/linefix/gate_full_out/scores_*.npz) as voter features.
No trainC scores -> ClusterLin fit by item-grouped 5-fold CV on dev (out-of-fold picks), and fit on all dev -> test dev2.
Same protocol with / without the detector feature (the protocol, not the trainC model, is the reference)."""
import sys, os, numpy as np
sys.argv = [sys.argv[0]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'offline.py')).read().split('\ndef main')[0]
src = src.replace("NAMES = ['lg',", "NAMES = ['pr_dmax', 'pr_dcnt', 'lg',")   # prepend (lg col index shifts; ClusterLin lgcol set below)
exec(src)
from learn import ClusterLin, pick_cluster
_att = attach
def attach(by, S):
    _att(by, S)
    z = np.load(f'{H}/implants/linefix/gate_full_out/scores_{S}.npz'); g = z['gids']
    mx = np.full(len(g), -5.0); ct = np.zeros(len(g))
    np.maximum.at(mx, z['att'], z['score']); np.add.at(ct, z['att'], (z['score'] > 1.821).astype(float))
    d = {k: (m, c) for k, m, c in zip(g, mx, ct)}
    for L in by.values():
        for a in L: a['pr_dmax'], a['pr_dcnt'] = d.get(a['gid'], (-5.0, 0.0))
D = {S: make(S, 0) for S in ('dev', 'dev2')}
LG = NAMES.index('lg')
ref = {S: accv(D[S][1], lambda x: sig(x.F[:, LG])) for S in D}
# AUROC
for S in D:
    by = D[S][0]; A = [a for L in by.values() for a in L if a['committed'] and a['valid']]
    y = np.array([a['ok'] for a in A], float); idx = np.array([a['idx'] for a in A])
    for f in ('pr_dmax', 'pr_dcnt'):
        s = np.array([a[f] for a in A]); print(f'{S} {f}: AUROC pooled {auroc(y, s):.3f} within-item {within_auroc(idx, y, s):.3f}')
def run(ns):
    c = [NAMES.index(n) for n in ns]; out = ''
    st = D['dev'][1]; items = np.array([x.idx if hasattr(x, 'idx') else i for i, x in enumerate(st)])
    rng = np.random.RandomState(0); u = np.unique(items); fold = dict(zip(u, rng.randint(0, 5, len(u))))
    a = np.zeros(len(st))
    for k in range(5):
        tr = [proj(x, c) for x, it in zip(st, items) if fold[it] != k]
        m = ClusterLin(1e-3, lgcol=0).fit(tr)
        for i, (x, it) in enumerate(zip(st, items)):
            if fold[it] == k: a[i] = x.correct(pick_cluster(m, proj(x, c)))
    b = ref['dev']; out += f'dev CV {100*a.mean():.2f} d {100*(a-b).mean():+.2f} f{int(((a==1)&(b==0)).sum())} b{int(((a==0)&(b==1)).sum())}'
    m = ClusterLin(1e-3, lgcol=0).fit([proj(x, c) for x in st])
    a = np.array([x.correct(pick_cluster(m, proj(x, c))) for x in D['dev2'][1]], float); b = ref['dev2']
    out += f' | dev2 (dev-fit) {100*a.mean():.2f} d {100*(a-b).mean():+.2f} f{int(((a==1)&(b==0)).sum())} b{int(((a==0)&(b==1)).sum())}'
    print(f'{"+".join(ns):55s} {out}', flush=True)
for ns in (['lg', 'sane'], ['lg', 'sane', 'pr_dmax'], ['lg', 'sane', 'pr_dcnt'],
           ['lg', 'sane', 'pr_nnum', 'pr_z_ntok', 'pr_qcov_d'], ['lg', 'sane', 'pr_nnum', 'pr_z_ntok', 'pr_qcov_d', 'pr_dmax'],
           ['lg', 'sane', 'pr_nnum', 'pr_z_ntok', 'pr_qcov_d', 'pr_dcnt']):
    run(ns)
