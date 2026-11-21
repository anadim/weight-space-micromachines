"""Within-item AUROC of the coverage features by atlas item category; and coverage in selection-gap streams (right vs voted wrong)."""
import sys, os, gzip, json, collections, numpy as np
sys.argv = [sys.argv[0]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'offline.py')).read().split('\ndef main')[0]
exec(src)
LOG = open(f'{HERE}/bycat.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n')
FEATS = ['pr_qcov_w', 'pr_nmiss_w', 'pr_nnum', 'pr_z_ntok', 'pr_fk77', 'lg']
for S in ('dev', 'dev2'):
    cat = {}
    for l in gzip.open(f'{H}/atlas/data/diverge_{S}.jsonl.gz'):
        d = json.loads(l)
        if d['rec'] == 'item': cat[int(d['idx'])] = d['dominant_cat']
    _, st, _ = make(S, 0)
    F = np.concatenate([x.F for x in st if len(x.S)]); y = np.concatenate([x.ok for x in st if len(x.S)])
    idx = np.concatenate([[x.idx] * len(x.S) for x in st if len(x.S)])
    c = np.array([cat.get(int(i), 'none') for i in idx])
    P(f'## {S}: within-item AUROC by item dominant category (n items)')
    for k in ('relation', 'dropped', 'operand', 'target', 'unit', 'plan'):
        m = c == k
        P(f'{k:10s} n={len(set(idx[m])):3d} ' + ' '.join(f'{f.replace("pr_", "")}={within_auroc(idx[m], y[m], F[m, NAMES.index(f)]):.3f}' for f in FEATS))
    # selection-gap streams
    P(f'## {S}: gap streams (base vote wrong, some kept attempt right): right cluster vs voted wrong cluster (cluster means)')
    ref = [x for x in st if len(x.S)]
    cnt = collections.Counter(); n = 0; nright = []
    for x in ref:
        w = sig(x.F[:, 0]); p = x.pick(w)
        if p is None or x.ok[p] or not (x.ok * x.keep).any(): continue
        n += 1
        R = x.keep & (x.ok > 0); W = x.keep & x.same[p]
        nright.append(int(R.sum()))
        for f in FEATS:
            j = NAMES.index(f); a, b = x.F[R, j].mean(), x.F[W, j].mean()
            cnt[(f, 'right>' if a > b + 1e-9 else ('right<' if a < b - 1e-9 else '='))] += 1
    P('n gap streams', n, 'right-count dist', dict(collections.Counter(nright)))
    for f in FEATS:
        P(f'  {f:12s} right>wrong {cnt[(f, "right>")]:3d}  equal {cnt[(f, "=")]:3d}  right<wrong {cnt[(f, "right<")]:3d}')
