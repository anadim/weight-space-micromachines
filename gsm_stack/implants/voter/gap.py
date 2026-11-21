"""Structure of the selection gap: in streams where the stack vote is wrong but a right attempt exists,
how big is the right cluster vs the winning cluster; within-stream AUROC of single features on mixed streams."""
import os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vote as V

for s in sys.argv[1:] or ['dev', 'dev2']:
    by, items = V.load_set(s)
    st = V.build(V.seed_streams(by), V.BASE + V.CTX)
    C = collections.Counter(); n = 0
    for x in st:
        w = np.array([V.sig(a['rlg']) for a in x.S]); p = x.pick(w)
        if p is None or x.ok[p] or not x.ok.any(): continue
        n += 1
        nr = int((x.ok * x.keep).sum()); nr_all = int(x.ok.sum())
        nw = int((x.same[p] & x.keep).sum())
        C[(nr, nw)] += 1
    print(s, 'gap-a streams', n, ' (right kept, winning cluster size): count')
    for k, v in sorted(C.items()): print('   ', k, v)
    # within-stream pairwise AUROC: right vs wrong attempt in the same mixed stream
    names = V.BASE + V.CTX
    pos = collections.defaultdict(list)
    for x in st:
        if len(x.S) == 0 or x.ok.all() or not x.ok.any(): continue
        R = np.where(x.ok > 0)[0]; W = np.where(x.ok == 0)[0]
        for j, nm in enumerate(names):
            a = x.F[R, j][:, None]; b = x.F[W, j][None, :]
            pos[nm].append(((a > b).mean() + 0.5 * (a == b).mean()))
    print('  within-stream AUROC (mean over mixed streams):', ', '.join(f'{k}={np.mean(v):.3f}' for k, v in pos.items()))
