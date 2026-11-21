"""fit the in-stream ClusterLin weights (trainC only) -> job/stk/vf_w.pt"""
import sys, os, numpy as np, torch
sys.argv = [sys.argv[0]]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'offline.py')).read().split('\ndef main')[0])
from learn import ClusterLin, pick_cluster
NS = ['lg', 'sane', 'pr_nnum', 'pr_z_ntok', 'pr_qcov_d']
D = {S: make(S, 8 if S == 'trainC' else 0, seed=1) for S in ('trainC', 'dev', 'dev2')}
c = [NAMES.index(n) for n in NS]; m = ClusterLin(1e-3).fit([proj(x, c) for x in D['trainC'][1] + D['trainC'][2]])
for S in D:
    a = np.array([x.correct(pick_cluster(m, proj(x, c))) for x in D[S][1]], float); b = accv(D[S][1], lambda x: sig(x.F[:, 0]))
    print(S, round(100 * a.mean(), 2), round(100 * (a - b).mean(), 2))
torch.save(dict(th=[float(v) for v in m.th], c0=float(m.c0), feats=NS), f'{HERE}/job/stk/vf_w.pt')
print(list(np.round(m.th, 3)), m.c0)
