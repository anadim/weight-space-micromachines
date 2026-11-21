"""ClusterLin (answer-cluster conditional logit, voter/learn.py) with the cheap features; trainC fit."""
import sys, os, numpy as np
sys.argv = [sys.argv[0]]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'offline.py')).read().split('\ndef main')[0])
from learn import ClusterLin, pick_cluster
D = {S: make(S, 8 if S == 'trainC' else 0, seed=1) for S in ('trainC', 'dev', 'dev2')}
TR = D['trainC'][1] + D['trainC'][2]
ref = {S: accv(D[S][1], lambda x: sig(x.F[:, 0])) for S in D}
LOG = open(f'{HERE}/cluster.log', 'w')
for ns in (['lg', 'sane', 'pr_nnum', 'pr_z_ntok', 'pr_qcov_d'], ['lg', 'sane', 'pr_nnum', 'pr_z_ntok', 'ntok', 'pr_qcov_w'], ['lg', 'sane'], ['lg', 'sane', 'pr_nnum', 'pr_z_ntok'], ['lg', 'sane', 'pr_nnum', 'pr_z_ntok', 'pr_qcov_w'],
           ['lg', 'sane', 'pr_nnum', 'pr_z_ntok', 'pr_qcov_w', 'pr_fk77', 'pr_nmiss_w']):
    c = [NAMES.index(n) for n in ns]; m = ClusterLin(1e-3).fit([proj(x, c) for x in TR])
    s = f'cluster {"+".join(ns):60s}'
    for S in D:
        a = np.array([x.correct(pick_cluster(m, proj(x, c))) for x in D[S][1]], float); b = ref[S]
        sd = np.array([x.seed for x in D[S][1]]); m2 = np.isin(sd, SEEDS2)
        s += f' | {S} {100*a.mean():6.2f} d4 {100*(a-b).mean():+5.2f} d2 {100*(a-b)[m2].mean() if m2.any() else 0:+5.2f} f{int(((a==1)&(b==0)).sum())} b{int(((a==0)&(b==1)).sum())}'
    print(s, flush=True); LOG.write(s + '\n')
