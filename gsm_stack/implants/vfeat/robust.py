"""Robustness of the gate-passing ClusterLin voter: trainC bootstrap seed / l2 refits, paired item-bootstrap CI."""
import sys, os, numpy as np
sys.argv = [sys.argv[0]]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'offline.py')).read().split('\ndef main')[0])
from learn import ClusterLin, pick_cluster
LOG = open(f'{HERE}/robust.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n')
EV = {S: make(S, 0)[1] for S in ('dev', 'dev2')}
ref = {S: accv(EV[S], lambda x: sig(x.F[:, 0])) for S in EV}
NS = ['lg', 'sane', 'pr_nnum', 'pr_z_ntok', 'pr_qcov_w']; c = [NAMES.index(n) for n in NS]
for bseed in (1, 2, 3):
    _, st, bt = make('trainC', 8, seed=bseed)
    for l2 in (1e-3, 1e-2):
        m = ClusterLin(l2).fit([proj(x, c) for x in st + bt])
        s = f'boot_seed {bseed} l2 {l2:g}'
        for S in EV:
            a = np.array([x.correct(pick_cluster(m, proj(x, c))) for x in EV[S]], float); b = ref[S]
            idx = np.array([x.idx for x in EV[S]]); its = np.unique(idx)
            d_it = np.array([(a - b)[idx == i].mean() for i in its])
            rng = np.random.default_rng(0); bs = [d_it[rng.integers(0, len(its), len(its))].mean() for _ in range(2000)]
            s += f' | {S} d4 {100*(a-b).mean():+.2f} CI[{100*np.quantile(bs,.025):+.2f},{100*np.quantile(bs,.975):+.2f}] f{int(((a==1)&(b==0)).sum())} b{int(((a==0)&(b==1)).sum())}'
        P(s)
        if bseed == 1 and l2 == 1e-3:
            P('  theta (raw units, order = [c/8, log c, sum sig(lg), mean f, max f, min f] x f in', NS, '):')
            P('  ', np.round(m.th, 3).tolist(), 'c0', round(float(m.c0), 3))
