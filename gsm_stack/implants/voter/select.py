"""Greedy forward selection of feature groups for the listwise exp-weight voter, using ONLY trainC
(question-grouped 5-fold CV on trainC streams: 2 seed streams + bootstrap 8-subsets of the 16 attempts per item).
python3.11 voter/select.py [--kind exp|sig|cluster] [--probe hash,val_mean]"""
import os, sys, random, argparse, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vote as V
from learn import Listwise, ClusterLin, pick_cluster

ap = argparse.ArgumentParser(); ap.add_argument('--kind', default='exp'); ap.add_argument('--probe', default='')
ap.add_argument('--tboot', type=int, default=8); ap.add_argument('--steps', type=int, default=7)
A = ap.parse_args()
PK = [k for k in A.probe.split(',') if k]
GROUPS = dict(lg=['lg', 'sane'], logp=['lp_mean', 'lp_min', 'lp_n1', 'lp_ans'], calc=['calc_bad'], qcov=['qcov'],
              repair=['nrep'], length=['ntok', 'nsteps'], answer=['ans_isq', 'ans_int', 'ans_log', 'ans_in_body'],
              agree=['agr_i', 'agr_e', 'agr_i_d'], early=['a_in_o', 'o_in_a'])
if PK: GROUPS['probe'] = ['pr_' + k for k in PK]
ALL = [n for g in GROUPS.values() for n in g]

by, items = V.load_set('trainC')
if PK:
    import gzip
    sc = {json.loads(l)['gid']: json.loads(l) for l in gzip.open(os.path.join(V.WS, 'probe', 'out', 'scores_trainC.jsonl.gz'))}
    for L in by.values():
        for a in L:
            for k in PK: a['pr_' + k] = (sc.get(a['gid'], {}).get(k) or 0.0)
st = V.build(V.seed_streams(by), ALL) + V.build(V.random_subsets(by, A.tboot, seed=1), ALL)
its = sorted(by); rng = random.Random(0); rng.shuffle(its); fold = {it: i % 5 for i, it in enumerate(its)}


def proj(x, c):
    y = object.__new__(V.Stream); y.__dict__ = dict(x.__dict__); y.F = x.F[:, c] if len(x.S) else np.zeros((0, len(c))); return y


def cv(names):
    c = [ALL.index(n) for n in names]; P = [proj(x, c) for x in st]; acc = []
    for f in range(5):
        m = (ClusterLin(1e-3) if A.kind == 'cluster' else Listwise(1e-3, kind=A.kind)).fit([x for x in P if fold[x.idx] != f])
        for x in P:
            if fold[x.idx] == f:
                acc.append(x.correct(pick_cluster(m, x) if A.kind == 'cluster' else x.pick(m.w(x))))
    return 100 * np.mean(acc)


base = 100 * np.mean([x.correct(x.pick(np.array([V.sig(a['rlg']) for a in x.S]))) for x in st])
print(f'trainC CV ({A.kind}); stack probe voter on the same streams: {base:.2f}', flush=True)
chosen = ['lg']; cur = cv(GROUPS['lg']); print(f'  +lg -> {cur:.2f}', flush=True)
for step in range(A.steps):
    res = {g: cv([n for h in chosen + [g] for n in GROUPS[h]]) for g in GROUPS if g not in chosen}
    g, v = max(res.items(), key=lambda t: t[1])
    print('  step', step, ' '.join(f'{k}:{x:.2f}' for k, x in sorted(res.items(), key=lambda t: -t[1])), flush=True)
    if v <= cur + 0.02: break
    chosen.append(g); cur = v; print(f'  +{g} -> {cur:.2f}', flush=True)
print('chosen', chosen)
