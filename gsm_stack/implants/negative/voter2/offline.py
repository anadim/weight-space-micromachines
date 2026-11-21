"""Offline choice of the v2 verifier + voter (trainC-fit only; dev/dev2 held out).

python3.11 implants/voter2/offline.py [--write_w]
Probe scores: probe/out/pred_trainC/hash_{set}.npz (trainC: question-grouped 5-fold OOF; dev/dev2: full-trainC fit).
Voter: listwise exp-weight (voter/learn.py) fit on trainC streams (2 seed streams + 8 bootstrap 8-subsets per item).
--write_w writes job/stk/voter2_w.pt (probe weights from pred_trainC/probes.pt + the chosen voter weights).
"""
import os, sys, gzip, json, argparse, random
import numpy as np
H = os.path.expanduser('gsm_stack/implants'); HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f'{H}/voter')
import vote as V
from learn import Listwise

ap = argparse.ArgumentParser(); ap.add_argument('--write_w', action='store_true'); ap.add_argument('--tag', default='trainC')
A = ap.parse_args()
PD = f'{H}/probe/out/pred_{A.tag}'
COLS = ['hash_mnl_c4', 'c4', 'stack_lg', 'L18', 'mnl_c4']


def gids(S):
    out = []
    for l in gzip.open(f'{H}/probe/tables/attempts_{S}.jsonl.gz', 'rt'):
        i = l.index('"gid": "') + 8; out.append(l[i:l.index('"', i)])
    return out


def attach(by, S):
    z = dict(np.load(f'{PD}/hash_{S}.npz')); cols = list(z['cols']); G = gids(S)
    sc = {G[a]: z['S'][r] for r, a in enumerate(z['att'])}
    for L in by.values():
        for a in L:
            s = sc.get(a['gid'])
            for c in COLS:
                a['pr_' + c] = max(-12.0, float(s[cols.index(c)])) if s is not None else -12.0


def make(S, boot, seed=0, names=None):
    by, items = V.load_set(S); attach(by, S)
    ss = V.seed_streams(by)
    st = V.build(ss, names)
    for x, (idx, sd, _) in zip(st, ss):
        if items and not any(a['voted'] for a in by[idx] if a['seed'] == sd):
            x.fixed = items[idx]['vote_ok'][str(sd)]
        x.seed = sd
    bt = V.build(V.random_subsets(by, boot, seed=seed), names) if boot else []
    return st, bt


NAMES = ['lg', 'sane', 'ntok', 'nsteps', 'k'] + ['pr_' + c for c in COLS]
D = {S: make(S, 0, names=NAMES)[0] for S in ('dev', 'dev2')}
TR = make('trainC', 8, seed=1, names=NAMES); TR = TR[0] + TR[1]
ci = lambda ns: [NAMES.index(n) for n in ns]


def proj(x, c):
    y = object.__new__(V.Stream); y.__dict__ = dict(x.__dict__); y.F = x.F[:, c] if len(x.S) else np.zeros((0, len(c))); return y


def acc(streams, wfn):
    return np.array([x.correct(x.pick(wfn(x))) for x in streams], float)


sig = lambda z: 1 / (1 + np.exp(-z))
ref = {S: acc(D[S], lambda x: sig(x.F[:, 0]) if len(x.S) else np.zeros(0)) for S in D}   # stack voter: sigmoid(lg)
rows = {}


def report(name, wfn):
    r = {}
    for S in D:
        a = acc(D[S], wfn); b = ref[S]; sd = np.array([x.seed for x in D[S]])
        r[S] = dict(acc=100 * a.mean(), d=100 * (a - b).mean(), fix=int(((a == 1) & (b == 0)).sum()), brk=int(((a == 0) & (b == 1)).sum()),
                    per_seed={int(s): (round(100 * b[sd == s].mean(), 2), round(100 * a[sd == s].mean(), 2),
                                       int(((a == 1) & (b == 0) & (sd == s)).sum()), int(((a == 0) & (b == 1) & (sd == s)).sum()))
                              for s in np.unique(sd)})
    rows[name] = r
    print(f'{name:44s} | dev {r["dev"]["acc"]:6.2f} {r["dev"]["d"]:+5.2f} f{r["dev"]["fix"]:3d} b{r["dev"]["brk"]:3d} | '
          f'dev2 {r["dev2"]["acc"]:6.2f} {r["dev2"]["d"]:+5.2f} f{r["dev2"]["fix"]:3d} b{r["dev2"]["brk"]:3d}', flush=True)


def lw(ns, kind='exp', l2=1e-3):
    c = ci(ns); m = Listwise(l2, kind=kind).fit([proj(x, c) for x in TR])
    return m, (lambda x: m.w(proj(x, c)))


# AUROC of the probe columns on dev/dev2 committed attempts (sanity vs predict_fit log)
def auroc(y, s):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


for S in D:
    F = np.concatenate([x.F for x in D[S] if len(x.S)]); y = np.concatenate([x.ok for x in D[S] if len(x.S)])
    print(S, 'AUROC committed slots:', {n: round(auroc(y, F[:, NAMES.index(n)]), 4) for n in ['lg'] + ['pr_' + c for c in COLS]})

print(f'{"voter (trainC-fit)":44s} | dev acc  d(vs stack) fix brk | dev2')
report('stack: sigmoid(lg)', lambda x: sig(x.F[:, 0]) if len(x.S) else np.zeros(0))
report('count', lambda x: np.ones(len(x.S)))
for c in COLS:
    j = NAMES.index('pr_' + c)
    report(f'sigmoid({c})', lambda x, j=j: sig(x.F[:, j]) if len(x.S) else np.zeros(0))
MODELS = {}
for ns in (['lg', 'sane'], ['pr_hash_mnl_c4', 'sane'], ['pr_hash_mnl_c4', 'lg', 'sane'], ['pr_c4', 'lg', 'sane'],
           ['pr_stack_lg', 'lg', 'sane'], ['pr_hash_mnl_c4', 'lg', 'sane', 'ntok', 'nsteps'], ['pr_hash_mnl_c4', 'pr_c4', 'lg', 'sane']):
    m, f = lw(ns); MODELS[tuple(ns)] = m
    report('listwise exp ' + '+'.join(ns), f)
    print('    th=', np.round(m.th, 3), 'b0=', round(m.b0, 3))
m, f = lw(['pr_hash_mnl_c4', 'lg', 'sane'], kind='sig'); report('listwise sig pr_hash_mnl_c4+lg+sane', f)

CH = ('pr_hash_mnl_c4', 'lg', 'sane')
json.dump(dict(rows=rows, chosen=dict(feats=list(CH), th=MODELS[CH].th.tolist(), b0=float(MODELS[CH].b0))),
          open(f'{HERE}/offline_{A.tag}.json', 'w'), indent=1)
if A.write_w:
    import torch
    P = torch.load(f'{PD}/probes.pt', map_location='cpu', weights_only=False)['hash']['hash_mnl_c4']
    w = P['w'].float(); m = MODELS[CH]
    torch.save(dict(wh=w[:4096].clone(), wm=w[4096:].clone(), b=float(P['b']), th=m.th.tolist(), b0=float(m.b0),
                    src=f'probe: {PD}/probes.pt hash/hash_mnl_c4 (trainC full fit); voter: listwise exp on trainC {list(CH)}'),
               f'{HERE}/job/stk/voter2_w.pt')
    print('wrote job/stk/voter2_w.pt  th', m.th, 'b0', m.b0)
