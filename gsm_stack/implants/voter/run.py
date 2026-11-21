"""Voter comparison driver.

python3.11 voter/run.py [--train dev|trainC] [--boot 16]
 --train dev   : question-grouped 5-fold CV on dev (out-of-fold weights), model fit on all dev applied to dev2 (provisional)
 --train trainC: fit on trainC streams (2 seeds x 8 + bootstrap 8-subsets of its 16 attempts), applied to dev and dev2
Metrics: seed = mean over the 4 seed streams of N=8 vote acc; boot = random 8-subsets of the 32 attempts;
fix/brk = seed streams fixed / broken vs the stack's probe voter; gap a = vote wrong with a right attempt in the
stream, b = no right attempt.
"""
import os, sys, json, argparse, collections, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vote as V
from learn import Listwise, Pointwise, ClusterLin, pick_cluster

ap = argparse.ArgumentParser()
ap.add_argument('--train', default='dev'); ap.add_argument('--boot', type=int, default=16)
ap.add_argument('--tboot', type=int, default=8); ap.add_argument('--out', default=None)
ap.add_argument('--ablate', action='store_true'); ap.add_argument('--probe', default=None)
ap.add_argument('--probe_file', default=os.path.join(V.WS if False else os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'probe', 'out', 'scores_{set}.jsonl.gz'))
A = ap.parse_args()

PKEYS = A.probe.split(',') if A.probe else []
ALL = V.BASE + V.CTX + ['pr_' + k for k in PKEYS]


def attach_probe(by, s):
    if not A.probe: return
    import gzip
    p = A.probe_file.format(set=s)
    sc = {}
    for l in gzip.open(p):
        r = json.loads(l); sc[r['gid']] = r
    miss = 0
    for L in by.values():
        for a in L:
            r = sc.get(a['gid'], {}); miss += a['gid'] not in sc
            for k in PKEYS: a['pr_' + k] = r.get(k) if r.get(k) is not None else 0.0
    print(s, 'probe scores attached; missing', miss)


def make(s, boot, seed=0):
    by, items = V.load_set(s)
    attach_probe(by, s)
    st = V.build(V.seed_streams(by), ALL)
    for x, (idx, sd, S) in zip(st, V.seed_streams(by)):
        # nofinal stream (the voter never ran) or no kept slot: graded on the stream's own last '####'
        if items and not any(a['voted'] for a in by[idx] if a['seed'] == sd):
            x.fixed = items[idx]['vote_ok'][str(sd)]
        x.seed = sd
    bt = V.build(V.random_subsets(by, boot, seed=seed), ALL) if boot else []
    return by, items, st, bt


def cols(names): return [ALL.index(n) for n in names]


def evaluate(streams, wfn):
    pf = wfn.__self__.pick if getattr(wfn, '__name__', '') == 'w' and hasattr(wfn.__self__, 'pick') else (lambda x: x.pick(wfn(x)))
    picks = [pf(x) for x in streams]
    return np.array([x.correct(p) for x, p in zip(streams, picks)])


def summary(name, res, ref):
    out = {}
    for s, (seed_c, boot_c, st) in res.items():
        r = ref[s]
        anyok = np.array([x.ok.any() for x in st])
        idx = np.array([x.idx for x in st]); its = np.unique(idx)
        d_it = np.array([(seed_c[idx == i] - r[idx == i]).mean() for i in its])
        rng = np.random.default_rng(0); bs = [d_it[rng.integers(0, len(its), len(its))].mean() for _ in range(2000)]
        out[s] = dict(ci=(100 * np.quantile(bs, 0.025), 100 * np.quantile(bs, 0.975)), seed=100 * seed_c.mean(), boot=100 * boot_c.mean() if len(boot_c) else float('nan'),
                      fix=int(((seed_c == 1) & (r == 0)).sum()), brk=int(((seed_c == 0) & (r == 1)).sum()),
                      gap_a=100 * ((seed_c == 0) & anyok).mean(), gap_b=100 * ((seed_c == 0) & ~anyok).mean())
    return out


class Sub:
    """model restricted to a feature subset"""
    def __init__(self, model, names): self.m = model; self.c = cols(names)

    def fit(self, streams):
        self.m.fit([_proj(x, self.c) for x in streams]); return self

    def w(self, x): return self.m.w(_proj(x, self.c))

    def pick(self, x):
        y = _proj(x, self.c)
        return pick_cluster(self.m, y) if isinstance(self.m, ClusterLin) else y.pick(self.m.w(y))


def _proj(x, c):
    y = object.__new__(V.Stream); y.__dict__ = dict(x.__dict__); y.F = x.F[:, c] if len(x.S) else np.zeros((0, len(c)))
    return y


def main():
    sets = ['dev', 'dev2']
    D = {s: make(s, A.boot) for s in sets}
    ref = {s: evaluate(D[s][2], lambda x: np.array([V.sig(a['rlg']) for a in x.S])) for s in sets}
    rows = collections.OrderedDict()

    def fixed_voter(name, wfn):
        rows[name] = summary(name, {s: (evaluate(D[s][2], wfn), evaluate(D[s][3], wfn), D[s][2]) for s in sets}, ref)

    fixed_voter('count (sane keep)', lambda x: np.ones(len(x.S)))
    fixed_voter('probe sigmoid(lg) [stack]', lambda x: np.array([V.sig(a['rlg']) for a in x.S]))
    fixed_voter('probe * no-calc-error', lambda x: np.array([V.sig(a['rlg']) * (a['calc_bad'] == 0) + 1e-3 for a in x.S]))

    if A.train == 'trainC':
        T = make('trainC', A.tboot, seed=1)
        tr = T[2] + T[3]

    def learned(name, mk, names):
        m = Sub(mk(), names)
        res = {}
        if A.train == 'dev':
            # dev: 5-fold question-grouped CV (out-of-fold); dev2: fit on all dev
            st, bt = D['dev'][2], D['dev'][3]
            items = sorted({x.idx for x in st}); rng = random.Random(0); rng.shuffle(items)
            fold = {it: i % 5 for i, it in enumerate(items)}
            sc = np.zeros(len(st)); bc = np.zeros(len(bt))
            for f in range(5):
                trn = [x for x in st + bt if fold[x.idx] != f]
                mf = Sub(mk(), names).fit(trn)
                for i, x in enumerate(st):
                    if fold[x.idx] == f: sc[i] = x.correct(mf.pick(x))
                for i, x in enumerate(bt):
                    if fold[x.idx] == f: bc[i] = x.correct(mf.pick(x))
            res['dev'] = (sc, bc, st)
            m.fit(st + bt)
            res['dev2'] = (evaluate(D['dev2'][2], m.w), evaluate(D['dev2'][3], m.w), D['dev2'][2])
        else:
            m.fit(tr)
            for s in sets: res[s] = (evaluate(D[s][2], m.w), evaluate(D[s][3], m.w), D[s][2])
        rows[name] = summary(name, res, ref)
        return m

    FULL = [n for n in ALL]
    TEXT = [n for n in ALL if n != 'lg' and not n.startswith('pr_')]
    m_full = learned('listwise: all features', lambda: Listwise(1e-3), FULL)
    learned('listwise exp-weight: all features', lambda: Listwise(1e-3, kind='exp'), FULL)
    learned('cluster-linear: all features', lambda: ClusterLin(1e-3), FULL)
    learned('cluster-linear: lg+agree+calc+qcov', lambda: ClusterLin(1e-3), ['lg', 'sane', 'calc_bad', 'qcov', 'agr_i', 'agr_e'])
    SEL_E = ['lg', 'sane', 'agr_i', 'agr_e', 'agr_i_d', 'ntok', 'nsteps', 'qcov']
    learned('listwise exp: trainC-selected lg+agree+len+qcov', lambda: Listwise(1e-3, kind='exp'), SEL_E)
    learned('cluster-linear: trainC-selected lg+length', lambda: ClusterLin(1e-3), ['lg', 'sane', 'ntok', 'nsteps'])
    if PKEYS:
        PR = ['pr_' + k for k in PKEYS]
        learned('listwise sig: lg+hidden-state scores', lambda: Listwise(1e-3), ['lg', 'sane'] + PR)
        learned('listwise exp: lg+hidden-state scores', lambda: Listwise(1e-3, kind='exp'), ['lg', 'sane'] + PR)
        learned('cluster-linear: lg+hidden-state scores', lambda: ClusterLin(1e-3), ['lg', 'sane'] + PR)
        learned('listwise exp: selected text + hidden', lambda: Listwise(1e-3, kind='exp'), SEL_E + PR)
        learned('cluster-linear: lg+length + hidden', lambda: ClusterLin(1e-3), ['lg', 'sane', 'ntok', 'nsteps'] + PR)
    learned('pointwise LR weight: all features', lambda: Pointwise(1e-3), FULL)
    learned('listwise: lg only', lambda: Listwise(1e-3), ['lg', 'sane'])
    learned('listwise: text/register only (no lg)', lambda: Listwise(1e-3), TEXT)
    learned('listwise: no agreement ctx', lambda: Listwise(1e-3), [n for n in FULL if n not in V.CTX])
    if A.ablate:
        groups = dict(lg=['lg'], logp=['lp_mean', 'lp_min', 'lp_n1', 'lp_ans'], calc=['calc_bad'], qcov=['qcov'],
                      repair=['nrep', 'sane'], length=['ntok', 'nsteps'], answer=['ans_isq', 'ans_int', 'ans_log', 'ans_in_body'],
                      agree=['agr_i', 'agr_e', 'agr_i_d'], early=['a_in_o', 'o_in_a'], k=['k'])
        if A.probe: groups['probe'] = ['pr_' + k for k in PKEYS]
        for g, ns in groups.items():
            learned(f'  ablate -{g}', lambda: Listwise(1e-3), [n for n in FULL if n not in ns])
        for g, ns in groups.items():
            learned(f'  only lg+sane+{g}', lambda: Listwise(1e-3), sorted(set(['lg', 'sane'] + ns), key=FULL.index))
    print(f'train={A.train}  boot={A.boot}/item  features={FULL}')
    print(f'{"voter":42s} | {"dev seed":>8s} {"boot":>6s} {"fix":>4s} {"brk":>4s} {"gapA":>5s} {"gapB":>5s} | '
          f'{"dev2 seed":>9s} {"boot":>6s} {"fix":>4s} {"brk":>4s} {"gapA":>5s} {"gapB":>5s}')
    for n, r in rows.items():
        a, b = r['dev'], r['dev2']
        print(f'{n:42s} | {a["seed"]:8.2f} {a["boot"]:6.2f} {a["fix"]:4d} {a["brk"]:4d} {a["gap_a"]:5.2f} {a["gap_b"]:5.2f} | '
              f'{b["seed"]:9.2f} {b["boot"]:6.2f} {b["fix"]:4d} {b["brk"]:4d} {b["gap_a"]:5.2f} {b["gap_b"]:5.2f}'
              f' | CI dev [{a["ci"][0]:+.2f},{a["ci"][1]:+.2f}] dev2 [{b["ci"][0]:+.2f},{b["ci"][1]:+.2f}]')
    print('full listwise weights (raw feature units):')
    print('  ' + ', '.join(f'{n}={w:+.3f}' for n, w in zip(FULL, m_full.m.th)) + f', b0={m_full.m.b0:+.3f}, beta={m_full.m.beta_:.2f}')
    if A.out: json.dump(dict(rows=rows, w=dict(zip(FULL, m_full.m.th.tolist())), b0=m_full.m.b0, beta=m_full.m.beta_),
                        open(A.out, 'w'), indent=1)


main()
