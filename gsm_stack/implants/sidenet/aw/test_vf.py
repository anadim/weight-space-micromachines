"""CPU fuzz for the ladnext answer-group voter port (AW_VF=1). Run on the node: STK_DIR=stk python3 test_vf.py (torch + numpy).
[1] numreg automaton: 8192 random class/digit streams vs the regex reference (numreg.ref_numbers = feats.text_numbers rule).
[2] vf_qc: random body / question sets vs a python set computation.
[3] vf_vote (aw_vf.py, the compiled pick) vs the offline ClusterLin (voter/learn.py, AST-extracted verbatim into vf_learn_ref.py)
    on random registers (symbol strings, ok/bad/lg/nn/qc/nt, n used slots, san on/off)."""
import ast, os, random, sys, types
from fractions import Fraction
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get('STK_DIR', os.path.join(H, 'stk')))
import numpy as np, torch
import numreg as NM, aw_vf as AVF

_src = open(os.path.join(H, 'vf_learn_ref.py')).read(); _ns = {'np': np}
_keep = [n for n in ast.parse(_src).body if isinstance(n, ast.ClassDef) and n.name == 'ClusterLin']
assert len(_keep) == 1
exec(compile(ast.Module(body=_keep, type_ignores=[]), 'vf_learn_ref.py', 'exec'), _ns)
ClusterLin = _ns['ClusterLin']
W = torch.load(os.path.join(os.environ.get('STK_DIR', os.path.join(H, 'stk')), 'vf_w.pt'))
M = ClusterLin(); M.th = np.array(W['th'], dtype=np.float64); M.c0 = float(W['c0'])
PAD, KS = 12, 16


def t_numreg(trials=8192, B=256, T=40, seed=1):
    rng = random.Random(seed); bad = 0
    for b0 in range(0, trials, B):
        C = torch.zeros(B, T, dtype=torch.long); D = torch.zeros(B, T, dtype=torch.float64); L = torch.zeros(B, dtype=torch.long)
        for i in range(B):
            L[i] = rng.randint(0, T)
            for t in range(T):
                c = rng.choice([3, 3, 3, 3, 4, 5, 0, 0]); C[i, t] = c; D[i, t] = rng.randint(0, 9) if c == 3 else 0
        nr = NM.NumReg(B, 32, 'cpu'); nr.run(torch.arange(B), C, D, L)
        for i in range(B):
            l = int(L[i]); ref = NM.ref_numbers(C[i, :l].tolist(), [int(x) for x in D[i, :l].tolist()])
            got = {float(x) for x in nr.S[i, :int(nr.n[i])].tolist()}
            ref_f = {float(x) for x in ref}
            bad += (got != ref_f) if len(ref_f) <= 32 else int(nr.n[i]) != 32
    print(f'[1] numreg fuzz {trials} streams: mismatches {bad}', 'PASS' if bad == 0 else 'FAIL', flush=True)
    return bad == 0


def t_qc(trials=4000, seed=2):
    rng = random.Random(seed); bad = 0; B = trials
    POOL = [1.0, 2.0, 3.0, 5.0, 50.0, 0.5, 12.0, 7.0, 0.25, 100.0, 9.0, 1e6]
    Sq = torch.zeros(B, 16, dtype=torch.float64); nq = torch.zeros(B, dtype=torch.long)
    Sb = torch.zeros(B, 32, dtype=torch.float64); nb = torch.zeros(B, dtype=torch.long); ref = []
    for i in range(B):
        q = rng.sample(POOL, rng.randint(0, 6)); b = rng.sample(POOL, rng.randint(0, 10))
        Sq[i, :len(q)] = torch.tensor(q, dtype=torch.float64); nq[i] = len(q)
        Sb[i, :len(b)] = torch.tensor(b, dtype=torch.float64); nb[i] = len(b)
        Sb[i, len(b):] = rng.choice(POOL)         # stale entries beyond n must not count
        ref.append(sum(x in b for x in q) / len(q) if q else 1.0)
    got = AVF.vf_qc(Sq, nq, Sb, nb)
    bad = sum(abs(float(got[i]) - ref[i]) > 1e-12 for i in range(B))
    print(f'[2] vf_qc fuzz {trials}: mismatches {bad}', 'PASS' if bad == 0 else 'FAIL', flush=True)
    return bad == 0


def enc(s):
    v = ['0123456789-.'.index(c) for c in s][:KS]
    return v + [PAD] * (KS - len(v))


def t_vote(trials=20000, seed=3):
    rng = random.Random(seed); bad = diff = skip = 0
    POOL = ['0', '1', '2', '3', '5', '12', '100', '37.5', '-4', '1000000', '7', '0.5']
    for _ in range(trials):
        NS = rng.choice([8, 16]); pool = rng.sample(POOL, rng.randint(1, 4))
        sy = [rng.choice(pool) for _ in range(NS)]
        ok = [rng.random() < 0.85 for _ in range(NS)]; bd = [rng.random() < 0.2 for _ in range(NS)]
        lg = [float(np.float32(rng.choice([-30.0, rng.gauss(0, 3), rng.gauss(2, 1)]))) for _ in range(NS)]
        nn = [float(rng.randint(0, 20)) for _ in range(NS)]; qc = [rng.choice([1.0, 0.5, 0.75, 0.0, 2 / 3]) for _ in range(NS)]
        nt = [float(rng.choice([rng.randint(20, 400), 120])) for _ in range(NS)]
        n = rng.randint(1, NS); san = rng.random() < 0.8
        # reference: offline ClusterLin over the n used slots
        st = types.SimpleNamespace(S=list(range(n)))
        okn = np.array(ok[:n]); sane = okn & ~np.array(bd[:n])
        st.keep = sane if (san and sane.any()) else okn
        st.same = np.array([[sy[i] == sy[j] for j in range(n)] for i in range(n)])
        ntn = np.array(nt[:n]); zt = (ntn - ntn.mean()) / (ntn.std() + 1e-3) if n > 1 else np.zeros(n)
        st.F = np.stack([np.maximum(np.array(lg[:n]), -12), sane.astype(float), np.array(nn[:n]) / 10, zt, np.array(qc[:n])], 1)
        if st.keep.any():
            sc = M.score(st); sc = np.where(st.keep, sc - 1e-9 * np.arange(n), -1e18)
            srt = np.sort(sc[st.keep])[::-1]
            gv = {sy[i] for i in range(n) if st.keep[i]}
            if len(gv) > 1:     # two different answer groups within 1e-7: float order of the sums may differ (skip)
                gs = sorted({round(float(sc[i]), 12) for i in range(n) if st.keep[i]}, reverse=True)
                if len(gs) > 1 and gs[0] - gs[1] < 1e-7: skip += 1; continue
            pv = int(np.argmax(sc))
        else: pv = -1
        T_ = lambda x, dt: torch.tensor([x], dtype=dt)
        has, widx = AVF.vf_vote(T_(ok, torch.bool), T_(bd, torch.bool), T_(lg, torch.float32), torch.tensor([[enc(s) for s in sy]]),
                                T_(nn, torch.float64), T_(qc, torch.float64), T_(nt, torch.float64), torch.tensor([n]),
                                torch.tensor(san), torch.tensor(W['th'], dtype=torch.float64), float(W['c0']))
        p = int(widx[0]) if bool(has[0]) else -1
        bad += p != pv
        # base (sigmoid-lg) pick for the diagnostic count of rows where the voters differ
        if pv >= 0:
            bs = {}
            for i in range(n):
                if st.keep[i]: bs[sy[i]] = bs.get(sy[i], 0) + 1 / (1 + np.exp(-lg[i]))
            b0 = min((i for i in range(n) if st.keep[i] and bs[sy[i]] == max(bs.values())))
            diff += sy[b0] != sy[pv]
    print(f'[3] vf_vote fuzz {trials}: pick mismatches {bad} (near-tie skipped {skip}); rows where vf group != base group {diff}',
          'PASS' if bad == 0 else 'FAIL', flush=True)
    return bad == 0


if __name__ == '__main__':
    r = [t_numreg(), t_qc(), t_vote()]
    print('VF FUZZ', 'PASS' if all(r) else 'FAIL', flush=True)
    sys.exit(0 if all(r) else 1)
