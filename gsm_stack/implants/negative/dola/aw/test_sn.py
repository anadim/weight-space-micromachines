"""lane sidenet fuzz (local, CPU): aw_sn.vf_vote_sn (the compiled BLD writer) vs the offline re-pick code that produced the
evaluated numbers (repick_sn.St + pick, mode BLD), on random registers (symbols, ok/bad/lg/sn/nn/qc/nt, n used slots).
python3.11 test_sn.py [SN_W.pt]"""
import os, random, sys
import numpy as np, torch
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, 'stk')); sys.path.insert(0, os.path.join(H, '..'))
import aw_sn as ASN
import repick_sn as RS

PAD, KS = 12, 16
P = torch.load(sys.argv[1] if len(sys.argv) > 1 else os.path.join(H, 'stk', 'sn_w.pt'))
th_off = np.array(P['th'], np.float64); c0 = float(P['c0']); names = P['feats']
th_vf = torch.tensor(P['th_vf'], dtype=torch.float64)


def enc(s):
    v = ['0123456789-.'.index(c) for c in s][:KS]
    return v + [PAD] * (KS - len(v))


def main(trials=20000, seed=5):
    rng = random.Random(seed); bad = skip = 0
    POOL = ['0', '1', '2', '3', '5', '12', '100', '37.5', '-4', '1000000', '7', '0.5']
    for t in range(trials):
        NS = rng.choice([8, 16, 24]); pool = rng.sample(POOL, rng.randint(1, 4))
        sy = [rng.choice(pool) for _ in range(NS)]
        ok = [rng.random() < 0.85 for _ in range(NS)]; bd = [rng.random() < 0.2 for _ in range(NS)]
        lg = [float(np.float32(rng.choice([-30.0, rng.gauss(0, 3), rng.gauss(2, 1)]))) for _ in range(NS)]
        sn = [float(rng.choice([-30.0, rng.gauss(-1, 2), rng.gauss(1, 1)])) for _ in range(NS)]
        nn = [float(rng.randint(0, 20)) for _ in range(NS)]; qc = [rng.choice([1.0, 0.5, 0.75, 0.0, 2 / 3]) for _ in range(NS)]
        nt = [float(rng.choice([rng.randint(20, 400), 120])) for _ in range(NS)]
        n = rng.randint(1, NS)
        r = dict(natt=n, idx=t, ok=0, pick=-1, pick0=-1, gold=1.0, rok=ok, bad=bd, val=[0.0] * NS, lg=lg, nn=nn, qc=qc, nt=nt,
                 sym=[enc(s) for s in sy])
        st = RS.St(r, sn)
        ref = RS.pick(st, names, th_off, c0)
        if len(st.rep) > 1:       # skip near ties between groups (float order)
            sc = st.G(names) @ th_off + c0
            o = np.sort(sc)[::-1]
            if o[0] - o[1] < 1e-7: skip += 1; continue
        T = lambda x, dt: torch.tensor([x], dtype=dt)
        has, w = ASN.vf_vote_sn(T(ok, torch.bool), T(bd, torch.bool), T(lg, torch.float32), T(sn, torch.float64),
                                torch.tensor([[enc(s) for s in sy]]), T(nn, torch.float64), T(qc, torch.float64), T(nt, torch.float64),
                                torch.tensor([n]), torch.tensor(True), th_vf, c0)
        got = int(w[0]) if bool(has[0]) else -1
        bad += got != ref
    print(f'vf_vote_sn fuzz {trials}: mismatches {bad} (near ties skipped {skip})', 'PASS' if bad == 0 else 'FAIL', flush=True)
    return bad == 0


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
