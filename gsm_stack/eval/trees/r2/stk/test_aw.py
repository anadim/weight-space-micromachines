"""CPU fuzz for aw_adapt.py (A+B+C+stop4 on the comp_serial symbol register) vs the adapt lane's python references
(ref_vote / ref_ext / same_val / sig taken verbatim from implants/adapt/test_adapt.py by AST, so they are the same code).
Run on the node: STK_DIR=aw/stk python3 test_aw.py   (needs torch only)."""
import ast, math, os, random, sys, types
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get('STK_DIR', os.path.join(H, 'stk')))
import torch
import aw_adapt as AWA

TA = os.environ.get('TEST_ADAPT', os.path.join(H, 'test_adapt_refs.py'))
_src = open(TA).read(); _ns = {'math': math}
_keep = [n for n in ast.parse(_src).body if isinstance(n, ast.FunctionDef) and n.name in ('same_val', 'sig', 'ref_vote', 'ref_ext')]
assert len(_keep) == 4
exec(compile(ast.Module(body=_keep, type_ignores=[]), TA, 'exec'), _ns)
same_val, sig, ref_vote, ref_ext = (_ns[k] for k in ('same_val', 'sig', 'ref_vote', 'ref_ext'))

PAD = 12; KS = 16
POOL = ['0', '1', '2', '3', '5', '12', '100', '37.5', '-4', '1000000', '7', '0.5']


def enc(s):
    v = ['0123456789-.'.index(c) for c in s][:KS]
    return v + [PAD] * (KS - len(v))


def rnd_reg(rng, NS):
    pool = rng.sample(POOL, 4)
    sy = [rng.choice(pool) for _ in range(NS)]
    ok = [rng.random() < 0.85 for _ in range(NS)]
    bad = [rng.random() < 0.2 for _ in range(NS)]
    lg = [rng.choice([-30.0, rng.gauss(0, 3), rng.gauss(2, 1), 0.85]) for _ in range(NS)]
    lg = [float(torch.tensor(x, dtype=torch.float32)) for x in lg]
    return sy, ok, bad, lg


def near_tie(sy, ok, bad, lg, n, san, vm):
    """two different values with (near-)equal scores: the circuit's 1e-6 tie band vs the reference's 1e-12 (skip)"""
    S = range(n); okk = [ok[i] for i in S]; sane = [okk[i] and not bad[i] for i in S]
    keep = sane if (san and any(sane)) else okk
    w = [sig(lg[i]) if vm == 1 else 1.0 for i in S]
    sc = {}
    for i in S:
        if keep[i]: sc[sy[i]] = sum(w[j] for j in S if keep[j] and sy[j] == sy[i])
    v = sorted(sc.values(), reverse=True)
    return len(v) > 1 and v[0] - v[1] < 1e-5


def test_vote(trials=20000, seed=0):
    rng = random.Random(seed); bp = bs = be = sk = 0
    for _ in range(trials):
        NS = rng.choice([8, 16, 24]); sy, ok, bad, lg = rnd_reg(rng, NS); n = rng.randint(1, NS)
        san = rng.random() < 0.8; vm = rng.choice([1, 1, 0])
        if near_tie(sy, ok, bad, lg, n, san, vm): sk += 1; continue
        T = lambda x, dt: torch.tensor([x], dtype=dt)
        has, widx, share, eq = AWA.vote_share(T(ok, torch.bool), T(bad, torch.bool), T(lg, torch.float32),
                                              torch.tensor([[enc(s) for s in sy]]), torch.tensor([n]), torch.tensor(san), torch.tensor(vm))
        rp, rs = ref_vote([float(s) for s in sy], ok, bad, lg, n, san, vm)
        p = int(widx[0]) if bool(has[0]) else -1
        bp += p != rp; bs += abs(float(share[0]) - rs) > 1e-5
        nmax = rng.choice([16, 24]); tau = rng.choice([0.5, 0.7, 0.9]); ext = rng.random() < 0.9; nn = rng.randint(1, nmax + 2)
        if abs(rs - tau) < 1e-5: continue
        d = AWA.ext_more(torch.tensor([nn]), share, torch.tensor(ext), torch.tensor(8), torch.tensor(nmax), torch.tensor(4),
                         torch.tensor(tau, dtype=torch.float32))
        be += bool(d[0]) != ref_ext(nn, rs, ext, 8, nmax, 4, tau)
    print(f'vote fuzz: {trials}: pick mismatches {bp}, share mismatches {bs}, extend mismatches {be} (near-tie skipped {sk})')
    return bp + bs + be == 0


def ref_decide(r, N, R, K, ext, nmax, blk, tau, stop, n0, th, san, vm, acap, maxnew):
    n = r['a'] + 1
    pick, share = ref_vote([float(s) for s in r['sy']], r['ok'], r['bad'], r['lg'], n, san, vm)
    room = r['glen'] + acap + 56 <= maxnew
    S = range(n0)
    st = stop and n == n0 and all(r['ok'][j] and not r['bad'][j] for j in S) and all(r['sy'][j] == r['sy'][0] for j in S) \
        and min(sig(r['lg'][j]) for j in S) >= th
    more = (n < N or ref_ext(n, share, ext, N, nmax, blk, tau)) and room and not st
    rlim = max(R, K) if (r['valid'] and r['fl'] == 2) else R
    rep = r['end'] and r['rbad'] and r['rep'] < rlim and room
    wait = r['end'] and more and not rep
    fin = r['end'] and not more and not rep
    return rep, wait, fin, (pick if fin else None), share, st


def test_decide(trials=20000, seed=3):
    rng = random.Random(seed); bad_d = sk = fires = {'rep': 0, 'wait': 0, 'fin': 0, 'stop': 0}
    bad_d = 0; sk = 0; fires = dict(rep=0, ext=0, stop=0, fin=0)
    for _ in range(trials):
        NS = rng.choice([16, 24]); sy, ok, bad, lg = rnd_reg(rng, NS)
        if rng.random() < 0.3:                                # unanimous sane early slots (stop4 candidates)
            v = rng.choice(POOL); k = rng.randint(3, 6)
            for j in range(k): sy[j] = v; ok[j] = True; bad[j] = False
        a = rng.choice([0, 1, 2, 3, 3, 4, 6, 7, 7, 8, 9, 11, 12, 15, 16, 19, NS - 1]) % NS
        N = 8; R = rng.choice([0, 1]); K = rng.choice([0, 3]); nmax = rng.choice([16, NS]); blk = 4; tau = rng.choice([0.5, 0.7])
        ext = rng.random() < 0.8; stop = rng.random() < 0.7; n0 = rng.choice([3, 4]); th = rng.choice([0.0, 0.5, 0.7])
        san = rng.random() < 0.9; vm = 1; acap = rng.choice([320, 400]); maxnew = 7936
        r = dict(sy=sy, ok=ok, bad=bad, lg=lg, a=a, glen=rng.choice([100, 3000, 7400, 7500, 7600]), rep=rng.choice([0, 1, 2, 3]),
                 end=rng.random() < 0.9, valid=rng.random() < 0.8, fl=rng.choice([0, 1, 2, 3]), rbad=rng.random() < 0.4)
        if near_tie(sy, ok, bad, lg, a + 1, san, vm): sk += 1; continue
        if any(abs(sig(x) - th) < 1e-6 for x in lg[:n0]): sk += 1; continue
        _, rs = ref_vote([float(s) for s in sy], ok, bad, lg, a + 1, san, vm)
        if abs(rs - tau) < 1e-5: sk += 1; continue
        dev = 'cpu'; t = lambda x, dt=torch.long: torch.tensor(x, dtype=dt)
        eng = types.SimpleNamespace(dev=dev, R_ok=t([ok], torch.bool), R_bad=t([bad], torch.bool), R_lg=t([lg], torch.float32),
                                    R_sym=t([[enc(s) for s in sy]]), s_a=t([a]), s_rep=t([r['rep']]), glen=t([r['glen']]),
                                    maxnew=t(maxnew), k_N=t(N), k_R=t(R), k_san=t(san, torch.bool), k_vm=t(vm))
        spec = ':'.join(x for x in [f'A={tau}/{nmax}/{blk}' if ext else '', f'S={n0}/{th}' if stop else '', f'B={K}', f'C={acap}'] if x)
        AWA.setup(eng, spec)
        if not ext: eng.k_ext.fill_(False)
        rep, wait, fin, has, widx = AWA.decide(eng, 1, t([r['end']], torch.bool), t([r['valid']], torch.bool), t([r['fl']]),
                                                 t([r['rbad']], torch.bool))
        R_ = ref_decide(r, N, R, K, ext, nmax, blk, tau, stop, n0, th, san, vm, acap, maxnew)
        got = (bool(rep[0]), bool(wait[0]), bool(fin[0]), (int(widx[0]) if bool(has[0]) else -1) if bool(fin[0]) else None)
        bad_d += got != R_[:4]
        fires['rep'] += R_[0]; fires['fin'] += R_[2]; fires['stop'] += R_[5] and r['end']
        fires['ext'] += R_[1] and a + 1 >= N
    print(f'decide fuzz: {trials}: (rep, wait, fin, pick) mismatches {bad_d} (skipped near ties {sk}); ref fires {fires}')
    return bad_d == 0


if __name__ == '__main__':
    ok = test_vote() & test_decide()
    print('AW FUZZ', 'PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
