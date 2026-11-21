"""CPU fuzz tests for the adapt-lane implants (compiled decisions in job/stk/serial_stack.py vs python references).
 1. voter + implant A decision: vote_slots (pick, probe-weighted winner share) and ext_more vs a python reference that
    mirrors voter/vote.py (keep_mask, weighted_pick, same_val, sig) and voter/adaptive.py (share, blocks of 4).
 2. implant B detector: the register reader automaton (SerialEngine._read) on random '#### <number>' token streams vs the
    scorer regex ####\\s*(-?[\\d,]*\\.?\\d+): valid, value, negative, and non-integer (the integer-guard trigger).
Run: WIRING_DIR=... python3 test_adapt.py  (needs torch; imports job/stk/serial_stack.py)."""
import math, os, random, re, sys, types
H = os.path.dirname(os.path.abspath(__file__))
STK = os.environ.get('STK_DIR', os.path.join(H, 'job', 'stk'))
sys.path[:0] = [STK, os.environ.get('WIRING_DIR', os.path.join(H, 'job', 'wiring'))]
import torch
import serial_stack as SS


# ---------------------------------------------------------------- python references
def same_val(x, y): return abs(x - y) <= 1e-6 * max(1.0, abs(y))
def sig(x): return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))


def ref_vote(val, ok, bad, lg, n, san, vmode):
    S = list(range(n))
    okk = [ok[i] for i in S]
    sane = [okk[i] and not bad[i] for i in S]
    keep = sane if (san and any(sane)) else okk
    w = [sig(lg[i]) if vmode == 1 else 1.0 for i in S]
    if not any(keep): return -1, 0.0
    if vmode == 2:
        pick = max(i for i in S if keep[i])
    else:
        best, pick = -1e18, None
        for i in S:
            if not keep[i]: continue
            sc = sum(w[j] for j in S if keep[j] and same_val(val[i], val[j]))
            if sc > best + 1e-12: best, pick = sc, i
    tot = sum(w[j] for j in S if keep[j])
    win = sum(w[j] for j in S if keep[j] and same_val(val[pick], val[j]))
    return pick, win / max(tot, 1e-9)


def ref_ext(n, share, ext, N, nmax, blk, tau):
    if not ext or n < N or n >= nmax: return False
    return n % blk != 0 or share < tau


def test_voter(trials=20000, seed=0):
    rng = random.Random(seed); bad_pick = bad_share = bad_ext = 0; near = 0
    for t in range(trials):
        NM = rng.choice([8, 16, 24]); B = 1
        n = rng.randint(1, NM)
        pool = [rng.choice([0, 1, 2, 3, 5, 12, 100, 37.5, -4, 1e6]) for _ in range(4)]
        val = [rng.choice(pool) * (1 + rng.choice([0, 0, 0, 5e-7, 3e-6])) for _ in range(NM)]
        ok = [rng.random() < 0.85 for _ in range(NM)]
        bad = [rng.random() < 0.2 for _ in range(NM)]
        lg = [rng.choice([-30.0, rng.gauss(0, 3), rng.gauss(2, 1)]) for _ in range(NM)]
        san = rng.random() < 0.8; vmode = rng.choice([1, 1, 0, 2])
        T = lambda x, dt: torch.tensor([x], dtype=dt)
        pick, share = SS.vote_slots(T(val, torch.float64), T(ok, torch.bool), T(bad, torch.bool), T(lg, torch.float32),
                                    torch.tensor([n]), torch.tensor(san), torch.tensor(vmode))
        lg32 = [float(torch.tensor(x, dtype=torch.float32)) for x in lg]
        rp, rs = ref_vote(val, ok, bad, lg32, n, san, vmode)
        if int(pick[0]) != rp: bad_pick += 1
        if vmode != 2 and abs(float(share[0]) - rs) > 1e-9: bad_share += 1
        # A decision on the same register
        N = 8; nmax = rng.choice([16, 24]); blk = 4; tau = rng.choice([0.5, 0.7, 0.9]); ext = rng.random() < 0.9
        nn = rng.randint(1, nmax + 2)
        sh = float(share[0]) if vmode != 2 else rs
        if abs(sh - tau) < 1e-9: near += 1; continue
        d = SS.ext_more(torch.tensor([nn]), share if vmode != 2 else torch.tensor([rs], dtype=torch.float64), torch.tensor(ext),
                        torch.tensor(N), torch.tensor(nmax), torch.tensor(blk), torch.tensor(tau, dtype=torch.float64))
        if bool(d[0]) != ref_ext(nn, sh, ext, N, nmax, blk, tau): bad_ext += 1
    print(f'voter fuzz: {trials} registers: pick mismatches {bad_pick}, share mismatches {bad_share}, '
          f'extend-decision mismatches {bad_ext} (skipped |share-tau|<1e-9: {near})')
    return bad_pick + bad_share + bad_ext == 0


def test_stop(trials=20000, seed=2):
    """both-ways budget: stop_early (n == n0, n0 sane unanimous slots, all sigmoid(lg) >= th) vs python reference."""
    rng = random.Random(seed); bad_d = fired = near = 0
    for t in range(trials):
        NM = rng.choice([16, 24]); n0 = rng.choice([3, 4, 4, 5]); n = rng.choice([n0, n0, n0 - 1, n0 + 1, 8])
        v0 = rng.choice([1, 12, 37.5, -4, 1e6])
        val = [v0 if rng.random() < 0.9 else rng.choice([2, 13, 0]) for _ in range(NM)]
        ok = [rng.random() < 0.95 for _ in range(NM)]; bad = [rng.random() < 0.05 for _ in range(NM)]
        lg = [rng.choice([-30.0, rng.gauss(1, 1.5), rng.gauss(3, 1)]) for _ in range(NM)]
        th = rng.choice([0.0, 0.5, 0.7, 0.9]); stop = rng.random() < 0.9
        lg32 = [float(torch.tensor(x, dtype=torch.float32)) for x in lg]
        if any(abs(sig(x) - th) < 1e-6 for x in lg32[:n0]): near += 1; continue
        T = lambda x, dt: torch.tensor([x], dtype=dt)
        Vv, okR, badR, lgR = T(val, torch.float64), T(ok, torch.bool), T(bad, torch.bool), T(lg, torch.float32)
        _, share = SS.vote_slots(Vv, okR, badR, lgR, torch.tensor([n]), torch.tensor(True), torch.tensor(1))
        slg = math.log(th / (1 - th)) if th > 0 else -1e30
        d = bool(SS.stop_early(okR, badR, lgR, torch.tensor([n]), share, torch.tensor(stop), torch.tensor(n0),
                               torch.tensor(slg, dtype=torch.float32))[0])
        S = list(range(n0))
        ref = stop and n == n0 and all(ok[j] and not bad[j] for j in S) and all(same_val(val[j], val[0]) for j in S) \
            and min(sig(lg32[j]) for j in S) >= th
        bad_d += d != ref; fired += ref
    print(f'stop fuzz: {trials} registers ({fired} stop): decision mismatches {bad_d} (skipped |sig-th|<1e-6: {near})')
    return bad_d == 0


# ---------------------------------------------------------------- reader / integer guard
VOC = ['####', '##', '###', '#', ' ', '  ', '\n', '-', ' -', ',', '.', 'x', ' dollars', '$', ' the', '.\n'] + list('0123456789')
NUMRE = re.compile(r'####\s*(-?[\d,]*\.?\d+)')


def rand_stream(rng):
    t = [rng.choice(['x', ' the', ' ', '$']) for _ in range(rng.randint(0, 3))]
    t += rng.choice([['####'], ['##', '##'], ['###', '#'], ['#', '###']])
    t += [rng.choice([' ', '  ', '\n']) for _ in range(rng.choice([0, 1, 1, 1, 2]))]
    if rng.random() < 0.15: t.append(rng.choice(['-', ' -']))
    if rng.random() < 0.05: t.append(rng.choice(['$', 'x', ' ']))
    for _ in range(rng.choice([0, 1, 1, 2, 3, 4, 6])):
        t.append(rng.choice(list('0123456789') + [','] * (rng.random() < 0.1)))
    if rng.random() < 0.45:
        t.append('.')
        fr = rng.choice([[], ['0'], ['5'], ['0', '0'], ['5', '0'], ['2', '5'], ['0', '1'], ['3', '3', '3', '3']]
                        + [[rng.choice('0123456789') for _ in range(rng.randint(1, 5))]])
        t += fr
        if rng.random() < 0.05: t += ['.', rng.choice('0123456789')]
    t.append(rng.choice([' dollars', '\n', '.\n', ' ', 'x', '.']))
    return t


def ref_read(s):
    m = NUMRE.search(s)
    if not m: return None
    x = m.group(1).replace(',', '')
    ip, _, fp = x.partition('.')
    return dict(val=float(x), neg=x.startswith('-'), nonint=any(c != '0' for c in fp))


def test_reader(trials=20000, seed=1):
    cls, dig, nh = SS.token_classes(VOC)
    tid = {t: i for i, t in enumerate(VOC)}
    rng = random.Random(seed)
    streams = [rand_stream(rng) for _ in range(trials)]
    B = trials; L = max(map(len, streams))
    st = types.SimpleNamespace()
    z = lambda dt=torch.long: torch.zeros(B, dtype=dt)
    st.r_cls = torch.tensor(cls); st.r_dig = torch.tensor(dig, dtype=torch.float64); st.r_nh = torch.tensor(nh)
    st.ph, st.hc, st.an, st.qp, st.ql = z(), z(), z(), z(), z()
    st.done = z(torch.bool); st.cl = z(torch.bool)
    st.val, st.fsc = z(torch.float64), z(torch.float64)
    st.neg, st.fnz, st.hasd = z(torch.bool), z(torch.bool), z(torch.bool)
    st.atok = torch.zeros(B, SS.ATOK, dtype=torch.long)
    r = torch.arange(B)
    eos = tid['x']
    for k in range(L + 1):
        tok = torch.tensor([tid[s[k]] if k < len(s) else eos for s in streams])
        SS.SerialEngine._read(st, B, tok, r)
    valid = (st.ph >= 3) & (st.ph <= 6) & st.hasd
    nonint = valid & st.fnz & ~st.neg          # integer-guard trigger (rlim in _act)
    mm = {'valid': 0, 'val': 0, 'neg': 0, 'nonint': 0}; n_ni = 0; ex = []
    for j, s in enumerate(streams):
        ref = ref_read(''.join(s)); v = bool(valid[j])
        if (ref is not None) != v: mm['valid'] += 1; ex.append(''.join(s)); continue
        if ref is None: continue
        sv = -float(st.val[j]) if bool(st.neg[j]) else float(st.val[j])
        if abs(sv - ref['val']) > 1e-9 * max(1, abs(ref['val'])): mm['val'] += 1; ex.append(''.join(s))
        if bool(st.neg[j]) != ref['neg']: mm['neg'] += 1
        rni = ref['nonint'] and not ref['neg']
        n_ni += rni
        if bool(nonint[j]) != rni: mm['nonint'] += 1; ex.append(''.join(s))
    print(f'reader fuzz: {trials} streams ({sum(ref_read("".join(s)) is not None for s in streams)} with a number, '
          f'{n_ni} non-integer): mismatches {mm}', ('examples ' + repr(ex[:5])) if ex else '')
    return sum(mm.values()) == 0


if __name__ == '__main__':
    a = test_voter(); b = test_reader(); c = test_stop()
    print('ALL PASS' if a and b and c else 'FAIL')
    sys.exit(0 if a and b and c else 1)
