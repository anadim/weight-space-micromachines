"""[18](a) CPU differential fuzz: purist.sw_step (fixed ReLU layers) vs the reference comp_serial.SerialW._sw (+ aw_adapt.decide),
on random register states / tokens / knobs (AW on and off, ties forced), plus the side labels, WPR offset and probe capture.
Every register, the write vector, bits and cap_probe must be bitwise equal.   python3 test_purist.py [trials]"""
import copy, math, os, random, sys, types
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
import torch
from unittest import mock

for m in ('triton', 'triton.language', 'engine', 'fastqwen', 'c2_engine', 'san_engine', 'attn3'):
    sys.modules[m] = mock.MagicMock()


class _C2: pass


sys.modules['c2_engine'].C2Engine = _C2
fw = types.ModuleType('forget_ws'); fw.wrap_stack = lambda c: c; sys.modules['forget_ws'] = fw
os.environ['PURIST'] = '0'
os.environ.setdefault('SW_NS', '16')
import comp_serial as CS
import aw_adapt as AWA
import purist as PU

NS, KS, NSYM, PAD = CS.NS, CS.KS, CS.NSYM, CS.PAD
REGS = ['s_dfa', 's_wr', 's_c', 's_k', 's_lg', 's_lgset', 'R_sym', 'R_ok', 'R_bad', 'R_lg', 's_a', 's_rep', 's_nf', 's_fr',
        's_p', 's_win', 's_fin', 's_nrep', 's_pe', 'done']


class Eng:
    _vote = CS.SerialW._vote


def make_eng(rng, B, V=64, nS=34, D=8):
    e = Eng(); dev = 'cpu'; e.dev = dev
    g = torch.Generator().manual_seed(rng.randrange(1 << 30))
    e.t_TT = torch.randint(0, nS, (V, nS), generator=g)
    e.t_TT[:, 0] = torch.randint(0, 3, (V,), generator=g)           # make the W0 / done states reachable often
    e.t_done = torch.rand(nS, generator=g) < 0.3; e.t_numst = torch.rand(nS, generator=g) < 0.5
    e.t_FLAG = torch.randint(0, 4, (nS,), generator=g); e.W0 = 1; e.nS = nS
    e.t_sym = torch.randint(-1, 12, (V,), generator=g)
    e.t_stop = torch.rand(V, generator=g) < 0.15; e.t_ENL = torch.rand(V, generator=g) < 0.3; e.t_SNL = torch.rand(V, generator=g) < 0.3
    lens = [rng.randint(1, 4) for _ in range(5)]
    rows = [None]; start = []; ends = []
    for L in lens:
        start.append(len(rows)); rows += [1] * L; ends.append(len(rows) - 1)
    e.r_dig0 = len(rows); rows += [None] * KS; e.r_eos = len(rows); rows += [2]; P_ = len(rows)
    SH = torch.zeros(P_, P_); SH[0, 0] = 1
    for i in range(1, P_):
        if i in ends[:3]: SH[i, 0] = 1
        elif i in ends[3:]: SH[i, e.r_dig0] = 1
        elif e.r_dig0 <= i < e.r_eos - 1: SH[i, i + 1] = 1
        elif i == e.r_eos - 1: SH[i, e.r_eos] = 1
        elif i == e.r_eos: SH[i, 0] = 1
        else: SH[i, i + 1] = 1
    e.t_SH = SH; e.b_start = start; e.P_ = P_
    e.t_DROW = torch.randn(P_, D, generator=g); e.t_DROW[0] = 0
    for i, t in enumerate(rows):
        if t is None: e.t_DROW[i] = 0
    e.t_DSYM = torch.randn(NSYM, D, generator=g); e.gain = 3e4
    Bp = B + 1
    e.k_N = torch.tensor(rng.choice([2, 4, 8, NS])); e.k_R = torch.tensor(rng.choice([0, 1, 2]))
    e.k_vm = torch.tensor(rng.choice([0, 1, 1])); e.k_san = torch.tensor(rng.random() < 0.8); e.k_rm = torch.tensor(rng.random() < 0.3)
    e.acap = rng.choice([8, 320, 400]); e.aw_on = False
    e.maxnew = torch.tensor(rng.choice([7936, 900, 600]))
    e.glen = torch.randint(0, 900, (Bp,), generator=g)
    e.slot = torch.randint(0, 2000, (Bp,), generator=g)
    if rng.random() < 0.7:
        A = (rng.choice([0.5, 0.7, 0.9]), rng.choice([n for n in (4, 8, 16, NS) if n <= NS]), rng.choice([1, 2, 4])) if rng.random() < 0.8 else None
        S = (rng.choice([2, 4]), rng.choice([0.5, 0.8, 0.0])) if rng.random() < 0.7 else None
        spec = ':'.join(([f"A={A[0]}/{A[1]}/{A[2]}"] if A else []) + ([f"S={S[0]}/{S[1]}"] if S else []) +
                        ([f"B={rng.choice([2, 3])}"] if rng.random() < 0.6 else []) + ([f"C={e.acap}"]))
        e.R_ok = torch.zeros(Bp, NS, dtype=torch.bool); AWA.setup(e, spec)
    return e


def rand_regs(e, rng, B):
    g = torch.Generator().manual_seed(rng.randrange(1 << 30)); Bp = B + 1
    ri = lambda lo, hi, *s: torch.randint(lo, hi, s, generator=g)
    rb = lambda p, *s: torch.rand(*s, generator=g) < p
    e.s_dfa = ri(0, e.nS, Bp); e.s_wr = rb(0.3, Bp); e.s_c = ri(0, e.acap + 2, Bp); e.s_k = ri(0, KS + 1, Bp)
    lgpool = torch.tensor([-30.0, 0.0, 0.85, 1.5, -2.0, 3.25], dtype=torch.float32)
    e.s_lg = torch.where(rb(0.5, Bp), lgpool[ri(0, 6, Bp)], torch.randn(Bp, generator=g) * 3)
    e.s_lgset = rb(0.6, Bp)
    pool = torch.stack([torch.tensor((list(range(rng.randrange(1, 4))) + [PAD] * KS)[:KS]) for _ in range(3)] +
                       [ri(0, 13, KS) for _ in range(2)])
    e.R_sym = pool[ri(0, len(pool), Bp, NS)]
    e.R_ok = rb(0.8, Bp, NS); e.R_bad = rb(0.25, Bp, NS)
    e.R_lg = torch.where(rb(0.6, Bp, NS), lgpool[ri(0, 6, Bp, NS)], torch.randn(Bp, NS, generator=g) * 3)
    e.s_a = ri(0, NS, Bp); e.s_rep = ri(0, 4, Bp); e.s_nf = ri(0, 2 * NS, Bp)
    e.s_fr = torch.where(rb(0.3, Bp), torch.full((Bp,), -1), ri(0, 2000, Bp))
    e.s_qlen = ri(0, 500, Bp)
    e.s_p = torch.where(rb(0.5, Bp), torch.zeros(Bp, dtype=torch.long), ri(0, e.P_, Bp))
    e.s_win = ri(-1, NS, Bp); e.s_fin = rb(0.1, Bp); e.s_nrep = ri(0, 5, Bp); e.s_pe = rb(0.3, Bp)
    e.done = rb(0.1, Bp)
    if rng.random() < 0.3:                                       # equal scores across slots (tie path)
        e.R_lg[:] = 0.85


def snap(e):
    return {k: getattr(e, k).clone() for k in REGS}


def same(a, b):
    if a.dtype.is_floating_point: return torch.equal(a.view(torch.int32) if a.dtype == torch.float32 else a, b.view(torch.int32) if b.dtype == torch.float32 else b)
    return torch.equal(a, b)


def main(T):
    rng = random.Random(1234); nbad = 0; nev = dict(rep=0, wait=0, fin=0, write=0, cap=0)
    for t in range(T):
        B = rng.choice([1, 5, 33])
        e = make_eng(rng, B); rand_regs(e, rng, B)
        tok = torch.randint(0, e.t_TT.shape[0], (B,), generator=torch.Generator().manual_seed(t))
        e0 = copy.copy(e)
        for k in REGS: setattr(e0, k, getattr(e, k).clone())
        e1 = copy.copy(e)
        for k in REGS: setattr(e1, k, getattr(e, k).clone())
        w0, b0, c0 = CS.SerialW._sw(e0, B, tok)
        PU.setup(e1)
        w1, b1, c1 = PU.sw_step(e1, B, tok, NS, KS, NSYM, PAD)
        errs = [k for k in REGS if not same(getattr(e0, k), getattr(e1, k))]
        if not torch.equal(w0, w1): errs.append('write')
        if not torch.equal(b0, b1): errs.append('bits')
        if not torch.equal(c0, c1.bool()): errs.append('cap_probe')
        # probe capture / side labels / wpr offset on the post-step registers
        vl = torch.randn(B)
        e0b, e1b = copy.copy(e0), copy.copy(e0); e0b.s_lg, e0b.s_lgset = e0.s_lg.clone(), e0.s_lgset.clone()
        e1b.s_lg, e1b.s_lgset = e0.s_lg.clone(), e0.s_lgset.clone()
        e0b.s_lg[:B] = torch.where(c0, vl, e0b.s_lg[:B]); e0b.s_lgset[:B] = e0b.s_lgset[:B] | c0
        PU.probe_capture(e1b, B, c0.double(), vl)
        if not (same(e0b.s_lg, e1b.s_lg) and torch.equal(e0b.s_lgset, e1b.s_lgset)): errs.append('probe_capture')
        e0.w_E = rng.choice([8, 2 * NS + 1]); slot = e0.slot[:B]
        a = e0.s_nf[:B].clamp(max=e0.w_E - 1); lab = (a - (slot < e0.s_fr[:B]).long()).clamp(min=0)
        ohl, oha = PU.side_labels(e0, B, slot)
        if not (torch.equal(ohl, torch.nn.functional.one_hot(lab, e0.w_E).float()) and torch.equal(oha, torch.nn.functional.one_hot(a, e0.w_E).float())):
            errs.append('side_labels')
        off = torch.where(e0.s_fr[:B] > 0, e0.s_fr[:B] - e0.s_qlen[:B], torch.zeros_like(slot))
        if not torch.equal(PU.wpr_offset(e0, B, slot), off.double()): errs.append('wpr_offset')
        e0._wpr_c = torch.randn(2100, 4); e0._wpr_s = torch.randn(2100, 4); PU.setup(e0)
        offc = off.clamp(min=0)
        rc, rs = PU.wpr_rows(e0, offc.double())
        if not (torch.equal(rc, e0._wpr_c[offc][:, None]) and torch.equal(rs, e0._wpr_s[offc][:, None])): errs.append('wpr_rows')
        nev['write'] += int((b0 & 2).bool().sum()); nev['cap'] += int(c0.sum())
        nev['fin'] += int((e0.s_fin[:B] & ~e.s_fin[:B]).sum()); nev['wait'] += int((e0.s_a[:B] > e.s_a[:B]).sum())
        nev['rep'] += int((e0.s_nrep[:B] > e.s_nrep[:B]).sum())
        if errs:
            nbad += 1
            if nbad <= 5: print('MISMATCH trial', t, 'B', B, 'aw', e.aw_on, errs)
    print(f'PURIST_FUZZ trials {T} mismatches {nbad} events {nev}')
    return nbad


def prim_test(T=200000):
    """primitives on adversarial fp32 values: comparator, select, integer compare / one-hot."""
    g = torch.Generator().manual_seed(7)
    x = torch.randn(T, generator=g) * torch.pow(10.0, torch.randint(-40, 38, (T,), generator=g).float())
    sp = torch.tensor([0.0, -0.0, 1e-45, -1e-45, 1.17e-38, 3.4e38, -3.4e38, 0.5, 0.7, 1e-9, -30.0, 1e30, -1e30])
    a = torch.cat([x, sp, x]).float()
    b = torch.cat([torch.nextafter(x, torch.full_like(x, float('inf'))) if T else x, sp.flip(0), torch.nextafter(x, -torch.full_like(x, float('inf')))]).float()
    perm = torch.randperm(a.numel(), generator=g)
    a2 = torch.cat([a, a]); b2 = torch.cat([b, a[perm]])
    bad = int((PU.gtf(a2, b2).bool() != (a2 > b2)).sum()) + int((PU.gtf(b2, a2).bool() != (b2 > a2)).sum())
    c = (torch.rand(a2.numel(), generator=g) < 0.5).double()
    m = PU.mux(c, a2.double(), b2.double()).float()
    ok = (a2.abs() < 1e18) & (b2.abs() < 1e18)         # select bound |v| < BIG = 2^60
    ref = torch.where(c.bool(), a2, b2)
    mo, ro = m[ok], ref[ok]
    bad += int((mo != ro).sum()) + int(((mo == ro) & (ro != 0) & (mo.view(torch.int32) != ro.view(torch.int32))).sum())
    nz = int(((ro == 0) & (mo.view(torch.int32) != ro.view(torch.int32))).sum())      # -0.0 selected -> +0.0 (sign of zero only)
    i = torch.randint(-3000, 3000, (T,), generator=g).double(); j = torch.randint(-3000, 3000, (T,), generator=g).double()
    bad += int((PU.ge(i, j).bool() != (i >= j)).sum()) + int((PU.eqc(i, j).bool() != (i == j)).sum())
    oh = PU.dec(torch.arange(-2, 40).double(), 37)
    bad += int(not torch.equal(oh[2:39], torch.eye(37, dtype=torch.float64))) + int(oh[[0, 1, 39, 40, 41]].abs().sum() > 0)
    print(f'PURIST_PRIM values {a2.numel()} mismatches {bad} (signed-zero only: {nz})')
    return bad


def vote_fuzz(T):
    """voter alone vs aw_adapt.vote_share, probe logits chosen so scores tie exactly or differ inside the 1e-6 band."""
    rng = random.Random(99); g = torch.Generator().manual_seed(5); nbad = 0; nfrac = 0
    for t in range(T):
        B = 64
        base = torch.tensor([0.85, 0.0, 2.0, -1.0])
        eps = torch.tensor([0.0, 3e-7, 6e-7, 1.2e-6, 2.4e-6, 4e-6])
        lg = (base[torch.randint(0, 4, (B, NS), generator=g)] + eps[torch.randint(0, 6, (B, NS), generator=g)]).float()
        pool = torch.stack([torch.tensor((list(range(i)) + [PAD] * KS)[:KS]) for i in range(1, 4)])
        sym = pool[torch.randint(0, 3, (B, NS), generator=g)]
        ok = torch.rand(B, NS, generator=g) < 0.85; bad = torch.rand(B, NS, generator=g) < 0.2
        n = torch.randint(1, NS + 1, (B,), generator=g)
        ks, kv = torch.tensor(rng.random() < 0.8), torch.tensor(rng.choice([0, 1]))
        h0, w0, s0, q0 = AWA.vote_share(ok, bad, lg, sym, n, ks, kv)
        h1, w1, s1, q1, win = PU.vote_share(ok.double(), bad.double(), lg, sym.double(), n.double(), ks, kv)
        assert float(win.sum(1).max()) <= 1
        if not (torch.equal(h0, h1.bool()) and torch.equal(w0, w1.long()) and same(s0, s1) and torch.equal(q0, q1)):
            nbad += 1
        sc = torch.einsum('bij,bj->bi', q0, ok.float() * torch.sigmoid(lg))
        nfrac += int(((sc[:, :, None] - sc[:, None, :]).abs().lt(1e-6) & (sc[:, :, None] != sc[:, None, :])).sum())
    print(f'PURIST_VOTE trials {T} mismatches {nbad} (near-tie pairs seen ~{nfrac})')
    return nbad


if __name__ == '__main__':
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    bad = prim_test() + vote_fuzz(max(1, T // 3)) + main(T)
    sys.exit(1 if bad else 0)
