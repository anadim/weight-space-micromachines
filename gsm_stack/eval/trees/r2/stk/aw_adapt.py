"""A+B+C+stop4 for the all-weights serial stack (comp_serial.SerialW), as fixed tensor ops on the register (no host decision).
Port of implants/adapt (serial_stack.py vote_slots / ext_more / stop_early / integer guard / cap) onto the symbol register.
  A  adaptive N: after the base N tries, extend in blocks of blk while the winner's probe-weighted share < tau, up to Nmax
     (compiled comparator: relu(tau * total - winner score) > 0 on the voter's own score registers).
  S  stop at n0: write the vote after exactly n0 tries if they are all sane, unanimous (symbol-equal) and sigmoid(lg) >= th.
  B  integer guard: a valid non-integer answer (FLAG 2) may be repaired up to K times (k_R otherwise); s_rep is the counter.
  C  attempt cap: s_c (host tokens in the segment) against the cap constant (comp_serial self.acap).
  room budget guard: a new segment starts only if glen + acap + 56 <= max_new (a register compare).
Pure torch: imported by comp_serial and by the CPU fuzz test (test_aw.py)."""
import math, torch, torch.nn.functional as F

NSYM = 13


def vote_share(R_ok, R_bad, R_lg, R_sym, nslots, k_san, k_vm):
    """comp_serial._vote over slots < nslots (per row) plus the winner's share; same ReLU circuit.
    returns has (B,), widx (B,), share (B,) (0 if nothing kept), eq (B, NS, NS) all-symbol equality."""
    dev = R_ok.device; NS, KS = R_sym.shape[1], R_sym.shape[2]
    ar = torch.arange(NS, device=dev)
    used = ar[None] < nslots[:, None]
    ok = R_ok & used
    sane = ok & ~R_bad
    anys = sane.any(1, keepdim=True) & k_san
    keep = ok & (sane | ~anys)
    Rh = F.one_hot(R_sym, NSYM).float()
    same = torch.einsum('bikc,bjkc->bij', Rh, Rh)
    eq = torch.relu(same - (KS - 1))
    w = torch.where(k_vm == 1, torch.sigmoid(R_lg), torch.ones_like(R_lg))
    kw = keep.float() * w
    score = torch.einsum('bij,bj->bi', eq, kw)
    dlt = score[:, :, None] - score[:, None, :]
    earlier = (ar[:, None] < ar[None, :]).float()
    beat = torch.clamp(torch.relu(1e6 * dlt) + torch.relu(1 - 1e6 * dlt.abs()) * earlier, max=1.0)
    beat = torch.maximum(beat, (~keep).float()[:, None, :])
    beat = beat + torch.eye(NS, device=dev)[None]
    win = keep & (torch.relu(beat.sum(-1) - (NS - 1)) > 0.5)
    has = win.any(1); widx = win.float().argmax(1)
    top = (score * win.float()).sum(1)
    tot = kw.sum(1)
    share = torch.where(has, top / torch.clamp(tot, min=1e-9), torch.zeros_like(tot))
    return has, widx, share, eq


def ext_more(n, share, k_ext, k_N, k_nmax, k_blk, k_tau):
    return k_ext & (n >= k_N) & (n < k_nmax) & ((n % k_blk != 0) | (share < k_tau))


def stop_early(R_ok, R_bad, R_lg, eq, n, k_stop, k_n0, k_slg):
    used = torch.arange(R_ok.shape[1], device=R_ok.device)[None] < k_n0
    good = (R_ok & ~R_bad & (R_lg >= k_slg) & (eq[:, 0, :] > 0.5)) | ~used
    return k_stop & (n == k_n0) & good.all(1)


def parse_spec(spec):
    """'A=0.7/16/4:S=4/0.5:B=3:C=400' (any subset) -> dict"""
    d = {}
    for f in [x for x in (spec or '').split(':') if x]:
        k, v = f.split('=')
        d[k] = [float(x) for x in v.split('/')]
    return d


def setup(eng, spec):
    """knobs (fixed constants of the circuit). Every field optional; an empty spec leaves comp_serial untouched."""
    d = parse_spec(spec); dev = eng.dev
    t = lambda x, dt=torch.long: torch.tensor(x, device=dev, dtype=dt)
    eng.aw_on = bool(d)
    A = d.get('A'); S = d.get('S')
    eng.k_ext = t(A is not None, torch.bool)
    eng.k_tau = t(A[0] if A else 1.0, torch.float32); eng.k_nmax = t(int(A[1]) if A else 0); eng.k_blk = t(int(A[2]) if A else 1)
    if A: assert int(A[1]) <= eng.R_ok.shape[1], ('SW_NS too small for Nmax', A)
    eng.k_stop = t(S is not None, torch.bool); eng.k_n0 = t(int(S[0]) if S else 0)
    th = S[1] if S else 0.5
    eng.k_slg = t(math.log(th / (1 - th)) if 0 < th < 1 else (-1e30 if th <= 0 else 1e30), torch.float32)
    eng.k_rni = t(int(d['B'][0]) if 'B' in d else 0)
    if 'C' in d: eng.acap = int(d['C'][0])
    return d


def decide(eng, B, end, valid, fl, rbad):
    """replaces comp_serial's (do_rep, do_wait, do_fin) when eng.aw_on; returns them + the final voter's (has, widx)."""
    n = eng.s_a[:B] + 1
    has, widx, share, eq = vote_share(eng.R_ok[:B], eng.R_bad[:B], eng.R_lg[:B], eng.R_sym[:B], n, eng.k_san, eng.k_vm)
    room = eng.glen[:B] + eng.acap + 56 <= eng.maxnew
    more = ((n < eng.k_N) | ext_more(n, share, eng.k_ext, eng.k_N, eng.k_nmax, eng.k_blk, eng.k_tau)) & room
    more = more & ~stop_early(eng.R_ok[:B], eng.R_bad[:B], eng.R_lg[:B], eq, n, eng.k_stop, eng.k_n0, eng.k_slg)
    rlim = torch.where(valid & (fl == 2), torch.maximum(eng.k_R, eng.k_rni), eng.k_R)
    do_rep = end & rbad & (eng.s_rep[:B] < rlim) & room
    do_wait = end & more & ~do_rep
    do_fin = end & ~more & ~do_rep
    return do_rep, do_wait, do_fin, has, widx
