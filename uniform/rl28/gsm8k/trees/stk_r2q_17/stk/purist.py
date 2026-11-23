"""task [18](a) PURIST pass: the serial registers, the voter, the adaptive comparators, the probe capture, the WPR offset
and the forget side labels of comp_serial.SerialW / aw_adapt as literal fixed ReLU layers.

Every operation below is one of
  * a fixed affine map (elementwise affine = matmul with a fixed diagonal weight; contractions = matmul / einsum with fixed
    0/1 tables: shift table SH, DFA tables, symbol / phrase direction tables, arange encoders), or
  * ReLU.
No torch.where / comparison operator / argmax / boolean logic / gather by a register / clamp / min / max is used on a register.
Registers keep their integer / bool storage (the storage format of the KV-cache-like state); every read decodes the stored
value with a fixed ReLU layer (integer -> one-hot), every write encodes with a fixed linear map (one-hot -> integer = arange).
The token-indexed tables (DFA row, symbol, stop, newline flags) are one fixed implant embedding E_imp[tok], read by the input
token exactly like the host reads its embedding E[tok].

Exactness (bit-identical to the reference circuit): the circuit runs in float64 on values that are small integers or fp32
numbers, so
  integer compare  [x >= c] = 1 - relu(1 - relu(x - c + 1))                 (x - c integer: exact)
  float compare    [a > b]  = 1 - relu(1 - relu(KF (a - b)))  KF = 2^200     (any nonzero fp32 difference saturates)
  select           c ? a : b = gate(c, a) + gate(1 - c, b),  gate(c, v) = relu(v + BIG(c-1)) - relu(-v + BIG(c-1))
                    (c = 1: relu(v) - relu(-v) = v exactly; c = 0: 0 for |v| < BIG = 2^60)
  AND / OR / NOT    relu(sum - (n-1)) / 1 - relu(1 - sum) / 1 - x
The float arithmetic of the original circuit (probe logit, sigmoid weight, score einsum, the 1e6 tie band, the beat row sum,
the share division) is kept verbatim in fp32 so the voter's numbers are the same bits; only its compares/selects change.
Remaining non-ReLU fixed elementwise maps (disclosed): sigmoid of the probe logit (vote weight) and one division (winner share),
both arithmetic, as in the reference."""
import torch

DT = torch.float64
BIG = 2.0 ** 60
KF = 2.0 ** 200
R = torch.relu


def f64(t):
    return t.to(DT) if torch.is_tensor(t) else float(t)


def ge(x, c=0.0):                 # [x >= c], x - c integer valued
    return 1 - R(1 - R(x - c + 1))


def gtf(a, b):                    # [a > b], a, b float (fp32 values, compared in fp64)
    return 1 - R(1 - R(KF * (f64(a) - f64(b))))


def AND(*xs):
    return R(sum(xs) - (len(xs) - 1))


def OR(*xs):
    return 1 - R(1 - sum(xs))


def gate(c, v):                   # c in {0,1}: v or 0
    t = BIG * (c - 1)
    return R(v + t) - R(-v + t)


def mux(c, a, b):
    return gate(c, a) + gate(1 - c, b)


def dec(x, n):                    # integer x -> one-hot over 0..n-1 (all-zero outside)
    j = torch.arange(n + 1, device=x.device, dtype=DT)
    g = ge(x.unsqueeze(-1), j)
    return g[..., :-1] - g[..., 1:]


def eqc(x, c):                    # [x == c], integers
    return ge(x, c) - ge(x, c + 1)


def minc(x, c):                   # min(x, c)
    return x - R(x - c)


def maxi(a, b):                   # max(a, b), integers
    return b + R(a - b)


def enc(oh):                      # one-hot -> index (fixed arange weights)
    return oh @ torch.arange(oh.shape[-1], device=oh.device, dtype=DT)


def bsum(x, dim):                 # OR over a dim
    return 1 - R(1 - x.sum(dim))


def ball(x, dim):                 # AND over a dim
    return R(x.sum(dim) - (x.shape[dim] - 1))


def read2(Rf, ohA, ohK):          # Rf (B, NS, KS) nonneg ints, one-hot pointers -> Rf[b, a, k]
    sel = AND(ohA[:, :, None], ohK[:, None, :])
    return gate(sel, Rf).sum((1, 2))


# ------------------------------------------------------------------------------------------------ fixed tables
def setup(eng):
    """fixed implant embedding E_imp (V, nS + 5) and the decoded state tables (built once from the reference tables)."""
    dev = eng.t_TT.device
    sym = f64(eng.t_sym)
    eng.p_E = torch.cat([f64(eng.t_TT), sym[:, None], f64(eng.t_sym >= 0)[:, None], f64(eng.t_stop)[:, None],
                         f64(eng.t_ENL)[:, None], f64(eng.t_SNL)[:, None]], 1).contiguous()
    eng.p_numst, eng.p_done, eng.p_FLAG = f64(eng.t_numst), f64(eng.t_done), f64(eng.t_FLAG)
    eng.p_SH = f64(eng.t_SH)
    eng.p_W = torch.cat([f64(eng.t_DROW), f64(eng.t_DSYM)], 0)       # (P_ + 13, D) direction rows
    eng.p_nS = eng.p_numst.shape[0]
    if getattr(eng, '_wpr_c', None) is not None:                                       # WPR rotary table (fixed)
        eng._p_wcs = torch.cat([f64(eng._wpr_c), f64(eng._wpr_s)], 1)
    return eng


def side_labels(eng, B, slot):
    """forget_ws side coordinates: a = min(seg, E-1), lab = max(a - [slot < s_fr], 0) -> fp32 one-hots (oh_lab, oh_a)."""
    E = eng.w_E
    a = minc(f64(eng.s_nf[:B]), E - 1)
    before = 1 - ge(f64(slot), f64(eng.s_fr[:B]))
    lab = R(a - before)
    return dec(lab, E).float(), dec(a, E).float()


def wpr_offset(eng, B, slot):
    """offset register: s_fr > 0 ? s_fr - s_qlen : 0"""
    fr = f64(eng.s_fr[:B])
    return gate(ge(fr, 1), fr - f64(eng.s_qlen[:B]))


def wpr_rows(eng, off):
    """fixed rotary table row of the offset register as a one-hot matmul: (B, 1, HD) cos / sin"""
    oh = dec(off, eng._p_wcs.shape[0])
    cs = (oh @ eng._p_wcs).float()
    HD = eng._wpr_c.shape[1]
    return cs[:, None, :HD].contiguous(), cs[:, None, HD:].contiguous()


def probe_capture(eng, B, cap_probe, vl):
    """s_lg = cap ? vl : s_lg ; s_lgset |= cap"""
    eng.s_lg[:B] = mux(cap_probe, f64(vl), f64(eng.s_lg[:B])).float()
    eng.s_lgset[:B] = OR(f64(eng.s_lgset[:B]), cap_probe).bool()


# ------------------------------------------------------------------------------------------------ voter
def vote_share(ok_r, bad_r, lg_r, Rf, nslots, k_san, k_vm, NSYM=13):
    """purist comp_serial._vote / aw_adapt.vote_share. ok_r, bad_r (B, NS) {0,1} f64; lg_r (B, NS) fp32; Rf (B, NS, KS) f64;
    nslots (B,) or scalar f64. returns has, widx (f64), share (fp32), eq (fp32), win (f64)."""
    dev = ok_r.device; NS, KS = Rf.shape[1], Rf.shape[2]
    ar = torch.arange(NS, device=dev, dtype=DT)
    nsl = nslots if nslots.dim() == 0 else nslots[:, None]
    used = 1 - ge(ar[None], nsl)
    ok = AND(ok_r, used)
    sane = AND(ok, 1 - bad_r)
    anys = AND(bsum(sane, 1)[:, None], f64(k_san))
    keep = AND(ok, OR(sane, 1 - anys))
    Rh = dec(Rf, NSYM).float()                                                   # == F.one_hot(R_sym, 13).float()
    same = torch.einsum('bikc,bjkc->bij', Rh, Rh)
    eq = torch.relu(same - (KS - 1))
    sg = torch.sigmoid(lg_r)
    w = mux(eqc(f64(k_vm), 1), f64(sg), torch.ones_like(ok)).float()
    kw = gate(keep, f64(w)).float()
    score = torch.einsum('bij,bj->bi', eq, kw)
    dlt = score[:, :, None] - score[:, None, :]
    earlier = torch.ones(NS, NS, device=dev).triu(1)                              # fixed constant table [i < j]
    adl = torch.relu(dlt) + torch.relu(-dlt)
    t = torch.relu(1e6 * dlt) + torch.relu(1 - 1e6 * adl) * earlier
    t = f64(t)
    beat = t - R(t - 1)                                                           # min(t, 1)
    dropj = (1 - keep)[:, None, :]
    beat = dropj + R(beat - dropj)                                                # max(beat, dropped_j), dropped in {0,1}
    beat = (beat.float() + torch.eye(NS, device=dev)[None])
    y = torch.relu(beat.sum(-1) - (NS - 1))
    win = AND(keep, gtf(y, 0.5))
    has = bsum(win, 1)
    widx = enc(win)
    top = gate(win, f64(score)).sum(1).float()
    tot = kw.sum(1)
    c9 = torch.full_like(tot, 1e-9)
    totc = mux(gtf(tot, c9), f64(tot), f64(c9)).float()                            # max(tot, 1e-9)
    share = gate(has, f64(top / totc)).float()
    return has, widx, share, eq, win


def ext_more(n, share, k_ext, k_N, k_nmax, k_blk, k_tau, NS):
    """k_ext & n >= N & n < Nmax & (n % blk != 0 | share < tau); n integer in 0..NS (f64)"""
    m = torch.arange(NS + 1, device=n.device, dtype=DT)
    multiple = (eqc(n[:, None] - m[None] * f64(k_blk), 0)).sum(1)                # [n % blk == 0] (blk >= 1, n <= NS)
    return AND(f64(k_ext), ge(n, f64(k_N)), 1 - ge(n, f64(k_nmax)), OR(1 - multiple, gtf(k_tau, share)))


def stop_early(ok_r, bad_r, lg_r, eq, n, k_stop, k_n0, k_slg):
    NS = ok_r.shape[1]
    ar = torch.arange(NS, device=ok_r.device, dtype=DT)
    used = 1 - ge(ar[None], f64(k_n0))
    g = OR(AND(ok_r, 1 - bad_r, 1 - gtf(k_slg, lg_r), gtf(eq[:, 0, :], 0.5)), 1 - used)
    return AND(f64(k_stop), eqc(n, f64(k_n0)), ball(g, 1))


def decide(eng, B, end, valid, fl, rbad, ok_r, bad_r, lg_r, Rf, s_a, s_rep):
    n = s_a + 1
    has, widx, share, eq, _ = vote_share(ok_r, bad_r, lg_r, Rf, n, eng.k_san, eng.k_vm)
    room = ge(f64(eng.maxnew) - f64(eng.glen[:B]) - (eng.acap + 56), 0)
    ext = ext_more(n, share, eng.k_ext, eng.k_N, eng.k_nmax, eng.k_blk, eng.k_tau, ok_r.shape[1])
    more = AND(OR(1 - ge(n, f64(eng.k_N)), ext), room)
    more = AND(more, 1 - stop_early(ok_r, bad_r, lg_r, eq, n, eng.k_stop, eng.k_n0, eng.k_slg))
    kR = f64(eng.k_R)
    rlim = mux(AND(valid, eqc(fl, 2)), maxi(kR, f64(eng.k_rni)), kR + 0 * fl)
    do_rep = AND(end, rbad, 1 - ge(s_rep, rlim), room)
    do_wait = AND(end, more, 1 - do_rep)
    do_fin = AND(end, 1 - more, 1 - do_rep)
    return do_rep, do_wait, do_fin, has, widx


# ------------------------------------------------------------------------------------------------ one decode step
def sw_step(eng, B, tok, NS, KS, NSYM, PAD):
    """purist comp_serial.SerialW._sw: same registers, same outputs (write (B, D) fp32, bits int8, cap_probe f64)."""
    dev = eng.dev
    e = eng.p_E[tok]                                                              # implant embedding row of the input token
    nS = eng.p_nS
    TTrow, sy, isy, stop, ENL, SNL = e[:, :nS], e[:, nS], e[:, nS + 1], e[:, nS + 2], e[:, nS + 3], e[:, nS + 4]
    live = R(1 - f64(eng.done[:B]) - f64(eng.s_fin[:B]))
    rd = R(live - f64(eng.s_wr[:B]))
    dfa0 = f64(eng.s_dfa[:B]); oh0 = dec(dfa0, nS)
    nxt = gate(oh0, TTrow).sum(1)
    dfa1 = mux(rd, nxt, dfa0); oh1 = dec(dfa1, nS)
    W0 = eng.W0
    cap_probe = AND(rd, oh1[:, W0], 1 - oh0[:, W0])
    numst1, done1, done0, fl = oh1 @ eng.p_numst, oh1 @ eng.p_done, oh0 @ eng.p_done, oh1 @ eng.p_FLAG
    s_k = f64(eng.s_k[:B]); s_c = f64(eng.s_c[:B]); s_a = f64(eng.s_a[:B]); s_p = f64(eng.s_p[:B])
    store = AND(rd, numst1, isy, 1 - ge(s_k, KS))
    a = minc(s_a, NS - 1); kk = minc(s_k, KS - 1)
    ohA, ohK = dec(a, NS), dec(kk, KS)
    Rf = f64(eng.R_sym[:B])
    wsel = AND(AND(ohA[:, :, None], ohK[:, None, :]), store[:, None, None])
    Rf = mux(wsel, sy[:, None, None].expand_as(Rf), Rf)
    s_k = s_k + store
    s_c = s_c + rd
    closed = AND(rd, done1, 1 - done0)
    pe0 = f64(eng.s_pe[:B])
    stopin = AND(rd, OR(stop, AND(pe0, SNL)))
    pe1 = mux(live, ENL, pe0)
    capped = AND(rd, ge(s_c + 2, eng.acap))
    idle = 1 - ge(s_p, 1)
    end = AND(live, idle, OR(closed, stopin, capped))
    valid = OR(closed, AND(rd, done1))
    k_rm = f64(eng.k_rm)
    bad = OR(1 - valid, AND(OR(eqc(fl, 1), eqc(fl, 2)), 1 - k_rm))
    oh = AND(ohA, end[:, None])
    ok_r = mux(oh, valid[:, None].expand_as(oh), f64(eng.R_ok[:B]))
    bad_r = mux(oh, bad[:, None].expand_as(oh), f64(eng.R_bad[:B]))
    lgv = mux(f64(eng.s_lgset[:B]), f64(eng.s_lg[:B]), torch.full_like(dfa0, -30.0))
    lg_r = mux(oh, lgv[:, None].expand_as(oh), f64(eng.R_lg[:B])).float()
    rbad = mux(k_rm, 1 - valid, bad)
    s_rep = f64(eng.s_rep[:B])
    if eng.aw_on:
        do_rep, do_wait, do_fin, has, widx = decide(eng, B, end, valid, fl, rbad, ok_r, bad_r, lg_r, Rf, s_a, s_rep)
    else:
        do_rep = AND(end, rbad, 1 - ge(s_rep, f64(eng.k_R)))
        do_wait = AND(end, 1 - do_rep, 1 - ge(s_a + 1, f64(eng.k_N)))
        do_fin = AND(end, 1 - do_rep, 1 - do_wait)
        has, widx, _, _, _ = vote_share(ok_r, bad_r, lg_r, Rf, f64(eng.k_N), eng.k_san, eng.k_vm)
    s_win = mux(do_fin, mux(has, widx, torch.full_like(widx, -1.0)), f64(eng.s_win[:B]))
    # writer pointer
    ohp = dec(s_p, eng.P_)
    p_next = enc(ohp @ eng.p_SH)
    wsl = R(s_win)
    ohW = dec(wsl, NS)
    r0, re = eng.r_dig0, eng.r_eos
    dk = minc(R(p_next - r0), KS - 1)
    dsym = read2(Rf, ohW, dec(dk, KS))
    indig = AND(ge(p_next, r0), 1 - ge(p_next, re))
    p_next = mux(AND(indig, eqc(dsym, PAD)), torch.full_like(p_next, re), p_next)
    enl = mux(stopin, pe0, ENL)
    st = lambda i: torch.full_like(s_p, eng.b_start[i])
    p_new = mux(do_rep, st(0), mux(do_wait, mux(enl, st(2), st(1)),
                mux(do_fin, mux(has, mux(enl, st(4), st(3)), torch.full_like(s_p, re)), p_next)))
    p_new = gate(live, p_new)
    ohpn = dec(p_new, eng.P_)
    dig = AND(ge(p_new, r0), 1 - ge(p_new, re))
    dsym2 = read2(Rf, ohW, dec(minc(R(p_new - r0), KS - 1), KS))
    sel = torch.cat([AND(ohpn, 1 - dig[:, None]), AND(dec(dsym2, NSYM), dig[:, None])], 1)
    writing = ge(p_new, 1)
    write = ((sel @ eng.p_W).float() * eng.gain) * writing[:, None].float()
    fin1 = OR(f64(eng.s_fin[:B]), eqc(p_new, re))
    # segment change
    newseg = OR(do_rep, do_wait)
    s_nf = f64(eng.s_nf[:B]) + newseg
    s_fr = mux(newseg, f64(eng.slot[:B]) + 1, f64(eng.s_fr[:B]))
    s_a2 = s_a + do_wait
    s_rep2 = mux(do_wait, torch.zeros_like(s_rep), s_rep + do_rep)
    s_nrep = f64(eng.s_nrep[:B]) + do_rep
    clr = AND(dec(minc(s_a2, NS - 1), NS), newseg[:, None])
    Rf = mux(clr[:, :, None].expand_as(Rf), torch.full_like(Rf, PAD), Rf)
    s_dfa = gate(1 - OR(newseg, do_fin), dfa1)
    keepseg = 1 - newseg
    lgset = AND(f64(eng.s_lgset[:B]), keepseg)
    # ---- register stores (storage format)
    L = torch.long
    eng.R_sym[:B] = Rf.to(L); eng.R_ok[:B] = ok_r.bool(); eng.R_bad[:B] = bad_r.bool(); eng.R_lg[:B] = lg_r
    eng.s_k[:B] = gate(keepseg, s_k).to(L); eng.s_c[:B] = gate(keepseg, s_c).to(L)
    eng.s_pe[:B] = pe1.bool(); eng.s_win[:B] = s_win.to(L); eng.s_p[:B] = p_new.to(L)
    eng.s_fin[:B] = fin1.bool(); eng.s_wr[:B] = writing.bool()
    eng.s_nf[:B] = s_nf.to(L); eng.s_fr[:B] = s_fr.to(L); eng.s_a[:B] = s_a2.to(L); eng.s_rep[:B] = s_rep2.to(L)
    eng.s_nrep[:B] = s_nrep.to(L); eng.s_dfa[:B] = s_dfa.to(L); eng.s_lgset[:B] = lgset.bool()
    bits = (writing * 2 + AND(OR(do_rep, do_wait, do_fin), live) * 8).to(torch.int8)
    return write, bits, cap_probe
