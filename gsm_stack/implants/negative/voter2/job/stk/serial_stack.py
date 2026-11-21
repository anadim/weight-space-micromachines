"""SERIAL TRIES + VOTER implant, one generation stream, forward-only (no rewind, no restart, no second call).

Three parts, all fixed tables / tensor ops inside the captured CUDA-graph decode step (state per row, indexed by rowid):

 1. WRITER ("Wait" implant). When the host's sampled next token would END the answer (the engine's stop rule: eos, a
    token containing a blank line, or '\n' followed by a '\n'-token), or the attempt hits its token cap, and fewer than
    N attempts are done (and, adaptive mode, the attempt is flagged: impossible answer OR probe logit < thr), the end
    token is replaced by a queued token sequence (an attempt also ends right after its '#### number' closes)  ['\n' if the text does not end in '\n'] "Wait, let me solve it
    again.\n"  and the host keeps writing in the same context.
 2. REGISTER (exact answer reader). A per-row automaton over the tokens the host writes, equal to the scorer's regex
    ####\s*(-?[\d,]*\.?\d+) on the first '####' of each attempt (Qwen writes one digit per token):
      phase 0 wait for '####' (hash counter over all-'#' tokens) -> 1 (ws*) -> 2 ('-') -> 3 (int digits / commas)
      -> 4 ('.') -> 5 (fraction digits) -> 6 (closed). Value accumulated in float64, sign, "fraction nonzero".
    The probe (verifier) logit w . h_L + b is read from the residual at the token that completes '####' (layer L
    input), the same read as ProbeEngine. At the end of attempt k the register slot k stores: value, valid,
    impossible (negative / non-integer / missing), probe logit, and the answer's own tokens (to be re-emitted).
 3. VOTER. After the last attempt: keep = sane slots (or all valid slots if none is sane); score of a slot = summed
    weight of kept slots with the same value (weight 1 = count, sigmoid(logit) = probe vote; vmode 2 = last kept
    slot); earliest best slot wins. The writer then emits  ['\n'] "####" + that slot's answer tokens + eos, exactly as
    the calculator writes its result tokens. Only this final '####' is graded (last '#### number' in the stream).
    Adaptive mode with a single (unflagged) attempt writes nothing: the row ends as the plain engine would.

Silence: with N = 1 (or adaptive with no flag) the row is token-for-token the plain engine.
The act is a token replacement (equivalent to an infinite-gain write onto the forced token's LM-head row), as in the
sanity implant; the reader / register / voter are exact tables and arithmetic on row state.
"""
import torch, triton, triton.language as tl, torch.nn.functional as F
import fastqwen as Q
import hostkern as HK            # replaced by calc v2's sampler proxy at setup (stack port)
from engine import Engine

# forget-mask attention kernel: copied verbatim from reading/comp_sanity_v2_maskkernel.py (reading agent)
@triton.jit
def attn_suffix_mask_kernel(q, Kc, Vc, slot, lo_p, hi_p, m_in, l_in, acc_in, out, sK_b, sK_h, sK_s, scale,
                            H: tl.constexpr, HD: tl.constexpr, BN: tl.constexpr, IEEE: tl.constexpr):
    """hostkern.attn_suffix_kernel plus a per-row hard mask: keys j in [lo, hi) are not attended (hi <= lo: none)."""
    r = tl.program_id(0); g = tl.program_id(1)
    mi = tl.arange(0, 16)
    qm = mi < 2
    head = 2 * g + mi
    d = tl.arange(0, HD)
    Q = tl.load(q + (r * H + head[:, None]) * HD + d[None, :], mask=qm[:, None], other=0.0)
    o = r * H + head
    m = tl.load(m_in + o, mask=qm, other=0.0); l = tl.load(l_in + o, mask=qm, other=1.0)
    acc = tl.load(acc_in + o[:, None] * HD + d[None, :], mask=qm[:, None], other=0.0)
    n = tl.arange(0, BN)
    L = tl.load(slot + r) + 1
    lo = tl.load(lo_p + r); hi = tl.load(hi_p + r)
    for j0 in range(0, L, BN):
        jj = j0 + n
        km = (jj < L) & ((jj < lo) | (jj >= hi))
        K = tl.load(Kc + r * sK_b + g * sK_h + jj[:, None] * sK_s + d[None, :], mask=km[:, None], other=0.0)
        if IEEE:
            s = tl.dot(Q, tl.trans(K), input_precision='ieee') * scale
        else:
            s = tl.dot(Q, tl.trans(K)) * scale
        s = tl.where(km[None, :], s, float('-inf'))
        mn = tl.maximum(m, tl.max(s, axis=1))
        pr = tl.exp(s - mn[:, None])
        al = tl.exp(m - mn)
        l = l * al + tl.sum(pr, axis=1)
        V = tl.load(Vc + r * sK_b + g * sK_h + jj[:, None] * sK_s + d[None, :], mask=km[:, None], other=0.0)
        if IEEE:
            acc = acc * al[:, None] + tl.dot(pr.to(V.dtype), V, input_precision='ieee')
        else:
            acc = acc * al[:, None] + tl.dot(pr.to(V.dtype), V)
        m = mn
    res = acc / l[:, None]
    tl.store(out + o[:, None] * HD + d[None, :], res.to(out.dtype.element_ty), mask=qm[:, None])



# weight-space forget (attempt-label q/k extra dims): kernel from the stack agent's final/code/forget_ws.py,
# with a rowid indirection for the side cache
@triton.jit
def attn_suffix_bias_kernel(q, Kc, Vc, slot, Qx, Kx, sKx_b, rowid, m_in, l_in, acc_in, out, wmax, sK_b, sK_h, sK_s, scale,
                            H: tl.constexpr, HD: tl.constexpr, BN: tl.constexpr, IEEE: tl.constexpr, E: tl.constexpr,
                            DBG: tl.constexpr):
    """hostkern.attn_suffix_kernel with E extra (un-rotated) q/k dims: score_j = q.k_j*scale + qx.kx_j.
    DBG: atomic-max of the softmax numerator over keys with a negative extra term (abandoned keys) -> wmax[r]
    (an upper bound on their final attention weight)."""
    r = tl.program_id(0); g = tl.program_id(1)
    mi = tl.arange(0, 16)
    qm = mi < 2
    head = 2 * g + mi
    d = tl.arange(0, HD)
    Qv = tl.load(q + (r * H + head[:, None]) * HD + d[None, :], mask=qm[:, None], other=0.0)
    o = r * H + head
    m = tl.load(m_in + o, mask=qm, other=0.0); l = tl.load(l_in + o, mask=qm, other=1.0)
    acc = tl.load(acc_in + o[:, None] * HD + d[None, :], mask=qm[:, None], other=0.0)
    n = tl.arange(0, BN)
    e = tl.arange(0, E)
    rid = tl.load(rowid + r)
    qx = tl.load(Qx + rid * E + e)
    L = tl.load(slot + r) + 1
    for j0 in range(0, L, BN):
        jj = j0 + n
        km = jj < L
        K = tl.load(Kc + r * sK_b + g * sK_h + jj[:, None] * sK_s + d[None, :], mask=km[:, None], other=0.0)
        if IEEE:
            s = tl.dot(Qv, tl.trans(K), input_precision='ieee') * scale
        else:
            s = tl.dot(Qv, tl.trans(K)) * scale
        kx = tl.load(Kx + rid * sKx_b + jj[:, None] * E + e[None, :], mask=km[:, None], other=0.0)
        bias = tl.sum(kx * qx[None, :], axis=1)
        s = s + bias[None, :]
        s = tl.where(km[None, :], s, float('-inf'))
        mn = tl.maximum(m, tl.max(s, axis=1))
        pr = tl.exp(s - mn[:, None])
        if DBG:
            ab = (bias < 0) & km
            pm = tl.max(tl.where(ab[None, :] & qm[:, None], pr, 0.0))
            tl.atomic_max(wmax + r, pm)
        al = tl.exp(m - mn)
        l = l * al + tl.sum(pr, axis=1)
        V = tl.load(Vc + r * sK_b + g * sK_h + jj[:, None] * sK_s + d[None, :], mask=km[:, None], other=0.0)
        if IEEE:
            acc = acc * al[:, None] + tl.dot(pr.to(V.dtype), V, input_precision='ieee')
        else:
            acc = acc * al[:, None] + tl.dot(pr.to(V.dtype), V)
        m = mn
    res = acc / l[:, None]
    tl.store(out + o[:, None] * HD + d[None, :], res.to(out.dtype.element_ty), mask=qm[:, None])



import os as _os0
_PLAIN = _os0.environ.get('STK_PLAINATT') == '1'
WAIT = "Wait, let me solve it again.\n"
NMAX = 8
ATOK = 16          # answer tokens stored per slot
QL = 48            # writer queue length


class SerialEngine(Engine):
    def serial_setup(self, acap=320, layer=18, wait=WAIT):
        tok, dev = self.tok, self.dev
        V = self.W['E'].shape[0]
        txt = [tok.decode([i]) for i in range(len(tok))] + [''] * max(0, V - len(tok))
        cls, dig, nh = [], [], []
        for t in txt:
            s = t.strip()
            nh.append(len(s) if (s and set(s) == {'#'}) else 0)
            if t != '' and s == '': c = 1                                   # whitespace only
            elif s == '-' and t.lstrip() == '-' : c = 2                     # ' -' / '-'
            elif len(t) == 1 and t in '0123456789': c = 3
            elif t == ',': c = 4
            elif t == '.': c = 5
            else: c = 0
            cls.append(c); dig.append(int(t) if c == 3 else 0)
        self.r_cls = torch.tensor(cls, device=dev, dtype=torch.long)
        self.r_dig = torch.tensor(dig, device=dev, dtype=torch.float64)
        self.r_nh = torch.tensor(nh, device=dev, dtype=torch.long)
        self.r_sp = torch.tensor([t.startswith(' ') for t in txt], device=dev)   # ' -' has a leading space
        T_ = lambda s: tok(s, add_special_tokens=False)['input_ids']
        self.q_wait = [T_(wait), T_('\n' + wait)]            # [text ends in '\n', does not]
        self.q_hash = [T_('####'), T_('\n####')]
        self.acap = int(acap); self.v_L = int(layer)
        Bp = self.B_max + 1
        z = lambda *s, dt=torch.long: torch.zeros(*s, device=dev, dtype=dt)
        # writer queue
        self.qs, self.ql, self.qp = z(Bp, QL), z(Bp), z(Bp)
        self.qfin = z(Bp, dt=torch.bool)                     # queue ends the row (final line)
        # reader state
        self.ph, self.hc = z(Bp), z(Bp)
        self.cl = z(Bp, dt=torch.bool)                      # the answer number closed at this token
        self.val = z(Bp, dt=torch.float64); self.fsc = z(Bp, dt=torch.float64)
        self.neg, self.fnz, self.hasd = z(Bp, dt=torch.bool), z(Bp, dt=torch.bool), z(Bp, dt=torch.bool)
        self.atok, self.an = z(Bp, ATOK), z(Bp)
        self.lg = z(Bp, dt=torch.float32); self.lgset = z(Bp, dt=torch.bool)
        # attempts + register
        self.att, self.astart = z(Bp), z(Bp)
        self.R_val = z(Bp, NMAX, dt=torch.float64); self.R_ok = z(Bp, NMAX, dt=torch.bool)
        self.R_bad = z(Bp, NMAX, dt=torch.bool); self.R_lg = z(Bp, NMAX, dt=torch.float32)
        self.R_tok, self.R_n = z(Bp, NMAX, ATOK), z(Bp, NMAX)
        self.R_end = z(Bp, NMAX)                                # glen at the end of each attempt (token accounting)
        self.pick = torch.full((Bp,), -1, device=dev, dtype=torch.long)
        self.m_hi = torch.full((Bp,), -1, device=dev, dtype=torch.long)   # forget mask end (by rowid); keys [q_end, m_hi) hidden
        self.k_forget = torch.zeros((), device=dev, dtype=torch.bool)
        # ---- stack port: S2F repair inside each attempt + segment label for the weight-space forget
        self.seg = z(Bp); self.rep = z(Bp); self.seg_new = z(Bp, dt=torch.bool)
        self.k_R = torch.zeros((), device=dev, dtype=torch.long)          # repairs allowed per attempt (0 = off)
        self.k_repmiss = torch.zeros((), device=dev, dtype=torch.bool)    # repair only missing answers (GSM-Hard)
        self.k_san = torch.ones((), device=dev, dtype=torch.bool)         # voter sanity filter
        self.c_rep = torch.tensor(T_(" We need to find"), device=dev)
        self.k_posfix = torch.zeros((), device=dev, dtype=torch.bool)
        import os as _os
        self.ws = _os.environ.get('SER_WS') == '1'                     # weight-space forget instead of the key-range mask
        self.w_E = 16; self.w_M = float(_os.environ.get('SER_WS_M', 1e4))
        self.w_Kx = torch.zeros(Bp, self.S_max, self.w_E, device=dev)  # per-key attempt label (side coordinates), by rowid
        self.w_Qx = torch.zeros(Bp, self.w_E, device=dev)
        self.w_wmax = torch.zeros(Bp, device=dev)
        I_ = torch.eye(self.w_E)
        self.w_WK = I_.to(dev); self.w_WQ = (-(torch.ones(self.w_E, self.w_E) - I_)).to(dev)
        # knobs (tensors: one graph for all settings)
        self.k_N = torch.ones((), device=dev, dtype=torch.long)
        self.k_thr = torch.full((), 1e9, device=dev)            # adaptive: continue only if flagged; 1e9 = always continue
        self.k_adapt = torch.zeros((), device=dev, dtype=torch.bool)
        self.k_vmode = torch.zeros((), device=dev, dtype=torch.long)   # 0 count, 1 probe, 2 last
        self.v_w = torch.zeros(Q.D, device=dev); self.v_b = torch.zeros((), device=dev)
        self.graphs = {}; self.pool = None; self._outshape = None
        # fixed token constants
        self.c_wait = [torch.tensor(s, device=dev) for s in self.q_wait]
        self.c_hash = [torch.tensor(s, device=dev) for s in self.q_hash]
        # ---- verifier v2 (SER_V2=1): multi-layer '####' read + running mean of the attempt's line-end states, and an
        # weighted voter over (v2 logit, L18 logit, sane): w = sigmoid|exp(th . f + b0). Reads only (no write into the residual stream).
        self.v2 = _os.environ.get('SER_V2') == '1'
        self.v2_L = (14, 18, 22, 27)
        self.r_nl = torch.tensor(['\n' in t for t in txt], device=dev)
        self.v2_W = torch.zeros(len(self.v2_L), Q.D, 2, device=dev)     # [:, :, 0] '####' read, [:, :, 1] line-end mean read
        self.v2_b = torch.zeros((), device=dev)
        self.v2_th = torch.zeros(3, device=dev, dtype=torch.float64)    # voter weights on (v2 clip -12, lg clip -10, sane)
        self.v2_b0 = torch.zeros((), device=dev, dtype=torch.float64)
        self.v2_sig = torch.zeros((), device=dev, dtype=torch.bool)       # slot weight sigmoid(z) (else exp(z))
        self.v2_hz, self.v2_ms, self.v2_mc = z(Bp, dt=torch.float32), z(Bp, dt=torch.float32), z(Bp, dt=torch.float32)
        self.v2_wr = z(Bp, dt=torch.bool)                       # the next input token is written by the queue
        self.v2_nl = z(Bp, dt=torch.bool)                       # per-step temp: input token is a line end of the attempt body
        self.R_v2 = torch.full((Bp, NMAX), -30.0, device=dev)
        self.pick0 = torch.full((Bp,), -1, device=dev, dtype=torch.long)  # the base voter's pick (diagnostic)
        if self.v2:
            P2 = torch.load(_os.environ['SER_V2_W'], map_location='cpu')
            self.v2_W[:, :, 0] = P2['wh'].float().view(len(self.v2_L), Q.D).to(dev)
            self.v2_W[:, :, 1] = P2['wm'].float().view(len(self.v2_L), Q.D).to(dev)
            self.v2_b.fill_(float(P2['b'])); self.v2_th.copy_(torch.tensor(P2['th'], dtype=torch.float64)); self.v2_b0.fill_(float(P2['b0']))
            self.v2_sig.fill_(P2.get('kind', 'exp') == 'sig')
        return self

    def set_serial(self, N=1, adaptive=False, thr=1e9, vmode=1, w=None, b=0.0, forget=False, posfix=False, R=0, repmiss=False, san=True):
        self.k_R.fill_(int(R)); self.k_repmiss.fill_(bool(repmiss)); self.k_san.fill_(bool(san))
        self.k_forget.fill_(bool(forget)); self.k_posfix.fill_(bool(posfix)); self.k_N.fill_(N); self.k_adapt.fill_(bool(adaptive)); self.k_thr.fill_(float(thr)); self.k_vmode.fill_(vmode)
        if w is not None: self.v_w.copy_(w.to(self.v_w)); self.v_b.fill_(float(b))

    def _mut(self):
        m = super()._mut()
        if hasattr(self, 'qs'):
            m += [self.qs, self.ql, self.qp, self.qfin, self.ph, self.hc, self.val, self.fsc, self.neg, self.fnz, self.hasd,
                  self.cl, self.atok, self.an, self.lg, self.lgset, self.att, self.astart, self.R_val, self.R_ok, self.R_bad, self.R_lg,
                  self.R_tok, self.R_n, self.R_end, self.pick, self.m_hi, self.seg, self.rep, self.seg_new]
            if self.v2: m += [self.v2_hz, self.v2_ms, self.v2_mc, self.v2_wr, self.v2_nl, self.R_v2, self.pick0]
        return m

    # ---------------- register: read the input token ----------------
    def _read(self, B, tok, r):
        c = self.r_cls[tok]; d = self.r_dig[tok]; nh = self.r_nh[tok]
        ph = self.ph[r]
        hc = torch.where(nh > 0, self.hc[r] + nh, torch.zeros_like(nh)); self.hc[r] = hc
        start = (ph == 0) & (hc >= 4)
        live = ~self.done[:B] & (self.qp[r] >= self.ql[r])       # tokens written by the queue are not read
        # transitions for phases 1..5 (6 = closed)
        p1 = ph == 1; p2 = ph == 2; p3 = ph == 3; p4 = ph == 4; p5 = ph == 5
        ws, mi, dg, cm, dt = c == 1, c == 2, c == 3, c == 4, c == 5
        nph = ph.clone()
        nph = torch.where(p1 & ws, 1, nph)
        nph = torch.where(p1 & mi, 2, nph)
        nph = torch.where((p1 | p2) & (dg | cm), 3, nph)
        nph = torch.where((p1 | p2) & dt, 4, nph)
        nph = torch.where(p3 & (dg | cm), 3, nph)
        nph = torch.where(p3 & dt, 4, nph)
        nph = torch.where(p4 & dg, 5, nph)
        nph = torch.where(p5 & dg, 5, nph)
        inph = (ph >= 1) & (ph <= 5)
        close = inph & (nph == ph) & ~((p1 & ws) | (p3 & (dg | cm)) | (p5 & dg))    # no valid transition -> close
        nph = torch.where(close, 6, nph)
        self.cl[r] = live & close
        acc = inph & ~close
        intd = acc & dg & ((p1 | p2 | p3))
        frd = acc & dg & (p4 | p5)
        self.val[r] = torch.where(live & intd, self.val[r] * 10 + d, self.val[r])
        fsc = torch.where(live & frd, self.fsc[r] / 10, self.fsc[r])
        fsc = torch.where(fsc == 0, torch.ones_like(fsc), fsc)
        self.val[r] = torch.where(live & frd, self.val[r] + d * fsc, self.val[r])
        self.fsc[r] = fsc
        self.fnz[r] = self.fnz[r] | (live & frd & (d > 0))
        self.hasd[r] = self.hasd[r] | (live & dg & acc)
        self.neg[r] = self.neg[r] | (live & acc & mi & p1)
        # store accepted tokens (to re-emit)
        an = self.an[r]
        put = live & acc & (an < ATOK)
        self.atok[r, an.clamp(max=ATOK - 1)] = torch.where(put, tok, self.atok[r, an.clamp(max=ATOK - 1)])
        self.an[r] = an + put.long()
        self.ph[r] = torch.where(live & start, torch.ones_like(ph), torch.where(live, nph, ph))
        if self.v2:     # sampled (not queue-written) token containing '\n', before the attempt's '####'
            self.v2_nl[r] = ~self.done[:B] & ~self.v2_wr[r] & self.r_nl[tok] & (ph == 0) & ~start
        return live & start

    def _reset_reader(self, r, m):
        for t in (self.ph, self.hc, self.an):
            t[r] = torch.where(m, torch.zeros_like(t[r]), t[r])
        self.val[r] = torch.where(m, torch.zeros_like(self.val[r]), self.val[r])
        self.fsc[r] = torch.where(m, torch.zeros_like(self.fsc[r]), self.fsc[r])
        for t in (self.neg, self.fnz, self.hasd, self.lgset):
            t[r] = t[r] & ~m
        if self.v2:
            for t in (self.v2_ms, self.v2_mc):
                t[r] = torch.where(m, torch.zeros_like(t[r]), t[r])

    # ---------------- writer + register commit + voter (after the host's sample) ----------------
    def _act(self, B, tok, nxt, lp, logits, r):
        qp, ql = self.qp[r], self.ql[r]
        inq = qp < ql
        qtok = self.qs[r, qp.clamp(max=QL - 1)]
        enl = self.ends_nl[tok]
        stopping = self.has_nn[nxt] | (nxt == self.eos) | (enl & self.starts_nl[nxt])
        alen = self.glen[:B] - self.astart[r]
        capped = alen + 2 >= self.acap
        end = ~inq & ~self.done[:B] & (stopping | capped | self.cl[r])
        att = self.att[r]
        # commit the attempt to register slot att
        ph = self.ph[r]
        valid = (ph >= 3) & (ph <= 6) & self.hasd[r]
        bad = ~valid | self.neg[r] | self.fnz[r]
        v = torch.where(self.neg[r], -self.val[r], self.val[r])
        a1 = att.clamp(max=NMAX - 1)
        oh = F.one_hot(a1, NMAX).bool() & end[:, None]
        self.R_val[r] = torch.where(oh, v[:, None], self.R_val[r])
        self.R_ok[r] = torch.where(oh, valid[:, None], self.R_ok[r])
        self.R_bad[r] = torch.where(oh, bad[:, None], self.R_bad[r])
        lgv = torch.where(self.lgset[r], self.lg[r], torch.full_like(self.lg[r], -30.0))
        self.R_lg[r] = torch.where(oh, lgv[:, None], self.R_lg[r])
        if self.v2:
            v2v = torch.where(self.lgset[r], self.v2_hz[r] + self.v2_ms[r] / self.v2_mc[r].clamp(min=1.0) + self.v2_b,
                              torch.full_like(lgv, -30.0))
            self.R_v2[r] = torch.where(oh, v2v[:, None], self.R_v2[r])
        self.R_tok[r] = torch.where(oh[:, :, None], self.atok[r][:, None, :], self.R_tok[r])
        self.R_n[r] = torch.where(oh, self.an[r][:, None], self.R_n[r])
        self.R_end[r] = torch.where(oh, self.glen[:B][:, None], self.R_end[r])
        flagged = bad | (lgv < self.k_thr)
        more = (att + 1 < self.k_N) & (~self.k_adapt | flagged)
        # S2F repair inside the attempt: an impossible answer (missing only on GSM-Hard) restarts THIS attempt with
        # ' We need to find', the failed segment hidden (forget) and positions continuing from the question end
        rbad = torch.where(self.k_repmiss, ~valid, bad)
        do_rep = end & rbad & (self.rep[r] < self.k_R)
        do_wait = end & more & ~do_rep
        nslots = att + 1
        do_final = end & ~more & ((nslots > 1) | ~self.k_adapt) & (self.k_N > 1) & ~do_rep
        # ---- voter over slots 0..att ----
        ar = torch.arange(NMAX, device=self.dev)[None]
        used = ar < nslots[:, None]
        ok = self.R_ok[r] & used
        sane = ok & ~self.R_bad[r]
        keep = torch.where(sane.any(1, keepdim=True) & self.k_san, sane, ok)
        wgt = torch.where(self.k_vmode == 1, torch.sigmoid(self.R_lg[r].double()), torch.ones_like(self.R_val[r]))
        Vv = self.R_val[r]
        same = (Vv[:, :, None] - Vv[:, None, :]).abs() <= 1e-6 * torch.clamp(Vv[:, None, :].abs(), min=1.0)
        score = ((same & keep[:, None, :]).double() * (wgt * keep)[:, None, :]).sum(-1)
        score = torch.where(keep, score, torch.full_like(score, -1.0))
        last = torch.where(keep, ar.double(), torch.full_like(score, -1.0))
        score = torch.where(self.k_vmode == 2, last, score - 1e-9 * ar.double())
        pick = torch.where(keep.any(1), score.argmax(1), torch.full_like(att, -1))
        if self.v2:
            self.pick0[r] = torch.where(do_final, pick, self.pick0[r])
            Fv = torch.stack([self.R_v2[r].double().clamp(min=-12.0), self.R_lg[r].double().clamp(min=-10.0), sane.double()], -1)
            z2 = Fv @ self.v2_th + self.v2_b0
            w2 = torch.where(self.v2_sig, torch.sigmoid(z2), torch.exp(z2.clamp(-30.0, 30.0)))
            sc2 = ((same & keep[:, None, :]).double() * (w2 * keep)[:, None, :]).sum(-1)
            sc2 = torch.where(keep, sc2, torch.full_like(sc2, -1.0)) - 1e-9 * ar.double()
            pick = torch.where(keep.any(1), sc2.argmax(1), torch.full_like(att, -1))
        self.pick[r] = torch.where(do_final, pick, self.pick[r])
        # ---- fill the queue ----
        Bq = B
        nw = len(self.q_wait[0]); nw1 = len(self.q_wait[1]); nhh = len(self.q_hash[0]); nhh1 = len(self.q_hash[1])
        qw = torch.full((Bq, QL), self.eos, device=self.dev, dtype=torch.long)
        qw[:, :nw1] = torch.where(enl[:, None], F.pad(self.c_wait[0], (0, nw1 - nw), value=self.eos)[None], self.c_wait[1][None])
        lw_ = torch.where(enl, nw, nw1)
        # final: ['\n'] '####' + slot tokens + eos
        qf = torch.full((Bq, QL), self.eos, device=self.dev, dtype=torch.long)
        hs = torch.where(enl[:, None], F.pad(self.c_hash[0], (0, nhh1 - nhh), value=self.eos)[None], self.c_hash[1][None])
        qf[:, :nhh1] = hs
        hl = torch.where(enl, nhh, nhh1)
        pk = pick.clamp(min=0)
        st = self.R_tok[r][torch.arange(Bq, device=self.dev), pk]                  # (B, ATOK)
        sn = self.R_n[r][torch.arange(Bq, device=self.dev), pk]
        j = torch.arange(ATOK, device=self.dev)[None]
        dst = (hl[:, None] + j).clamp(max=QL - 1)
        qf.scatter_(1, dst, torch.where(j < sn[:, None], st, torch.full_like(st, self.eos)))
        lf = torch.where(pick >= 0, hl + sn + 1, torch.ones_like(hl))               # + eos;  no answer -> just eos
        qf[:, 0] = torch.where(pick >= 0, qf[:, 0], torch.full_like(qf[:, 0], self.eos))
        nr = self.c_rep.numel()
        qr = torch.full((Bq, QL), self.eos, device=self.dev, dtype=torch.long); qr[:, :nr] = self.c_rep[None]
        newq = torch.where(do_rep[:, None], qr, torch.where(do_wait[:, None], qw, qf))
        setq = do_wait | do_final | do_rep
        self.qs[r] = torch.where(setq[:, None], newq, self.qs[r])
        self.ql[r] = torch.where(do_rep, torch.full_like(lw_, nr), torch.where(do_wait, lw_, torch.where(do_final, lf, self.ql[r])))
        self.qp[r] = torch.where(setq, torch.ones_like(qp), torch.where(inq, qp + 1, qp))
        self.qfin[r] = self.qfin[r] | do_final
        first = newq[:, 0]
        new = torch.where(inq, qtok, torch.where(setq, first, nxt))
        # attempt bookkeeping
        self.att[r] = att + do_wait.long()
        self.astart[r] = torch.where(do_wait, self.glen[:B] + lw_, torch.where(do_rep, self.glen[:B] + nr, self.astart[r]))
        newseg = do_wait | do_rep
        self.seg[r] = self.seg[r] + newseg.long(); self.seg_new[r] = newseg
        self.rep[r] = torch.where(do_wait, torch.zeros_like(self.rep[r]), self.rep[r] + do_rep.long())
        # forget mask: from the next token on, the new attempt sees the question + the phrase + itself
        self.m_hi[r] = torch.where(newseg & self.k_forget, self.slot[:B] + 1, self.m_hi[r])
        self._reset_reader(r, newseg)
        # a capped attempt that is not continued / finalised ends the row
        force_eos = end & capped & ~setq
        new = torch.where(force_eos, torch.full_like(new, self.eos), new)
        changed = inq | setq | force_eos
        lsm = torch.log_softmax(logits.float() / self.temp, -1).gather(-1, new[:, None])[:, 0]
        lp = torch.where(changed, lsm, lp)
        wbit = torch.where(inq | setq, 2, 0).to(torch.int8) + torch.where(setq, 8, 0).to(torch.int8)
        if self.v2: self.v2_wr[r] = inq | setq
        # the queue must not be ended by the stop rule: rows still writing stay live
        return new, lp, wbit, inq | setq

    def _step_triton(self, B, S):
        import engine as _EM; HK = _EM.HK
        W, tok = self.W, self.tokb[:B]
        slot = self.slot[:B]
        imp = self.imp if self.imp_on else None
        emb = W['E'][tok]
        write = wf = None
        if imp is not None:
            e = imp.step(emb, B, tok) * self.row_on[:B, None]
            if self.use_write:
                write = (e @ self.Dp) if self.Dp is not None else (imp.last_w if imp.fk is not None else e @ imp.D)
            wf = (e.abs().sum(-1) > 0).to(torch.int8)
        r = self.rowid[:B]
        vfire = self._read(B, tok, r)
        lo_b = (slot - self.glen[:B]).contiguous()            # first answer slot (question end) for live rows
        hi_b = self.m_hi[r].contiguous()
        # position register: with the mask on, the new attempt's RoPE positions continue from the question end
        # (pos = Lp + slot - hidden_span), so the attempt sits right after the question as in a fresh sample
        off_b = torch.where((hi_b > lo_b) & self.k_posfix, hi_b - lo_b, torch.zeros_like(hi_b))
        x, h, q, att = self.xb[:B], self.hb[:B], self.qb[:B], self.ab[:B]
        x.copy_(emb)
        pos = self.Lp + slot - off_b
        ieee = x.dtype == torch.float32
        sc = 1.0 / (Q.HD ** 0.5)
        delta = None
        v2acc = None
        for i, lw in enumerate(W['L']):
            if i == self.v_L:
                if delta is not None: x += delta; delta = None
                vlg = x.float() @ self.v_w + self.v_b
                self.lg[r] = torch.where(vfire, vlg, self.lg[r]); self.lgset[r] = self.lgset[r] | vfire
            HK.add_rms_kernel[(B,)](x, delta if delta is not None else x, lw['ln1'], h, D=Q.D, HAS_DELTA=delta is not None, eps=Q.EPS)
            if self.v2 and i in self.v2_L:           # x = layer-i input (residual), read only
                v2a = x.float() @ self.v2_W[self.v2_L.index(i)]
                v2acc = v2a if v2acc is None else v2acc + v2a
            qkv = F.linear(h, lw['Wqkv'])
            Kc, Vc = self.Kc[i], self.Vc[i]
            HK.qk_rope_kernel[(B,)](qkv, lw['qn'], lw['kn'], self.cos_t, self.sin_t, pos, q, Kc, Vc, slot,
                                    Kc.stride(0), Kc.stride(1), Kc.stride(2), H=Q.H, KVH=Q.KVH, HD=Q.HD, eps=Q.EPS)
            rbk = 8 if ieee else 32; ns = 1 if ieee else 2
            HK.attn_prefix_kernel[(Q.KVH, triton.cdiv(B, rbk))](q, self.Kp[i], self.Vp[i], self.mp, self.lpf, self.accp, B,
                                                                self.Lp, sc, H=Q.H, HD=Q.HD, RBK=rbk, BN=64 if not ieee else 32,
                                                                IEEE=ieee, num_warps=4, num_stages=ns)
            if _PLAIN:                          # diagnostic: the host's own suffix-attention kernel (no forget possible)
                HK.attn_suffix_kernel[(B, Q.KVH)](q, Kc, Vc, slot, self.mp, self.lpf, self.accp, att,
                                                  Kc.stride(0), Kc.stride(1), Kc.stride(2), sc, H=Q.H, HD=Q.HD, BN=64 if not ieee else 32,
                                                  IEEE=ieee, num_warps=4, num_stages=1 if ieee else 2)
            elif self.ws:
                if i == 0:                      # side coordinates of this step: label of the key written now, query one-hot
                    av = self.seg[r].clamp(max=self.w_E - 1)
                    on = self.k_forget
                    self.w_Kx[r, slot] = (F.one_hot(av, self.w_E).float() @ self.w_WK) * on
                    self.w_Qx[r] = (F.one_hot(av, self.w_E).float() @ self.w_WQ) * self.w_M * on
                attn_suffix_bias_kernel[(B, Q.KVH)](q, Kc, Vc, slot, self.w_Qx, self.w_Kx, self.w_Kx.stride(0), r, self.mp, self.lpf,
                                                    self.accp, att, self.w_wmax, Kc.stride(0), Kc.stride(1), Kc.stride(2), sc,
                                                    H=Q.H, HD=Q.HD, BN=64 if not ieee else 32, IEEE=ieee, E=self.w_E, DBG=False,
                                                    num_warps=4, num_stages=1 if ieee else 2)
            else:
                attn_suffix_mask_kernel[(B, Q.KVH)](q, Kc, Vc, slot, lo_b, hi_b, self.mp, self.lpf, self.accp, att,
                                                    Kc.stride(0), Kc.stride(1), Kc.stride(2), sc, H=Q.H, HD=Q.HD, BN=64 if not ieee else 32,
                                                    IEEE=ieee, num_warps=4, num_stages=1 if ieee else 2)
            o = F.linear(att.view(B, Q.H * Q.HD), lw['Wo'])
            HK.add_rms_kernel[(B,)](x, o, lw['ln2'], h, D=Q.D, HAS_DELTA=True, eps=Q.EPS)
            gu = F.linear(h, lw['Wgu'])
            act = self.actb[:B]
            HK.silu_mul_kernel[(B, Q.FF // 1024)](gu, act, FF=Q.FF, BLK=1024)
            delta = F.linear(act, lw['Wd'])
        if self.v2:
            self.v2_hz[r] = torch.where(vfire, v2acc[:, 0], self.v2_hz[r])
            nlf = self.v2_nl[r]
            self.v2_ms[r] = self.v2_ms[r] + torch.where(nlf, v2acc[:, 1], torch.zeros_like(v2acc[:, 1]))
            self.v2_mc[r] = self.v2_mc[r] + nlf.float()
        x += delta
        if write is not None:
            x += write.to(x.dtype)
        HK.add_rms_kernel[(B,)](x, x, W['lnf'], h, D=Q.D, HAS_DELTA=False, eps=Q.EPS)
        logits = F.linear(h, W['E'])
        nxt, lp = self.nxt_b[:B], self.lp_b[:B]
        HK.sample_kernel[(B,)](logits, self.temp, self.greedy, self.seed, nxt, lp, logits.shape[1], BV=2048, num_warps=8)
        self.seed += 1
        new, lp2, wbit, writing = self._act(B, tok, nxt, lp, logits, r)
        wf2 = wbit if wf is None else (wf | wbit)
        self._book(B, tok, new, lp2, wf2, imp if imp is not None else None)
        # queue tokens never trigger the stop rule; a final-line queue ends the row when its eos is written
        qend = self.qfin[r] & (self.qp[r] >= self.ql[r])
        self.done[:B] = torch.where(writing & ~qend, torch.zeros_like(self.done[:B]), self.done[:B]) | (writing & qend)

    def _wave(self, qs, G, T, max_new, greedy):
        for t in (self.qs, self.ql, self.qp, self.ph, self.hc, self.atok, self.an, self.att, self.astart, self.R_tok, self.R_n, self.R_end, self.seg, self.rep, self.seg_new):
            t.zero_()
        for t in (self.val, self.fsc, self.R_val, self.lg, self.R_lg):
            t.zero_()
        for t in (self.cl, self.qfin, self.neg, self.fnz, self.hasd, self.lgset, self.R_ok, self.R_bad):
            t.zero_()
        self.pick.fill_(-1); self.m_hi.fill_(-1)
        if self.ws: self.w_Kx.zero_(); self.w_Qx.zero_()
        if self.v2:
            for t in (self.v2_hz, self.v2_ms, self.v2_mc, self.v2_wr, self.v2_nl): t.zero_()
            self.R_v2.fill_(-30.0); self.pick0.fill_(-1)
        st = super()._wave(qs, G, T, max_new, greedy)
        n = len(qs) * G
        self._rec.append(dict(val=self.R_val[:n].cpu(), ok=self.R_ok[:n].cpu(), bad=self.R_bad[:n].cpu(), lg=self.R_lg[:n].cpu(),
                              natt=(self.att[:n] + 1).cpu(), nseg=(self.seg[:n] + 1).cpu(), end=self.R_end[:n].cpu(), pick=self.pick[:n].cpu(),
                              v2=self.R_v2[:n].cpu(), pick0=self.pick0[:n].cpu()))
        return st

    def generate(self, qids, **kw):
        self._rec = []
        o = super().generate(qids, **kw)
        o['reg'] = {k: torch.cat([d[k] for d in self._rec]) for k in self._rec[0]}
        return o
