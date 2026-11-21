"""Serial tries + sanity repair + voter as a COMPILED forward-only implant (all-weights version of S3-serial).

Everything the implant does is a fixed table / fixed-weight tensor op on per-row registers, reading only the INPUT token
(the token the host or the implant wrote last) and the host's residual (probe), and acting only by residual writes
(gain * DIR[token], DIR solved against the frozen host head as in comp_sanity) and by the weight-space forget (forget_ws:
fixed extra q/k dims; attempt label = segment counter register).

registers per row (carried between steps like the KV cache; updated only by the fixed ops below)
  dfa      one-hot over the 34 states of the scorer's '#### number' automaton (table TT[token, state], san_engine)
  wr       1 bit: the input token was written by the implant (the reader skips it)
  c        tokens of the host in the current segment (cap counter)
  k        next answer-symbol position;  R_sym[slot, 16] one-hot over 13 symbols {0-9, '-', '.', pad}; commas / spaces
           are not stored (so '1,000' == '1000')
  lg       probe logit w . h_18 + b captured at the token that completes '####' (fixed linear map of the residual)
  R_ok, R_bad, R_lg   per attempt slot: closed answer, impossible (negative / non-integer / missing; GSM-Hard: missing), logit
  a, rep, seg (= s_nf), s_fr   attempt index, repairs in this attempt, segment counter and forget end (forget_ws inputs)
  p        pointer one-hot over the writer program rows: repair phrase | retry phrase (2 banks: text ends / does not end in
           '\n') | final prefix '####' ' ' (2 banks) | 16 digit rows | eos row;  SH = fixed shift table
  win      one-hot winner slot (voter output), fin (eos written: the generator's stop condition)
triggers (input token): END = answer closed (dfa enters a DONE state) OR input is a stop token (eos / blank line / '\n' after
  '\n') OR c reaches the cap.  On END: commit the slot; REPAIR if bad and rep < R; else RETRY if a + 1 < N; else FINAL.
voter (fixed ReLU circuit on the slot registers): eq_ij = relu(sum_k R_i[k] . R_j[k] - 15) (all 16 symbols equal);
  keep = ok AND (sane OR no slot sane); score_i = sum_j eq_ij keep_j w_j  (w = sigmoid(probe logit), or 1 for count);
  win_i = keep_i AND prod_{j != i} [score_i - score_j > 1e-6 OR (|.| <= 1e-6 AND i < j) OR NOT keep_j].
final writer: '####' ' ' then digit row k writes DIR[symbol R_win[k]], a pad symbol jumps to the eos row.
Generator (not weights, disclosed): the decode loop, T = 1 sampling, the token budget, and its stop rule = 'the implant wrote
eos' (host stop tokens are fed back as input tokens, where the END trigger reads them).
Programmatic, optional, for reference only: RL_POSRESET (RoPE positions of a new segment continue from the question end).
"""
import os, torch, triton, torch.nn.functional as F
import engine as EM
import fastqwen as Q
from c2_engine import C2Engine
import san_engine as SE

OVL = os.environ.get('FAST_OVL', '0') == '1'      # implant step on a side stream, joined after the last host layer
ATT3 = os.environ.get('FAST_ATT3', '0') == '1'    # split-KV prefix + forget-skip suffix attention (attn3.py; fastroll only)
import attn3
NS = int(os.environ.get('SW_NS', 8))          # attempt slots (AW: 16/24 for the adaptive-N extension)
import aw_adapt as AWA
KS = 16         # answer symbols per slot
NSYM = 13       # 0-9, '-', '.', pad(12)
PAD = 12
REP_PH = " We need to find"
RETRY = ["\nWait, let me solve it again.\n", "Wait, let me solve it again.\n"]      # text does not / does end in '\n'
FINAL = ["\n#### ", "#### "]


WPR = os.environ.get('WPR', '0') == '1'      # lane wpr: weight-space position reset (rotary re-phasing writer)
# ladnext: how the implant's R(-off) meets the host RoPE. 0 = rotate q / new K row after the host's bf16 RoPE (lane wpr);
# 1 = compose R(-off) R(pos) in fp32 (angle addition of the host's fp32 rotary row pos and the implant's fp32 row off) and
#     cast the composed row to the host table precision before the host's RoPE arithmetic (one rounding, as the host's own);
# 2 = fused: the composed row is the host's own bf16 table row at index pos - off (arithmetic identical to AW_PR=1).
WPR_K = int(os.environ.get('WPR_K', '0'))
VF = os.environ.get('AW_VF', '0') == '1'      # ladnext: answer-group voter (aw_vf.py) replaces the final pick
if VF: import aw_vf as AVF
SN = os.environ.get('AW_SN', '')            # lane sidenet: multi-layer linear verifier register + BLD writer (aw_sn.py)
if SN:
    assert VF, 'AW_SN needs AW_VF=1'
    import aw_sn as ASN
# lane dola: DOLA=l:alpha[:beta] -> contrastive layer logits on host-sampled steps (aw_dola.py); unset = rung-4 path untouched
DOLA = os.environ.get('DOLA', '')
if DOLA:
    import aw_dola as ADL
    DOLA_L, DOLA_A, DOLA_B = ADL.parse(DOLA)


def _wpr_rot(x, c, s):
    """x (B, h, HD) after the host RoPE at the absolute position; c, s (B, 1, HD) fp32 = table row of the offset
    register: returns R(-off) x = x cos - rot(x) sin (fixed linear map per row), in fp32, cast back."""
    xf = x.float()
    return (xf * c - Q.rot(xf) * s).to(x.dtype)


class SerialW(C2Engine):
    def _wpr_tab(self, off):
        if getattr(self, '_wpr_c', None) is None:    # fixed fp32 rotary table, rows 0..S_max (built once)
            inv = 1.0 / (Q.THETA ** (torch.arange(0, Q.HD, 2, dtype=torch.int64).to(device=self.dev, dtype=torch.float) / Q.HD))
            fr = torch.arange(self.Lp + self.S_max + 8, device=self.dev, dtype=torch.float64)[:, None] * inv.double()[None]
            emb = torch.cat([fr, fr], -1)
            self._wpr_c, self._wpr_s = emb.cos().float(), emb.sin().float()
            # (1, B_max+1): NOT a per-row register (a first dim of B_max+1 made fastroll move it on compaction)
            self._wpr_ar = torch.arange(self.B_max + 1, device=self.dev)[None]
            if WPR_K == 1:      # the host's rotary angles in fp32 (rope_tables before its bf16 cast)
                self._wpr_hc, self._wpr_hs = Q.rope_tables(self.Lp + self.S_max + 8, self.dev, torch.float32)
        return self._wpr_c[off][:, None], self._wpr_s[off][:, None]

    def sw_setup(self, N=4, acap=320, R=1, posreset=False, probe=None, cache=None, vmode=1, san=True, repmiss=False):
        tok, dev = self.tok, self.dev
        V = self.W['E'].shape[0]
        txt = [tok.decode([i]) for i in range(len(tok))] + [''] * max(0, V - len(tok))
        TT, FLAG, S = SE.build_tables(txt, cache)
        self.t_TT = TT.to(dev); self.t_FLAG = FLAG.to(dev); self.nS = len(S)
        self.t_done = torch.tensor([s[0] == 'X' for s in S], device=dev)
        self.t_numst = torch.tensor([s[0] in ('M', 'I', 'D', 'F') for s in S], device=dev)
        self.W0 = S.index(('W', 0))
        sym = torch.full((V,), -1, dtype=torch.long)
        for v, t in enumerate(txt):
            s_ = t.strip() if t.strip() == '-' else t
            if len(s_) == 1 and s_ in '0123456789': sym[v] = int(s_)
            elif s_ == '-' and t.lstrip() == '-': sym[v] = 10
            elif t == '.': sym[v] = 11
        self.t_sym = sym.to(dev)
        stop = torch.tensor([('\n\n' in t) for t in txt]); stop[self.eos] = True
        self.t_stop = stop.to(dev)
        T_ = lambda s: tok(s, add_special_tokens=False)['input_ids']
        # writer program rows: 0 idle | banks | digit rows | eos row
        banks = [T_(REP_PH), T_(RETRY[0]), T_(RETRY[1]), T_(FINAL[0]), T_(FINAL[1])]
        rows = [None]; start = []; ends = []
        for b in banks:
            start.append(len(rows)); rows += b; ends.append(len(rows) - 1)
        self.r_dig0 = len(rows); rows += [None] * KS
        self.r_eos = len(rows); rows += [self.eos]
        P_ = len(rows)
        SH = torch.zeros(P_, P_)
        SH[0, 0] = 1
        for i in range(1, P_):
            if i in ends[:3]: SH[i, 0] = 1                         # phrase banks end -> idle
            elif i in ends[3:]: SH[i, self.r_dig0] = 1             # final prefix -> first digit row
            elif self.r_dig0 <= i < self.r_eos - 1: SH[i, i + 1] = 1
            elif i == self.r_eos - 1: SH[i, self.r_eos] = 1
            elif i == self.r_eos: SH[i, 0] = 1
            else: SH[i, i + 1] = 1
        self.t_SH = SH.to(dev); self.b_start = start; self.P_ = P_
        fixed = [t for t in rows if t is not None]
        symtok = [T_(str(d))[0] for d in range(10)] + [T_('-')[0], T_('.')[0]]
        uniq = sorted(set(fixed + symtok + [self.eos]))
        D = self._solve_dirs(uniq)                                  # unit-rms directions, readout argmax = the token
        self.t_DIRtok = {t: D[i] for i, t in enumerate(uniq)}
        DROW = torch.zeros(P_, Q.D)
        for i, t in enumerate(rows):
            if t is not None: DROW[i] = D[uniq.index(t)]
        self.t_DROW = DROW.to(dev)
        DSYM = torch.zeros(NSYM, Q.D)
        for s_, t in enumerate(symtok): DSYM[s_] = D[uniq.index(t)]
        DSYM[PAD] = D[uniq.index(self.eos)]
        self.t_DSYM = DSYM.to(dev); self.gain = float(os.environ.get('SW_GAIN', 3e4))
        self.t_ENL = torch.tensor([t.endswith('\n') for t in txt], device=dev)
        self.t_SNL = torch.tensor([t.startswith('\n') for t in txt], device=dev)
        # probe
        pr = torch.load(probe); self.v_w = pr['w'].to(dev).float(); self.v_b = float(pr['b']); self.v_L = int(pr.get('layer', 18))
        # registers
        Bp = self.B_max + 1
        z = lambda *s, dt=torch.long: torch.zeros(*s, device=dev, dtype=dt)
        self.s_dfa, self.s_wr, self.s_c, self.s_k = z(Bp), z(Bp, dt=torch.bool), z(Bp), z(Bp)
        self.s_lg = z(Bp, dt=torch.float32); self.s_lgset = z(Bp, dt=torch.bool)
        self.R_sym = torch.full((Bp, NS, KS), PAD, device=dev, dtype=torch.long)
        self.R_ok, self.R_bad = z(Bp, NS, dt=torch.bool), z(Bp, NS, dt=torch.bool)
        self.R_lg = torch.full((Bp, NS), -30.0, device=dev)
        self.s_a, self.s_rep, self.s_nf = z(Bp), z(Bp), z(Bp)
        self.s_fr = torch.full((Bp,), -1, device=dev, dtype=torch.long)
        self.s_qlen = z(Bp); self.s_p = z(Bp); self.s_win = torch.full((Bp,), -1, device=dev, dtype=torch.long)
        self.s_fin = z(Bp, dt=torch.bool); self.s_nrep = z(Bp); self.s_pe = z(Bp, dt=torch.bool)
        self.k_N = torch.tensor(N, device=dev); self.k_R = torch.tensor(R, device=dev)
        self.k_vm = torch.tensor(vmode, device=dev); self.k_san = torch.tensor(bool(san), device=dev)
        self.k_rm = torch.tensor(bool(repmiss), device=dev)
        self.acap = int(acap); self.sw_posreset = bool(posreset)
        self.c_maxf = NS * 2; self.c_forget = True                   # forget_ws: segment labels 0..(slots x 2)
        self.sw_nmax = int(os.environ.get('SW_NMAX', 8))          # largest N this engine will run (dev evals)
        self.sw_maxn = int(os.environ.get('SW_MAXN', 0)) or self.sw_nmax * 2 * (acap + 16) + 40
        self.aw_on = False                                         # AW: aw_adapt.setup() turns A/S/B/C on
        self.graphs = {}; self.pool = None; self._outshape = None
        self.stack_setup()
        if VF: self._vf_setup(txt, S, Bp, z)
        if WPR: self._wpr_c = None; self._wpr_tab(torch.zeros(1, dtype=torch.long, device=dev))
        if DOLA: self._dola_h = torch.zeros(2 * Bp, Q.D, device=dev, dtype=self.xb.dtype); self._dola_plain = None
        return self

    def _vf_setup(self, txt, S, Bp, z):
        """vfeat registers: body-number automaton (numreg, K=32) read on host tokens while the '####' DFA is in a P state, reset
        per segment; question-number set (K=16, the same automaton over the question tokens, loaded at admission); segment
        token counter; per slot R_nn / R_qc / R_nt committed at END; s_pk0 = the base voter's pick (diagnostic)."""
        import numreg as NM
        dev = self.dev
        cls, dig = NM.token_tables(txt)
        self.v_cls = torch.tensor(cls, device=dev); self.v_dig = torch.tensor(dig, device=dev, dtype=torch.float64)
        self.t_isP = torch.tensor([s_[0] == 'P' for s_ in S], device=dev)
        self.vf_b = NM.NumReg(Bp, 32, dev)
        for nm_, t_ in zip(('st', 'E', 'pc', 'M', 'P', 'pfr', 'S', 'n'), self.vf_b.tensors()): setattr(self, 'vfb_' + nm_, t_)
        self.vfq_S = z(Bp, 16, dt=torch.float64); self.vfq_n = z(Bp)
        self.s_vnt = z(Bp); self.s_pk0 = torch.full((Bp,), -1, device=dev, dtype=torch.long)
        self.R_nn, self.R_qc, self.R_nt = (z(Bp, NS, dt=torch.float64) for _ in range(3))
        PV = torch.load(os.environ.get('AW_VF_W', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vf_w.pt')), map_location='cpu')
        self.vf_th = torch.tensor(PV['th'], dtype=torch.float64, device=dev); self.vf_c0 = float(PV['c0'])
        assert self.vf_th.numel() == 18
        self._vf_qkey = None
        if SN:      # side-net registers: '####' projection, line-end projection sum + count, per-step line-end gate, slot score
            self.sn_L, self.sn_W, self.sn_b, self.sn_th, self.sn_c0 = ASN.load(SN, dev)
            self.t_NL = torch.tensor(['\n' in t for t in txt], device=dev)
            self.s_snH, self.s_snM = z(Bp, dt=torch.float32), z(Bp, dt=torch.float32)
            self.s_sncnt = z(Bp); self.s_snml = z(Bp, dt=torch.bool); self.s_snhs = z(Bp, dt=torch.bool)
            self._sn_ns = None
            self.R_sn = torch.full((Bp, NS), -30.0, device=dev, dtype=torch.float64)
            self._sn_u = torch.zeros(Bp, 2, device=dev, dtype=torch.float32)

    def _vf_regs(self):
        return self.vf_b.tensors() + [self.vfq_S, self.vfq_n, self.s_vnt, self.s_pk0, self.R_nn, self.R_qc, self.R_nt] + (
            [self.s_snH, self.s_snM, self.s_sncnt, self.s_snml, self.s_snhs, self.R_sn] if SN else [])

    def vf_admit(self, srt, qids, pidx):
        """question-number registers of the admitted rows srt (question index pidx): the numreg automaton is run once per
        generate() over all questions (a fixed token automaton, batched), and the admitted rows load their question's set."""
        import numreg as NM
        if self._vf_qkey is not qids:
            nq = len(qids); Tq = max(len(q) for q in qids)
            I = torch.zeros(nq, Tq, dtype=torch.long)
            for i, q in enumerate(qids): I[i, :len(q)] = torch.as_tensor(list(q), dtype=torch.long)
            I = I.to(self.dev); Lq = torch.tensor([len(q) for q in qids], device=self.dev)
            nr = NM.NumReg(nq, 16, self.dev)
            nr.run(torch.arange(nq, device=self.dev), self.v_cls[I], self.v_dig[I], Lq)
            self._vf_qtab = (nr.S, nr.n); self._vf_qkey = qids
        P_ = torch.as_tensor(pidx, device=self.dev, dtype=torch.long)
        self.vfq_S[srt] = self._vf_qtab[0][P_]; self.vfq_n[srt] = self._vf_qtab[1][P_]

    def set_sw(self, N=None, vmode=None, san=None, repmiss=None, R=None):
        if N is not None: self.k_N.fill_(N)
        if vmode is not None: self.k_vm.fill_(vmode)
        if san is not None: self.k_san.fill_(bool(san))
        if repmiss is not None: self.k_rm.fill_(bool(repmiss))
        if R is not None: self.k_R.fill_(R)

    def stack_setup(self):
        pass

    def _solve_dirs(self, ids):
        E = self.W['E'].float(); g = self.W['lnf'].float(); D = []
        for t in ids:
            d = (E[t] / g).clone().requires_grad_(True)
            opt = torch.optim.Adam([d], lr=1e-2)
            for it in range(400):
                h = g * d / d.pow(2).mean().sqrt(); lg = E @ h
                oth = lg.clone(); oth[t] = -1e9
                if (lg[t] - oth.max()).item() > 8.0 and it > 0: break
                opt.zero_grad(); (-(lg[t] - torch.logsumexp(oth, 0))).backward(); opt.step()
            d = d.detach(); d = d / d.pow(2).mean().sqrt(); D.append(d)
            assert int((E @ (g * d)).argmax()) == t
        return torch.stack(D)

    def _mut(self):
        m = super()._mut()
        if hasattr(self, 'R_sym'):
            m += [self.s_dfa, self.s_wr, self.s_c, self.s_k, self.s_lg, self.s_lgset, self.R_sym, self.R_ok, self.R_bad, self.R_lg,
                  self.s_a, self.s_rep, self.s_nf, self.s_fr, self.s_qlen, self.s_p, self.s_win, self.s_fin, self.s_nrep, self.s_pe]
            if VF: m += self._vf_regs()
        im = self.imp
        if hasattr(im, 'chain'): m += [im.chain.sym_h, im.chain.S_h, im.chain.key_h, im.chain.fst]
        return m

    def _prefill(self, qs, rows):
        out = EM.Engine._prefill(self, qs, rows)
        G = rows[0].numel(); n = len(qs) * G
        self.s_qlen[:n] = out[1].repeat_interleave(G)
        if VF: self.vf_admit(torch.arange(n, device=self.dev), qs, [p for p in range(len(qs)) for _ in range(G)])
        return out

    def _wave(self, qs, G, T, max_new, greedy):
        for t in (self.s_dfa, self.s_c, self.s_k, self.s_a, self.s_rep, self.s_nf, self.s_p, self.s_nrep): t.zero_()
        for t in (self.s_wr, self.s_lgset, self.R_ok, self.R_bad, self.s_fin, self.s_pe): t.zero_()
        self.s_lg.zero_(); self.R_lg.fill_(-30.0); self.R_sym.fill_(PAD); self.s_fr.fill_(-1); self.s_win.fill_(-1)
        if VF:
            for t_ in self._vf_regs(): t_.zero_()
            self.s_pk0.fill_(-1)
            if SN: self.R_sn.fill_(-30.0)
        st = C2Engine._wave(self, qs, G, T, max_new, greedy)
        n = len(qs) * G
        self._rec.append(dict(sym=self.R_sym[:n].cpu(), ok=self.R_ok[:n].cpu(), bad=self.R_bad[:n].cpu(), lg=self.R_lg[:n].cpu(),
                              natt=(self.s_a[:n] + 1).cpu(), nseg=(self.s_nf[:n] + 1).cpu(), win=self.s_win[:n].cpu(),
                              nrep=self.s_nrep[:n].cpu()))
        if VF: self._rec[-1].update(nn=self.R_nn[:n].cpu(), qc=self.R_qc[:n].cpu(), nt=self.R_nt[:n].cpu(), pk0=self.s_pk0[:n].cpu())
        if SN: self._rec[-1].update(sn=self.R_sn[:n].cpu())
        return st

    def generate(self, qids, **kw):
        self._rec = []
        o = super().generate(qids, **kw)
        o['reg'] = {k: torch.cat([d[k] for d in self._rec]) for k in self._rec[0]}
        return o

    def _stops(self, row, nt, glen, max_new):
        return False

    def _attn_suffix(self, i, B, q, Kc, Vc, slot, att, sc, ieee):
        EM.HK.attn_suffix_kernel[(B, Q.KVH)](q, Kc, Vc, slot, self.mp, self.lpf, self.accp, att,
                                             Kc.stride(0), Kc.stride(1), Kc.stride(2), sc, H=Q.H, HD=Q.HD, BN=64 if not ieee else 32,
                                             IEEE=ieee, num_warps=4, num_stages=1 if ieee else 2)

    def _vote(self, R_ok, R_bad, R_lg, R_sym):
        """the voter circuit: returns (has_winner, winner slot index) per row."""
        dev = R_ok.device
        used = torch.arange(NS, device=dev)[None] < self.k_N
        ok = R_ok & used
        sane = ok & ~R_bad
        anys = sane.any(1, keepdim=True) & self.k_san
        keep = ok & (sane | ~anys)
        Rh = F.one_hot(R_sym, NSYM).float()                                     # (B, NS, KS, 13)
        same = torch.einsum('bikc,bjkc->bij', Rh, Rh)
        eq = torch.relu(same - (KS - 1))                                        # 1 iff all 16 symbols agree
        w = torch.where(self.k_vm == 1, torch.sigmoid(R_lg), torch.ones_like(R_lg))
        kw = keep.float() * w
        score = torch.einsum('bij,bj->bi', eq, kw)
        dlt = score[:, :, None] - score[:, None, :]
        ar = torch.arange(NS, device=dev)
        earlier = (ar[:, None] < ar[None, :]).float()
        beat = torch.clamp(torch.relu(1e6 * dlt) + torch.relu(1 - 1e6 * dlt.abs()) * earlier, max=1.0)
        beat = torch.maximum(beat, (~keep).float()[:, None, :])                 # never loses to a dropped slot
        beat = beat + torch.eye(NS, device=dev)[None]
        win = keep & (torch.relu(beat.sum(-1) - (NS - 1)) > 0.5)
        return win.any(1), win.float().argmax(1)

    def sw_read_sim(self, ids):
        """the reader registers (DFA + symbol store) on one token sequence, with the same fixed tables (fuzz reference)."""
        d = 0; k = 0; syms = []; closed = False
        for t in ids:
            d1 = int(self.t_TT[t, d])
            sy = int(self.t_sym[t])
            if bool(self.t_numst[d1]) and sy >= 0 and k < KS: syms.append(sy); k += 1
            if bool(self.t_done[d1]) and not bool(self.t_done[d]): closed = True
            d = d1
            if closed: break
        return closed, syms, int(self.t_FLAG[d])

    # ---------------------------------------------------------------- the compiled circuit, one decode step
    def _sw(self, B, tok):
        dev = self.dev
        live = ~self.done[:B] & ~self.s_fin[:B]
        rd = live & ~self.s_wr[:B]                                    # the input token was written by the host
        dfa0 = self.s_dfa[:B]
        dfa1 = torch.where(rd, self.t_TT[tok, dfa0], dfa0)
        cap_probe = rd & (dfa1 == self.W0) & (dfa0 != self.W0)       # this token completes '####'
        # answer-symbol store (fixed table sym[tok]; written into slot a at position k)
        sy = self.t_sym[tok]
        store = rd & self.t_numst[dfa1] & (sy >= 0) & (self.s_k[:B] < KS)
        a = self.s_a[:B].clamp(max=NS - 1); kk = self.s_k[:B].clamp(max=KS - 1)
        bi = torch.arange(B, device=dev)
        cur = self.R_sym[bi, a, kk]
        self.R_sym[bi, a, kk] = torch.where(store, sy, cur)
        self.s_k[:B] = self.s_k[:B] + store.long()
        self.s_c[:B] = self.s_c[:B] + rd.long()
        if VF:      # body numbers: host tokens read while the '####' automaton is before '####' (P states); segment length
            self.vf_b.step(bi, self.v_cls[tok], self.v_dig[tok], rd & self.t_isP[dfa0])
            self.s_vnt[:B] = self.s_vnt[:B] + live.long()
        # END trigger
        closed = rd & self.t_done[dfa1] & ~self.t_done[dfa0]
        pe0 = self.s_pe[:B].clone()
        stopin = rd & (self.t_stop[tok] | (pe0 & self.t_SNL[tok]))     # eos / blank line / '\n' then '\n'
        self.s_pe[:B] = torch.where(live, self.t_ENL[tok], self.s_pe[:B])
        capped = rd & (self.s_c[:B] + 2 >= self.acap)
        idle = self.s_p[:B] == 0
        end = live & idle & (closed | stopin | capped)
        valid = closed | (rd & self.t_done[dfa1])
        fl = self.t_FLAG[dfa1]
        bad = ~valid | ((fl == 1) | (fl == 2)) & ~self.k_rm
        oh = F.one_hot(a, NS).bool() & end[:, None]
        self.R_ok[:B] = torch.where(oh, valid[:, None], self.R_ok[:B])
        self.R_bad[:B] = torch.where(oh, bad[:, None], self.R_bad[:B])
        lgv = torch.where(self.s_lgset[:B], self.s_lg[:B], torch.full_like(self.s_lg[:B], -30.0))
        self.R_lg[:B] = torch.where(oh, lgv[:, None], self.R_lg[:B])
        if VF:      # commit nn / qc / ntok of the attempt (a number still open at END is emitted by the flush)
            self.vf_b.flush(bi, end)
            nb_ = self.vf_b.n[:B]
            qc_ = AVF.vf_qc(self.vfq_S[:B], self.vfq_n[:B], self.vf_b.S[:B], nb_)
            self.R_nn[:B] = torch.where(oh, nb_.double()[:, None], self.R_nn[:B])
            self.R_qc[:B] = torch.where(oh, qc_[:, None], self.R_qc[:B])
            self.R_nt[:B] = torch.where(oh, self.s_vnt[:B].double()[:, None], self.R_nt[:B])
        if SN:      # commit the side-net score; gate this step's line-end read: host token with '\n', segment before '####'
            snv = self.s_snH[:B].double() + self.s_snM[:B].double() / self.s_sncnt[:B].clamp(min=1).double() + self.sn_b
            self.R_sn[:B] = torch.where(oh, snv[:, None], self.R_sn[:B])
            self.s_snml[:B] = rd & self.t_NL[tok] & ~self.s_snhs[:B] & ~cap_probe & ~end
        rbad = torch.where(self.k_rm, ~valid, bad)
        do_rep = end & rbad & (self.s_rep[:B] < self.k_R)
        do_wait = end & ~do_rep & (self.s_a[:B] + 1 < self.k_N)
        do_fin = end & ~do_rep & ~do_wait
        # ---- voter (fixed ReLU circuit over the slot registers)
        if self.aw_on:                                              # AW: adaptive N / stop4 / integer guard / room (aw_adapt.py)
            do_rep, do_wait, do_fin, has, widx = AWA.decide(self, B, end, valid, fl, rbad)
        else:
            has, widx = self._vote(self.R_ok[:B], self.R_bad[:B], self.R_lg[:B], self.R_sym[:B])
        if VF:      # answer-group voter replaces the final pick (A/S/B decisions above stay on the base voter's registers)
            self.s_pk0[:B] = torch.where(do_fin, torch.where(has, widx, torch.full_like(widx, -1)), self.s_pk0[:B])
            if SN:
                has, widx = ASN.vf_vote_sn(self.R_ok[:B], self.R_bad[:B], self.R_lg[:B], self.R_sn[:B], self.R_sym[:B], self.R_nn[:B],
                                           self.R_qc[:B], self.R_nt[:B], self.s_a[:B] + 1, self.k_san, self.sn_th, self.sn_c0)
            else:
                has, widx = AVF.vf_vote(self.R_ok[:B], self.R_bad[:B], self.R_lg[:B], self.R_sym[:B], self.R_nn[:B], self.R_qc[:B],
                                        self.R_nt[:B], self.s_a[:B] + 1, self.k_san, self.vf_th, self.vf_c0)
        self.s_win[:B] = torch.where(do_fin, torch.where(has, widx, torch.full_like(widx, -1)), self.s_win[:B])
        # ---- writer pointer: advance, or start a bank
        p = self.s_p[:B]
        p_next = self.t_SH[p].argmax(-1)
        # digit rows: pad symbol -> eos row
        wsl = self.s_win[:B].clamp(min=0)
        dk = (p_next - self.r_dig0).clamp(0, KS - 1)
        dsym = self.R_sym[bi, wsl, dk]
        indig = (p_next >= self.r_dig0) & (p_next < self.r_eos)
        p_next = torch.where(indig & (dsym == PAD), torch.full_like(p_next, self.r_eos), p_next)
        enl = torch.where(stopin, pe0, self.t_ENL[tok])         # a stop token is hidden: the phrase follows the text before it
        st = lambda i: torch.full_like(p, self.b_start[i])
        p_new = torch.where(do_rep, st(0), torch.where(do_wait, torch.where(enl, st(2), st(1)),
                            torch.where(do_fin, torch.where(has, torch.where(enl, st(4), st(3)), torch.full_like(p, self.r_eos)), p_next)))
        p_new = torch.where(live, p_new, torch.zeros_like(p_new))
        self.s_p[:B] = p_new
        # write vector of this step
        wrow = self.t_DROW[p_new]
        dig = (p_new >= self.r_dig0) & (p_new < self.r_eos)
        dsym2 = self.R_sym[bi, wsl, (p_new - self.r_dig0).clamp(0, KS - 1)]
        wrow = torch.where(dig[:, None], self.t_DSYM[dsym2], wrow)
        writing = p_new > 0
        write = wrow * self.gain * writing[:, None].float()
        self.s_fin[:B] = self.s_fin[:B] | (p_new == self.r_eos)
        self.s_wr[:B] = writing
        # ---- segment change: forget + register resets (repair re-uses the slot; retry moves to the next slot)
        newseg = do_rep | do_wait
        self.s_nf[:B] = self.s_nf[:B] + newseg.long()
        self.s_fr[:B] = torch.where(newseg, self.slot[:B] + 1, self.s_fr[:B])
        self.s_a[:B] = self.s_a[:B] + do_wait.long()
        self.s_rep[:B] = torch.where(do_wait, torch.zeros_like(self.s_rep[:B]), self.s_rep[:B] + do_rep.long())
        self.s_nrep[:B] = self.s_nrep[:B] + do_rep.long()
        a2 = self.s_a[:B].clamp(max=NS - 1)
        clr = F.one_hot(a2, NS).bool() & newseg[:, None]
        self.R_sym[:B] = torch.where(clr[:, :, None], torch.full_like(self.R_sym[:B], PAD), self.R_sym[:B])
        self.s_dfa[:B] = torch.where(newseg | do_fin, torch.zeros_like(dfa1), dfa1)
        for t_ in (self.s_c, self.s_k): t_[:B] = torch.where(newseg, torch.zeros_like(t_[:B]), t_[:B])
        self.s_lgset[:B] = self.s_lgset[:B] & ~newseg
        if VF:
            self.vf_b.reset(bi, newseg); self.s_vnt[:B] = torch.where(newseg, torch.zeros_like(self.s_vnt[:B]), self.s_vnt[:B])
        if SN:
            for t_ in (self.s_snH, self.s_snM, self.s_sncnt): t_[:B] = torch.where(newseg, torch.zeros_like(t_[:B]), t_[:B])
            self.s_snhs[:B] = self.s_snhs[:B] & ~newseg; self._sn_ns = newseg      # a '####' at an END step belongs to the ended segment
        bits = (writing.to(torch.int8) * 2) + ((do_rep | do_wait | do_fin) & live).to(torch.int8) * 8
        return write, bits, cap_probe

    def _step_triton(self, B, S):
        HK = EM.HK
        W, tok = self.W, self.tokb[:B]
        slot = self.slot[:B]
        imp = self.imp if self.imp_on else None
        emb = W['E'][tok]
        write = wf = None
        side = None
        if imp is not None:
            if OVL:          # the implant step reads only the input token: run it on a side stream next to the host layers
                main = torch.cuda.current_stream()
                side = self.__dict__.get('_ovl_st')
                if side is None: side = self._ovl_st = torch.cuda.Stream()
                side.wait_stream(main)
                ctx = torch.cuda.stream(side)
            else:
                import contextlib; ctx = contextlib.nullcontext()
            with ctx:
                e = imp.step(emb, B, tok) * self.row_on[:B, None]
                if self.use_write:
                    write = (e @ self.Dp) if self.Dp is not None else (imp.last_w if imp.fk is not None else e @ imp.D)
                wf = (e.abs().sum(-1) > 0).to(torch.int8)
        sw, bits, cap_probe = self._sw(B, tok)
        x, h, q, att = self.xb[:B], self.hb[:B], self.qb[:B], self.ab[:B]
        x.copy_(emb)
        if self.sw_posreset:     # programmatic reference only: positions of a new segment continue from the question end
            off = torch.where(self.s_fr[:B] > 0, self.s_fr[:B] - self.s_qlen[:B], torch.zeros_like(slot))
            pos = self.Lp + slot - off
        else:
            pos = self.Lp + slot
        wpr = WPR and not self.sw_posreset
        if wpr:      # weight-space rotary re-phasing (lane wpr): offset register -> fixed rotation table row R(-off)
            off = torch.where(self.s_fr[:B] > 0, self.s_fr[:B] - self.s_qlen[:B], torch.zeros_like(slot))
            rc, rs = self._wpr_tab(off)
            rar = self._wpr_ar[0, :B]
            if WPR_K == 1:      # composed rotation row R(pos - off) = R(-off) R(pos), fp32, one cast to the table dtype
                hc, hs = self._wpr_hc[pos], self._wpr_hs[pos]; oc, os_ = rc[:, 0], rs[:, 0]
                k_cos = (hc * oc + hs * os_).to(self.cos_t.dtype).contiguous()
                k_sin = (hs * oc - hc * os_).to(self.sin_t.dtype).contiguous()
            elif WPR_K == 2:
                k_cos, k_sin = self.cos_t[pos - off].contiguous(), self.sin_t[pos - off].contiguous()
        ieee = x.dtype == torch.float32
        sc = 1.0 / (Q.HD ** 0.5)
        delta = None
        if SN: snu = self._sn_u[:B]; snu.zero_()
        for i, lw in enumerate(W['L']):
            if i == self.v_L:
                if delta is not None: x += delta; delta = None
                vl = x.float() @ self.v_w + self.v_b
                self.s_lg[:B] = torch.where(cap_probe, vl, self.s_lg[:B]); self.s_lgset[:B] = self.s_lgset[:B] | cap_probe
            HK.add_rms_kernel[(B,)](x, delta if delta is not None else x, lw['ln1'], h, D=Q.D, HAS_DELTA=delta is not None, eps=Q.EPS)
            if SN and i in self.sn_W: snu += x.float() @ self.sn_W[i]       # x = layer-i input: fixed (D, 2) projection [H, M]
            if DOLA and i == DOLA_L:        # premature residual = layer-l input, through the host's final norm
                HK.add_rms_kernel[(B,)](x, x, W['lnf'], self._dola_h[B:2 * B], D=Q.D, HAS_DELTA=False, eps=Q.EPS)
            qkv = F.linear(h, lw['Wqkv'])
            Kc, Vc = self.Kc[i], self.Vc[i]
            if wpr and WPR_K:   # the kernel reads row r of the per-row composed table (pos index = r)
                HK.qk_rope_kernel[(B,)](qkv, lw['qn'], lw['kn'], k_cos, k_sin, rar, q, Kc, Vc, slot,
                                        Kc.stride(0), Kc.stride(1), Kc.stride(2), H=Q.H, KVH=Q.KVH, HD=Q.HD, eps=Q.EPS)
            else:
                HK.qk_rope_kernel[(B,)](qkv, lw['qn'], lw['kn'], self.cos_t, self.sin_t, pos, q, Kc, Vc, slot,
                                        Kc.stride(0), Kc.stride(1), Kc.stride(2), H=Q.H, KVH=Q.KVH, HD=Q.HD, eps=Q.EPS)
            if wpr and not WPR_K:
                q.copy_(_wpr_rot(q, rc, rs))
                Kc[rar, :, slot] = _wpr_rot(Kc[rar, :, slot], rc, rs)
            rbk = 8 if ieee else 32; ns = 1 if ieee else 2
            if ATT3 and not ieee and getattr(type(self), 'f_att3', False):
                attn3.attention(self, i, B, q, Kc, Vc, slot, att, sc)
            else:
                HK.attn_prefix_kernel[(Q.KVH, triton.cdiv(B, rbk))](q, self.Kp[i], self.Vp[i], self.mp, self.lpf, self.accp, B,
                                                                    self.Lp, sc, H=Q.H, HD=Q.HD, RBK=rbk, BN=64 if not ieee else 32,
                                                                    IEEE=ieee, num_warps=4, num_stages=ns)
                self._attn_suffix(i, B, q, Kc, Vc, slot, att, sc, ieee)
            o = F.linear(att.view(B, Q.H * Q.HD), lw['Wo'])
            HK.add_rms_kernel[(B,)](x, o, lw['ln2'], h, D=Q.D, HAS_DELTA=True, eps=Q.EPS)
            gu = F.linear(h, lw['Wgu'])
            act = self.actb[:B]
            HK.silu_mul_kernel[(B, Q.FF // 1024)](gu, act, FF=Q.FF, BLK=1024)
            delta = F.linear(act, lw['Wd'])
        x += delta
        if SN:      # '####' token: set the H projection; gated line-end token: add the M projection, count
            cp = cap_probe & ~self._sn_ns
            self.s_snH[:B] = torch.where(cp, snu[:, 0], self.s_snH[:B]); self.s_snhs[:B] = self.s_snhs[:B] | cp
            ml = self.s_snml[:B]
            self.s_snM[:B] = self.s_snM[:B] + torch.where(ml, snu[:, 1], torch.zeros_like(snu[:, 1]))
            self.s_sncnt[:B] = self.s_sncnt[:B] + ml.long()
        if side is not None: torch.cuda.current_stream().wait_stream(side)
        if write is not None: x += write.to(x.dtype)
        x += sw.to(x.dtype)
        HK.add_rms_kernel[(B,)](x, x, W['lnf'], h, D=Q.D, HAS_DELTA=False, eps=Q.EPS)
        if DOLA:    # one lm_head pass over [n(h_fin); n(h_l)]; z' on rows without a write (wf == 0), z elsewhere
            hh = self._dola_h[:2 * B]; hh[:B].copy_(h)
            zz = F.linear(hh, W['E'])
            on = ((bits if wf is None else (wf | bits)) == 0)
            logits = ADL.dola_logits(zz[:B], zz[B:], on, DOLA_A, DOLA_B)
            self._dola_plain = zz[:B]          # calc v2's ALT argmax keeps reading the host logits
        else:
            logits = F.linear(h, W['E'])
        nxt, lp = self.nxt_b[:B], self.lp_b[:B]
        HK.sample_kernel[(B,)](logits, self.temp, self.greedy, self.seed, nxt, lp, logits.shape[1], BV=2048, num_warps=8)
        if DOLA: self._dola_plain = None
        self.seed += 1
        wf = bits if wf is None else (wf | bits)
        g0 = self.glen[:B].clone(); d0 = self.done[:B].clone()
        self._book(B, tok, nxt, lp, wf, imp if imp is not None else True, None, None)
        # generator stop rule: the implant wrote eos (its pointer is on the eos row), or the token budget
        self.done[:B] = d0 | (self.s_p[:B] == self.r_eos) | ((g0 + 2) >= self.maxnew)

    # ---------------------------------------------------------------- helpers for the RL hook
    def sw_is_final(self, t, w, s_):
        return all(x & 2 for x in w[s_:])

    @staticmethod
    def _val(sym_row):
        s = ''.join('0123456789-.'[x] for x in sym_row if x != PAD)
        try: return float(s)
        except ValueError: return None

    def sw_vote(self, sym, ok, bad, lg, k, vmode=1, san=True):
        """the voter circuit's decision recomputed from the logged registers for the first k slots (analysis only)."""
        res = []
        for j in range(sym.shape[0]):
            idx = [i for i in range(k) if ok[j, i]]
            if san:
                s = [i for i in idx if not bad[j, i]]
                if s: idx = s
            if not idx: res.append(None); continue
            w = {i: (float(torch.sigmoid(lg[j, i])) if vmode == 1 else 1.0) for i in idx}
            sc = {i: sum(w[t] for t in idx if torch.equal(sym[j, t], sym[j, i])) for i in idx}
            best = max(idx, key=lambda i: (round(sc[i], 6), -i))
            res.append(best)
        return res

    def sw_roll_stats(self, out):
        R = out['reg']
        return dict(natt=float(R['natt'].float().mean()), repairs=float(R['nrep'].float().mean()),
                    win_none=float((R['win'] < 0).float().mean()))

    def sw_dev_eval(self, qd, items, rollout, C, tok, seed=4242, N=None):
        import re
        N = N or self.sw_nmax
        self.set_sw(N=N)
        o = rollout(qd, seed=seed)
        self.set_sw(N=int(os.environ.get('RL_N', 4)))
        R = o['reg']; n = len(items)
        gold = [float(r['gold']) for r in items]
        vals = [[self._val(R['sym'][j, i].tolist()) if R['ok'][j, i] else None for i in range(N)] for j in range(n)]
        right = [[v is not None and abs(v - g) <= 1e-6 * max(1, abs(g)) for v in vs] for vs, g in zip(vals, gold)]
        out = {}
        txts = [tok.decode(t).replace('<|endoftext|>', '') for t in o['tokens']]
        lastok = []
        for t, g in zip(txts, items):
            ms = list(re.finditer(C.NUM, t)); la = ms[-1].group(1).replace(',', '') if ms else None
            lastok.append(float(C.correct(la, g['gold'])))
        out[f'vote{N}_stream'] = 100 * sum(lastok) / n
        for k in sorted({1, 4, N}):
            pk = self.sw_vote(R['sym'], R['ok'], R['bad'], R['lg'], k)
            out[f'vote{k}'] = 100 * sum(1 for j, p in enumerate(pk) if p is not None and right[j][p]) / n
            out[f'oracle{k}'] = 100 * sum(any(right[j][:k]) for j in range(n)) / n
            dist = [len({vals[j][i] for i in range(k) if vals[j][i] is not None}) for j in range(n)]
            out[f'allsame{k}'] = 100 * sum(d <= 1 for d in dist) / n
        out['attempt_acc'] = [round(100 * sum(right[j][i] for j in range(n)) / n, 2) for i in range(N)]
        out['tok_per_attempt'] = sum(map(len, o['tokens'])) / n / N
        out['repairs_per_q'] = float(R['nrep'].float().mean())
        if os.environ.get('SW_DEVDUMP'):                           # debug: per-row registers of the dev eval
            torch.save({'reg': {k: v.cpu() for k, v in R.items()}, 'tokens': o['tokens'], 'wflag': o['wflag'], 'N': N},
                       f"{os.environ['SW_DEVDUMP']}_{len(items)}.pt")
        return out


import forget_ws as _fw
SerialWD = _fw.wrap_stack(SerialW)
