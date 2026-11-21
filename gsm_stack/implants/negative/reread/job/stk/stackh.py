"""StackH (from stack3_hook.py) as an importable module: CompEngine (compiled sanity DFA + pointer phrase writer) over
C2Engine's no-compaction loop, + the stop-token missing trigger (switch m_on), + weight-space forget (forget_ws)."""
import os as _o, torch, triton, torch.nn.functional as F
import engine as EM
import fastqwen as Q
from c2_engine import C2Engine
import san_engine as SE
import comp_sanity as CS
from comp_sanity import CompEngine
MISS_ONLY = _o.environ.get('STK_MISSING_ONLY', '0') == '1'
FORGET = 'weights'


class StackH(CompEngine, C2Engine):
    def stack_setup(self):
        TT, CLS, TC, CN, CF, S = self.c_tables
        self.m_FLAG = torch.tensor([SE.flag_of(s) for s in S], device=self.dev)
        V = self.W['E'].shape[0]
        self.m_eos = torch.zeros(V, device=self.dev, dtype=torch.bool); self.m_eos[self.eos] = True
        self.m_on = True
        if MISS_ONLY: self.c_CN = torch.zeros_like(self.c_CN); self.c_CF = torch.zeros_like(self.c_CF)
        self._miss_ptr()

    def set_phrases(self, phrase_set, gain=None):
        CompEngine.set_phrases(self, phrase_set, gain)
        if hasattr(self, 'm_FLAG'): self._miss_ptr()

    def _miss_ptr(self):            # bank 3 = the 'missing' bank: its first token and pointer index
        T = lambda s: self.tok(s, add_special_tokens=False)['input_ids']
        ph = CS.PHRASES[self.c_phrase]; b1, b2 = T(ph[1]), T(ph[2])
        self.m_st3 = 1 + len(b1) + len(b2); self.m_tok3 = int(self.c_ids[self.m_st3 - 1])

    def _prefill(self, qs, rows):
        out = EM.Engine._prefill(self, qs, rows)
        lens = out[1]; G = rows[0].numel(); n = len(qs) * G
        self.s_qlen[:n] = lens.repeat_interleave(G)
        return out

    def _wave(self, qs, G, T, max_new, greedy):
        self.s_dfa.zero_(); self.s_nf.zero_(); self.s_fr.fill_(-1); self.s_fp.zero_(); self.s_rst.zero_()
        return C2Engine._wave(self, qs, G, T, max_new, greedy)

    def _mut(self):
        return SE.SanEngine._mut(self)

    def _attn_suffix(self, i, B, q, Kc, Vc, slot, att, sc, ieee):
        if self.c_forget and FORGET == 'kernel':
            CS.attn_suffix_mask_kernel[(B, Q.KVH)](q, Kc, Vc, slot, self.s_qlen, self.s_fr, self.mp, self.lpf, self.accp, att,
                                                   Kc.stride(0), Kc.stride(1), Kc.stride(2), sc, H=Q.H, HD=Q.HD, BN=64 if not ieee else 32,
                                                   IEEE=ieee, num_warps=4, num_stages=1 if ieee else 2)
        else:
            EM.HK.attn_suffix_kernel[(B, Q.KVH)](q, Kc, Vc, slot, self.mp, self.lpf, self.accp, att,
                                                 Kc.stride(0), Kc.stride(1), Kc.stride(2), sc, H=Q.H, HD=Q.HD, BN=64 if not ieee else 32,
                                                 IEEE=ieee, num_warps=4, num_stages=1 if ieee else 2)

    def _step_triton(self, B, S):
        HK = EM.HK                      # calc v2's sampler proxy (ALT token) around hostkern
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
        cw, wbits, p2 = self._comp(B, tok)
        x, h, q, att = self.xb[:B], self.hb[:B], self.qb[:B], self.ab[:B]
        x.copy_(emb)
        pos = self.Lp + slot
        ieee = x.dtype == torch.float32
        sc = 1.0 / (Q.HD ** 0.5)
        delta = None
        for i, lw in enumerate(W['L']):
            HK.add_rms_kernel[(B,)](x, delta if delta is not None else x, lw['ln1'], h, D=Q.D, HAS_DELTA=delta is not None, eps=Q.EPS)
            qkv = F.linear(h, lw['Wqkv'])
            Kc, Vc = self.Kc[i], self.Vc[i]
            HK.qk_rope_kernel[(B,)](qkv, lw['qn'], lw['kn'], self.cos_t, self.sin_t, pos, q, Kc, Vc, slot,
                                    Kc.stride(0), Kc.stride(1), Kc.stride(2), H=Q.H, KVH=Q.KVH, HD=Q.HD, eps=Q.EPS)
            rbk = 8 if ieee else 32; ns = 1 if ieee else 2
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
        if write is not None: x += write.to(x.dtype)
        x += cw.to(x.dtype)
        HK.add_rms_kernel[(B,)](x, x, W['lnf'], h, D=Q.D, HAS_DELTA=False, eps=Q.EPS)
        logits = F.linear(h, W['E'])
        nxt, lp = self.nxt_b[:B], self.lp_b[:B]
        HK.sample_kernel[(B,)](logits, self.temp, self.greedy, self.seed, nxt, lp, logits.shape[1], BV=2048, num_warps=8)
        self.seed += 1
        wf = wbits if wf is None else (wf | wbits)
        if self.m_on:   # missing answer at the stop token: the same attempt registers as the compiled trigger
            fl = self.m_FLAG[self.s_dfa[:B]]
            stopping = self.m_eos[nxt] | self.has_nn[nxt] | (self.ends_nl[tok] & self.starts_nl[nxt])
            room = (self.glen[:B] + self.c_Lmax + 24) < (self.s_cap0 + self.s_extra * self.s_nf[:B])
            mf = (fl == 3) & stopping & (self.s_fp[:B] == 0) & (self.s_nf[:B] < self.c_maxf) & room & ~self.done[:B] & (self.c_on > 0)
            nxt.copy_(torch.where(mf, torch.full_like(nxt, self.m_tok3), nxt))
            self.s_fp[:B] = torch.where(mf, torch.full_like(self.s_fp[:B], self.m_st3), self.s_fp[:B])
            self.s_nf[:B] += mf.long()
            self.s_dfa[:B] = torch.where(mf, torch.zeros_like(self.s_dfa[:B]), self.s_dfa[:B])
            if self.c_forget: self.s_fr[:B] = torch.where(mf, slot + 1, self.s_fr[:B])
            wf = wf | (mf.to(torch.int8) * 10)
        g0 = self.glen[:B].clone()
        self._book(B, tok, nxt, lp, wf, imp if imp is not None else True, None, None)
        self.done[:B] |= (g0 + 2) >= (self.s_cap0 + self.s_extra * self.s_nf[:B])




import forget_ws as _fw
StackH = _fw.wrap_stack(StackH)


class StackHD(StackH):
    """+ DualImplant (calc_s4 + compiled chain2) states in the mutable set."""
    def _mut(self):
        m = super()._mut()
        im = self.imp
        if hasattr(im, 'chain'): m += [im.chain.sym_h, im.chain.S_h, im.chain.key_h, im.chain.fst]
        return m
