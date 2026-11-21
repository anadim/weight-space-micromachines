"""Answer-sanity implant for the RL engine (engine.py): an exact token-level circuit that reads the final-answer tokens
and acts when the committed answer is impossible for GSM8K (negative, non-integer, or missing).

Everything runs inside the decode CUDA graph as fixed-table tensor ops (no Python state on data):

  reader   one transition table TT (V x S) over a 34-state character DFA, lifted to tokens offline:
           dfa_t = TT[tok_t, dfa_{t-1}]. The DFA is the scorer's own regex  `####\\s*(-?[\\d,]*\\.?\\d+)`  (first match)
           run left to right: hash counter -> whitespace -> sign -> integer digits/commas -> '.' -> fraction digits ->
           absorbing DONE(cls). It tracks "any nonzero integer digit" and "any nonzero fraction digit", so it knows the
           exact sign / integrality of the value the scorer will read.
  class    FLAG[dfa]  0 ok (non-negative integer), 1 negative, 2 non-integer, 3 missing (no parsable '#### number')
  trigger  the host's next token (after sampling) would END the answer (the engine's stop rule: '<|endoftext|>',
           any token containing '\\n\\n', or '\\n' then a token starting with '\\n') AND FLAG != 0 AND fires < MAXF AND room.
  act      variant 'phrase' (a): replace the end token by a fixed token sequence  [stem of the end token] '\\n' PHRASE
           (no '\\n' when the text already ends in '\\n'; never a blank line), written over the next steps from a
           queue; the DFA restarts, so the answer that counts is the first '#### number' after the phrase.
           variant 'veto' (b): the end token is vetoed; the host's next token is re-chosen with every end token masked
           (greedy: masked argmax; sampling: masked Gumbel-max), the DFA restarts -> the flagged answer is no longer
           final and the model keeps writing.
  silent   when it does not fire, nxt / lp are the host's own values and nothing is written to the residual:
           generation is bit-identical to the engine without the implant (checked in san_eval.py).

This is the decode-time version (it edits the sampled token, the same effect as an infinite-gain residual write onto
the written token's LM-head direction). TODO: compile it as a residual-write circuit in rethink_circuit_v2 style
(token tables -> DFA via a hard-attention scan, writer = make_out_dirs through the host's norm + head) -- the DFA and the
trigger are already exact fixed tables, only the writer would change from 'replace nxt' to 'D[nxt'] * gain'.

out_w bits per token: 1 = calculator write (unchanged), 2 = sanity-written token, 4 = token chosen under the veto,
8 = the implant fired at this step.
"""
import os, unicodedata, torch, torch.nn.functional as F
import fastqwen as Q
import hostkern as HK
import triton
from engine import Engine, B_BUCKETS, S_STEP

PHRASE = "Hmm, wait. Let me rethink this.\n"
TYPED = {1: "Hmm, wait. The answer can't be negative. Let me rethink this.\n",
         2: "Hmm, wait. The answer should be a whole number. Let me rethink this.\n",
         3: "Hmm, wait. Let me rethink this.\n"}
TYPED_MISSING_HASH = "####"         # 'typedh' variant: missing answer -> write '####' so the host commits a number


# ----------------------------------------------------------------------------------------------- char DFA (exact regex)
def _states():
    S = [('P', k) for k in range(4)] + [('W', w) for w in (0, 1)] + [('M',)]
    for kind in ('I', 'D'):
        S += [(kind, n, z, h) for n in (0, 1) for z in (0, 1) for h in (0, 1)]
    S += [('F', n, z, f) for n in (0, 1) for z in (0, 1) for f in (0, 1)]
    S += [('X', c) for c in (0, 1, 2)]
    return S


def _cls(neg, nz, fz):
    if neg and (nz or fz): return 1
    if fz: return 2
    return 0


def _isdig(c):
    return c.isdecimal()          # python re '\d' on str == Unicode decimal digits


def _dval(c):
    try: return unicodedata.decimal(c)
    except Exception: return 1


def _reset(c):
    return ('P', 1) if c == '#' else ('P', 0)


def char_step(s, c):
    k = s[0]
    if k == 'X': return s
    if k == 'P':
        if c == '#': return ('W', 0) if s[1] + 1 >= 4 else ('P', s[1] + 1)
        return ('P', 0)
    if k == 'W':
        if c.isspace(): return ('W', 1)
        if c == '-': return ('M',)
        if _isdig(c): return ('I', 0, int(_dval(c) != 0), 1)
        if c == ',': return ('I', 0, 0, 0)
        if c == '.': return ('D', 0, 0, 0)
        if c == '#': return ('W', 0) if s[1] == 0 else ('P', 1)
        return ('P', 0)
    if k == 'M':
        if _isdig(c): return ('I', 1, int(_dval(c) != 0), 1)
        if c == ',': return ('I', 1, 0, 0)
        if c == '.': return ('D', 1, 0, 0)
        return _reset(c)
    if k == 'I':
        _, n, z, h = s
        if _isdig(c): return ('I', n, int(z or _dval(c) != 0), 1)
        if c == ',': return ('I', n, z, h)
        if c == '.': return ('D', n, z, h)
        return ('X', _cls(n, z, 0)) if h else _reset(c)
    if k == 'D':
        _, n, z, h = s
        if _isdig(c): return ('F', n, z, int(_dval(c) != 0))
        return ('X', _cls(n, z, 0)) if h else _reset(c)
    if k == 'F':
        _, n, z, f = s
        if _isdig(c): return ('F', n, z, int(f or _dval(c) != 0))
        return ('X', _cls(n, z, f))
    raise ValueError(s)


def flag_of(s):
    """class of the answer if the text ended in state s: 0 ok, 1 negative, 2 non-integer, 3 missing."""
    k = s[0]
    if k == 'X': return s[1]
    if k == 'I': return _cls(s[1], s[2], 0) if s[3] else 3
    if k == 'D': return _cls(s[1], s[2], 0) if s[3] else 3
    if k == 'F': return _cls(s[1], s[2], s[3])
    return 3


def build_tables(txt, cache=None):
    S = _states(); ix = {s: i for i, s in enumerate(S)}
    if cache and os.path.exists(cache):
        d = torch.load(cache)
        if d['V'] == len(txt): return d['TT'], d['FLAG'], S
    V = len(txt); TT = torch.empty(V, len(S), dtype=torch.long)
    for v, t in enumerate(txt):
        for si, s in enumerate(S):
            for c in t:
                s = char_step(s, c)
                if s[0] == 'X': break
            TT[v, si] = ix[s]
    FLAG = torch.tensor([flag_of(s) for s in S], dtype=torch.long)
    if cache: torch.save({'V': V, 'TT': TT, 'FLAG': FLAG}, cache)
    return TT, FLAG, S


def py_flag(text):
    """reference: the scorer's view of a text (one block, before any blank line)."""
    import re
    m = re.search(r'####\s*(-?[\d,]*\.?\d+)', text)
    if not m: return 3
    v = float(m.group(1).replace(',', ''))
    if v < 0: return 1
    if v != int(v): return 2
    return 0


# ----------------------------------------------------------------------------------------------------------- engine
class SanEngine(Engine):
    """Engine + answer-sanity implant. san_setup() once; switch variant with set_san(mode, phrase_set) (no re-capture)."""

    def san_setup(self, maxf=2, room=24, cache=None, cap0=None, extra=0):
        # cap0: the host's own token budget (stop at cap0 tokens while the implant has not fired, as the plain engine
        # does at max_new); each fire grants `extra` more tokens. Call generate() with max_new = cap0 + maxf * extra.
        self.s_cap0 = cap0; self.s_extra = int(extra)
        tok, dev = self.tok, self.dev
        V = self.W['E'].shape[0]
        txt = [tok.decode([i]) for i in range(len(tok))] + [''] * max(0, V - len(tok))
        TT, FLAG, S = build_tables(txt, cache)
        cm = os.environ.get('SAN_CLASSES')                 # e.g. '3' = act on missing answers only (GSM-Hard)
        if cm: FLAG = torch.where(torch.isin(FLAG, torch.tensor([int(c) for c in cm.split(',')])), FLAG, torch.zeros_like(FLAG))
        self.s_TT = TT.to(dev); self.s_FLAG = FLAG.to(dev); self.s_nS = len(S)
        T = lambda s: tok(s, add_special_tokens=False)['input_ids']
        seqs, sidx = [], {}
        def add(ids):
            k = tuple(ids)
            if k not in sidx: sidx[k] = len(seqs); seqs.append(list(ids))
            return sidx[k]
        # stem of every possible end token (text before its first blank line; '' for eos / '\n'-starting tokens)
        stems = {}
        for v, t in enumerate(txt):
            if v == self.eos: stems[v] = ''
            elif '\n\n' in t: stems[v] = t.split('\n\n')[0].rstrip()
            elif t.startswith('\n'): stems[v] = ''
        phrase_sets = {'generic': {1: PHRASE, 2: PHRASE, 3: PHRASE}, 'typed': dict(TYPED),
                       'typedh': {**TYPED, 3: TYPED_MISSING_HASH}}
        self.s_sets = list(phrase_sets)
        IDX = torch.zeros(len(phrase_sets), V, 2, 4, dtype=torch.long)
        uniq = sorted(set(stems.values()))
        for pi, (nm, ph) in enumerate(phrase_sets.items()):
            for st in uniq:
                vs = [v for v, s_ in stems.items() if s_ == st]
                for enl in (0, 1):
                    for c in (1, 2, 3):
                        if st: body = st + '\n' + ph[c]
                        else: body = ph[c] if enl else '\n' + ph[c]
                        r = add(T(body))
                        IDX[pi, vs, enl, c] = r
        L = max(map(len, seqs))
        SEQ = torch.full((len(seqs), L), self.eos, dtype=torch.long); SL = torch.zeros(len(seqs), dtype=torch.long)
        for r, s_ in enumerate(seqs): SEQ[r, :len(s_)] = torch.tensor(s_); SL[r] = len(s_)
        self.s_IDX, self.s_SEQ, self.s_SL, self.s_L = IDX.to(dev), SEQ.to(dev), SL.to(dev), L
        self.s_IDXf, self.s_SEQf, self.s_TTf, self.s_V = self.s_IDX.reshape(-1), self.s_SEQ.reshape(-1), self.s_TT.reshape(-1), V
        self.s_seqs_txt = [tok.decode(s_) for s_ in seqs]
        Bp = self.B_max + 1
        z = lambda dt=torch.long: torch.zeros(Bp, device=dev, dtype=dt)
        self.s_dfa, self.s_nf, self.s_fr, self.s_fp = z(), z(), z(), z()
        self.s_fr.fill_(-1)
        self.s_mode = torch.zeros((), device=dev, dtype=torch.long)      # 0 off, 1 phrase, 2 veto, 3 observe (fire bit only)
        self.s_set = torch.zeros((), device=dev, dtype=torch.long)
        self.s_samp = torch.zeros((), device=dev, dtype=torch.long)
        self.s_maxf = torch.tensor(maxf, device=dev); self.s_room = int(room)
        self.vocab_eos = torch.zeros(V, device=dev, dtype=torch.bool); self.vocab_eos[self.eos] = True
        self.s_z0 = torch.zeros(Bp, V, device=dev)                       # restart: first-answer-token logits per row
        self.s_qlen = z()                                                # restart: question length (slot of the 1st answer token)
        self.s_rst = torch.zeros(Bp, device=dev, dtype=torch.bool)
        self.graphs = {}; self.pool = None; self._outshape = None
        return self

    def set_san(self, mode='off', phrase_set='generic', maxf=None, samp=0):
        self.s_samp.fill_(int(samp))
        self.s_mode.fill_({'off': 0, 'phrase': 1, 'veto': 2, 'observe': 3, 'restart': 4}[mode])
        self.s_set.fill_(self.s_sets.index(phrase_set))
        if maxf is not None: self.s_maxf.fill_(maxf)

    # -- the circuit (one decode step, rows :B) --
    def _sanity(self, B, tok, nxt, lp, wf, logits):
        dfa = self.s_TTf[tok * self.s_nS + self.s_dfa[:B]]
        cls = self.s_FLAG[dfa]
        nf, fr, fp = self.s_nf[:B], self.s_fr[:B], self.s_fp[:B]
        done = self.done[:B]
        on = self.s_mode > 0
        active = (fr >= 0) & on
        frc = fr.clamp(min=0)
        ftok = self.s_SEQf[frc * self.s_L + fp.clamp(max=self.s_L - 1)]
        enl = self.ends_nl[tok]
        stopping = self.vocab_eos[nxt] | self.has_nn[nxt] | (enl & self.starts_nl[nxt])
        room = (self.glen[:B] + 2 + self.s_room) < (self.maxnew if self.s_cap0 is None else self.s_cap0)
        fire = on & ~active & ~done & stopping & (cls > 0) & (nf < self.s_maxf) & room
        # (a) phrase: first token of the sequence for (end token stem, ends-in-newline, class)
        r = self.s_IDXf[((self.s_set * self.s_V + nxt) * 2 + enl.long()) * 4 + cls.clamp(min=1)]
        ph0 = self.s_SEQf[r * self.s_L]
        # (b) veto: host's choice with every end token masked
        lg = logits.float()
        endm = self.has_nn[None] | self.vocab_eos[None] | (enl[:, None] & self.starts_nl[None])
        zm = (lg / self.temp).masked_fill(endm, float('-inf'))
        gum = -torch.log(-torch.log(torch.rand_like(zm).clamp_(1e-20, 1.0)))
        vt = torch.where(self.greedy, zm.argmax(-1), (zm + gum).argmax(-1))
        is_ph = self.s_mode == 1; is_vt = self.s_mode == 2; is_rs = self.s_mode == 4
        # (c) restart: the row starts a NEW answer from the question (slot rewound, first token drawn at T=1 from the
        #     stored first-token logits); the old attempt stays in the output stream but leaves the KV window
        z0 = self.s_z0[:B]
        rs_tok = (z0 + -torch.log(-torch.log(torch.rand_like(z0).clamp_(1e-20, 1.0)))).argmax(-1)
        new = torch.where(active, ftok, torch.where(fire & is_ph, ph0, torch.where(fire & is_vt, vt, torch.where(fire & is_rs, rs_tok, nxt))))
        self.s_rst[:B] = fire & is_rs
        # optional: after a fire the row's own tokens are SAMPLED at T=1 (a fresh draw instead of the greedy re-derivation)
        samp = (self.s_samp > 0) & (nf > 0) & ~active & ~fire & ~done
        new = torch.where(samp, (lg + gum).argmax(-1), new)
        changed = active | (fire & (is_ph | is_vt | is_rs)) | samp
        lsm = torch.log_softmax(lg / self.temp, -1).gather(-1, new[:, None])[:, 0]
        lp2 = torch.where(changed, lsm, lp)
        # state updates
        fp_a = fp + 1
        fr_new = torch.where(active, torch.where(fp_a >= self.s_SL[frc], torch.full_like(fr, -1), fr), fr)
        fp_new = torch.where(active, fp_a, fp)
        start_q = fire & is_ph & (self.s_SL[r] > 1)
        fr_new = torch.where(start_q, r, fr_new); fp_new = torch.where(fire & is_ph, torch.ones_like(fp), fp_new)
        self.s_fr[:B] = fr_new; self.s_fp[:B] = fp_new
        self.s_nf[:B] = nf + fire.long()
        self.s_dfa[:B] = torch.where(fire, torch.zeros_like(dfa), dfa)
        bits = (torch.where(active | (fire & is_ph), 2, torch.where(fire & is_vt, 4, 0)) + 8 * fire.long() + 16 * (fire & is_rs).long()).to(torch.int8)
        wf2 = bits if wf is None else (wf | bits)
        return new, lp2, wf2

    def _step_triton(self, B, S):
        W, tok = self.W, self.tokb[:B]
        slot = self.slot[:B]
        imp = self.imp if self.imp_on else None
        links = self.links if self.imp_on else None
        emb = W['E'][tok]
        feat = write = wf = None; dec = lpd = None
        if imp is not None:
            e = imp.step(emb, B, tok) * self.row_on[:B, None]
            if self.inwire is not None: e, dec, lpd = self._inwire_step(e, B, tok)
            feat = e
            if self.use_write:
                if self.inwire is not None:
                    write = e @ (self.Dp if self.Dp is not None else imp.D.float())
                else:
                    write = (e @ self.Dp) if self.Dp is not None else (imp.last_w if imp.fk is not None else e @ imp.D)
            wf = (e.abs().sum(-1) > 0).to(torch.int8)
        x, h, q, att = self.xb[:B], self.hb[:B], self.qb[:B], self.ab[:B]
        x.copy_(emb)
        pos = self.Lp + slot
        ieee = x.dtype == torch.float32
        sc = 1.0 / (Q.HD ** 0.5)
        delta = None
        iwL = self.inwire.L if (self.inwire is not None and imp is not None) else -1
        for i, lw in enumerate(W['L']):
            if i == iwL:
                if delta is not None: x += delta; delta = None
                self.iw_h[:B].copy_(x)
            if links is not None and i in links:
                if delta is not None: x += delta; delta = None
                x += links[i](feat, x).to(x.dtype)
            HK.add_rms_kernel[(B,)](x, delta if delta is not None else x, lw['ln1'], h, D=Q.D, HAS_DELTA=delta is not None, eps=Q.EPS)
            qkv = F.linear(h, lw['Wqkv'])
            Kc, Vc = self.Kc[i], self.Vc[i]
            HK.qk_rope_kernel[(B,)](qkv, lw['qn'], lw['kn'], self.cos_t, self.sin_t, pos, q, Kc, Vc, slot,
                                    Kc.stride(0), Kc.stride(1), Kc.stride(2), H=Q.H, KVH=Q.KVH, HD=Q.HD, eps=Q.EPS)
            rbk = 8 if ieee else 32; ns = 1 if ieee else 2
            HK.attn_prefix_kernel[(Q.KVH, triton.cdiv(B, rbk))](q, self.Kp[i], self.Vp[i], self.mp, self.lpf, self.accp, B,
                                                                self.Lp, sc, H=Q.H, HD=Q.HD, RBK=rbk, BN=64 if not ieee else 32,
                                                                IEEE=ieee, num_warps=4, num_stages=ns)
            HK.attn_suffix_kernel[(B, Q.KVH)](q, Kc, Vc, slot, self.mp, self.lpf, self.accp, att,
                                              Kc.stride(0), Kc.stride(1), Kc.stride(2), sc, H=Q.H, HD=Q.HD, BN=64 if not ieee else 32,
                                              IEEE=ieee, num_warps=4, num_stages=1 if ieee else 2)
            o = F.linear(att.view(B, Q.H * Q.HD), lw['Wo'])
            HK.add_rms_kernel[(B,)](x, o, lw['ln2'], h, D=Q.D, HAS_DELTA=True, eps=Q.EPS)
            gu = F.linear(h, lw['Wgu'])
            act = self.actb[:B]
            HK.silu_mul_kernel[(B, Q.FF // 1024)](gu, act, FF=Q.FF, BLK=1024)
            delta = F.linear(act, lw['Wd'])
        x += delta
        if links is not None and 'final' in links:
            x += links['final'](feat, x).to(x.dtype)
        if write is not None:
            x += write.to(x.dtype)
        HK.add_rms_kernel[(B,)](x, x, W['lnf'], h, D=Q.D, HAS_DELTA=False, eps=Q.EPS)
        logits = F.linear(h, W['E'])
        nxt, lp = self.nxt_b[:B], self.lp_b[:B]
        HK.sample_kernel[(B,)](logits, self.temp, self.greedy, self.seed, nxt, lp, logits.shape[1], BV=2048, num_warps=8)
        self.seed += 1
        if lpd is not None: lp = lp + lpd
        nxt, lp, wf = self._sanity(B, tok, nxt, lp, wf, logits)
        g0 = self.glen[:B].clone() if self.s_cap0 is not None else None
        self._book(B, tok, nxt, lp, wf, imp if imp is not None else True, dec, self._ops_now if dec is not None else None)
        if g0 is not None:                     # per-row budget: cap0 + extra * fires (== the plain engine's cap when silent)
            self.done[:B] |= (g0 + 2) >= (self.s_cap0 + self.s_extra * self.s_nf[:B])
        self.slot[:B] = torch.where(self.s_rst[:B], self.s_qlen[:B], self.slot[:B])     # restart: rewind the KV window

    @torch.no_grad()
    def generate(self, qids, G=1, T=1.0, max_new=320, greedy=False):
        """Engine.generate without the S_max assert in restart mode (a restarted row rewinds its KV slot, so the slot
        never exceeds question + one attempt even though the output stream holds several attempts)."""
        if int(self.s_mode) != 4: return super().generate(qids, G, T, max_new, greedy)
        assert max(map(len, qids)) + (self.s_cap0 or max_new) + self.s_extra + 32 <= self.S_max
        return self._gen_noassert(qids, G, T, max_new, greedy)

    def _gen_noassert(self, qids, G, T, max_new, greedy):
        import time
        self._alloc_out(max_new)
        ppc = max(1, self.B_max // G)
        res_tok, res_lp, res_w, res_on = [], [], [], []
        torch.cuda.synchronize(); t0 = time.time(); steps = 0
        for p0 in range(0, len(qids), ppc):
            qs = qids[p0:p0 + ppc]; n = len(qs) * G
            steps += self._wave(qs, G, T, max_new, greedy)
            res_on += self._wave_on.tolist()
            ot, ol, ow, gl = self.out_tok[:n].cpu(), self.out_lp[:n].cpu(), self.out_w[:n].cpu(), self._glen_final.cpu()
            for j in range(n):
                L = int(gl[j]) + 1
                res_tok.append(ot[j, :L].tolist()); res_lp.append(ol[j, :L].tolist()); res_w.append(ow[j, :L].tolist())
        torch.cuda.synchronize(); dt = time.time() - t0
        ntok = sum(map(len, res_tok))
        self.stats = dict(seconds=dt, tokens=ntok, tok_s=ntok / dt, decode_steps=steps, rows=len(qids) * G)
        return dict(tokens=res_tok, logp=res_lp, wflag=res_w, dec=[[] for _ in res_tok], ops=[[] for _ in res_tok], iw=[], on=res_on, stats=self.stats)

    def _mut(self):
        return super()._mut() + [self.s_dfa, self.s_nf, self.s_fr, self.s_fp, self.s_rst]

    def _wave(self, qs, G, T, max_new, greedy):
        n = len(qs) * G; dev = self.dev; DUMMY = self.B_max
        self.out_tok.fill_(self.eos); self.out_lp.zero_(); self.out_w.zero_(); self.out_dec.zero_(); self.out_ops.zero_(); self.out_iwl.zero_(); self.out_run.zero_(); self.out_span.zero_()
        rows = [torch.arange(p * G, (p + 1) * G, device=dev) for p in range(len(qs))]
        logits, lens, e_last = self._prefill(qs, rows)
        self.temp.fill_(T); self.greedy.fill_(greedy); self.maxnew.fill_(max_new)
        z = (logits / T).repeat_interleave(G, 0)
        first = z.argmax(-1) if greedy else (z - torch.log(-torch.log(torch.rand_like(z).clamp_(1e-20, 1.0)))).argmax(-1)
        self.out_tok[:n, 0] = first
        self.out_lp[:n, 0] = torch.log_softmax(z, -1).gather(-1, first[:, None])[:, 0]
        if e_last is not None: self.out_w[:n, 0] = (e_last.abs().sum(-1) > 0).to(torch.int8).repeat_interleave(G)
        del z
        self.s_z0[:n] = logits.repeat_interleave(G, 0); self.s_qlen[:n] = lens.repeat_interleave(G); self.s_rst.zero_()
        Bb = next(b for b in B_BUCKETS if b >= n)
        self.tokb[:n] = first; self.slot[:n] = lens.repeat_interleave(G); self.glen[:n] = 0
        self.rowid[:n] = torch.arange(n, device=dev)
        self.done[:n] = (first == self.eos) | self.has_nn[first] | (max_new <= 1)
        self.tokb[n:Bb] = self.eos; self.slot[n:Bb] = 0; self.glen[n:Bb] = 0; self.rowid[n:Bb] = DUMMY; self.done[n:Bb] = True
        self.s_dfa.zero_(); self.s_nf.zero_(); self.s_fr.fill_(-1); self.s_fp.zero_()
        self._wave_on = (torch.rand(n, device=dev) >= self.drop_p).float() if self.drop_p > 0 else torch.ones(n, device=dev)
        self.row_on.fill_(1.0); self.row_on[:n] = self._wave_on
        maxlen = int(lens.max()); live_n = n; step = 0
        glen_final = torch.zeros(n + 1, dtype=torch.long, device=dev)
        while True:
            Bb = next(b for b in B_BUCKETS if b >= live_n)
            Sb = self.S_max if self.kern == 'triton' else min(self.S_max, ((maxlen + step + self.sync_every) // S_STEP + 1) * S_STEP)
            g = self._graph(Bb, Sb)
            for _ in range(self.sync_every):
                g.replay()
            step += self.sync_every
            d = self.done[:live_n]
            n_live = int((~d).sum())
            if n_live == 0 or step >= max_new:
                break
            nb = next(b for b in B_BUCKETS if b >= n_live)
            if nb < Bb:
                idx = torch.nonzero(~d)[:, 0]; fin = torch.nonzero(d)[:, 0]
                glen_final[torch.clamp(self.rowid[fin], max=n)] = self.glen[fin]
                su = min(self.S_max, maxlen + step + 1)
                for i in range(Q.NL):
                    self.Kc[i][:n_live, :, :su] = self.Kc[i][idx, :, :su]
                    self.Vc[i][:n_live, :, :su] = self.Vc[i][idx, :, :su]
                for t in (self.tokb, self.slot, self.glen, self.rowid, self.done, self.row_on,
                          self.s_dfa, self.s_nf, self.s_fr, self.s_fp, self.s_z0, self.s_qlen):
                    t[:n_live] = t[idx].clone()
                if self.imp is not None and self.imp_on: self.imp.gather(idx, n_live)
                if self.inwire is not None:
                    for t in (self.iw_h, self.iw_prev, self.iw_sup, self.iw_run, self.iw_max, self.iw_ops, self.iw_brk,
                              self.iw_cop, self.iw_sop):
                        t[:n_live] = t[idx].clone()
                self.done[n_live:nb] = True; self.rowid[n_live:nb] = DUMMY
                live_n = n_live
        glen_final[torch.clamp(self.rowid[:live_n], max=n)] = self.glen[:live_n]
        self._glen_final = glen_final[:n]
        return step


# ------------------------------------------------------------------------------------------------- scoring with fires
def san_score(C, tok, ids, wflag, gold):
    """C.score on rows where the implant never acted (bit-identical scoring); on rows where it acted, the answer that
    counts is the first '#### number' AFTER the last sanity-written / vetoed token (text cut at the first blank line)."""
    import re
    last = max([i for i, w in enumerate(wflag) if w & 6], default=-1)
    last = max(last, max([i - 1 for i, w in enumerate(wflag) if w & 16], default=-1))
    if last < 0 and not any(w & 16 for w in wflag): return C.score(tok, ids, gold) + (0,)
    txt = tok.decode(ids).replace('<|endoftext|>', '').split('\n\n')[0]
    pre = tok.decode(ids[:last + 1]).replace('<|endoftext|>', '')
    tail = txt[len(pre):] if txt.startswith(pre) else txt[len(pre):]
    m = re.search(C.NUM, tail); pred = m.group(1).replace(',', '') if m else None
    return float(C.correct(pred, gold)), pred is not None, txt, sum(1 for w in wflag if w & 22)
