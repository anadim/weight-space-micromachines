"""Fast rollout engine: shared-prefix KV, static preallocated suffix KV cache, the WHOLE decode step (implant + host +
sampling + logprob + stop rule + bookkeeping) captured as CUDA graphs bucketed by (live rows, cache length), finished-row
compaction, Gumbel-max sampling fused into the step. Host does ONE graph replay per token and syncs every `sync_every`.

Usage:
    eng = Engine(W_bf16, prefix_ids, implant=CalcImplant|None, links=None, B_max=512, S_max=544)
    eng.refresh()                      # after a weight update (prefix KV recomputed in place; graphs stay valid)
    out = eng.generate(list_of_question_id_lists, G=8, T=1.0, max_new=320, greedy=False)
"""
import time, torch, triton, torch.nn.functional as F
import fastqwen as Q
import hostkern as HK

B_BUCKETS = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
S_STEP = 64


class Engine:
    def __init__(self, W, prefix_ids, tok, implant=None, links=None, B_max=512, S_max=544, use_write=True, sync_every=16,
                 kern='triton'):
        self.W, self.tok, self.imp, self.links = W, tok, implant, links
        if implant is not None and getattr(implant, 'tagf', False): implant.Eref = W['E']   # calc_tag filter feeds
        B_max = next(b for b in B_BUCKETS if b >= B_max)          # graphs run whole buckets
        self.use_write = use_write                    # False: implant runs (feature for links) but its designed write is off
        dev = W['E'].device; self.dev = dev
        self.prefix_ids = torch.tensor(prefix_ids, device=dev)
        self.Lp = len(prefix_ids)
        self.B_max, self.S_max, self.sync_every = B_max, S_max, sync_every
        bf = W['E'].dtype                              # bf16 for RL; fp32 for the exactness check
        self.cos_t, self.sin_t = Q.rope_tables(self.Lp + S_max + 8, dev, bf)
        self.Kc = [torch.zeros(B_max, Q.KVH, S_max, Q.HD, device=dev, dtype=bf) for _ in range(Q.NL)]
        self.Vc = [torch.zeros(B_max, Q.KVH, S_max, Q.HD, device=dev, dtype=bf) for _ in range(Q.NL)]
        self.Kp = torch.zeros(Q.NL, Q.KVH, self.Lp, Q.HD, device=dev, dtype=bf)
        self.Vp = torch.zeros_like(self.Kp)
        z = lambda *s, dt=torch.long: torch.zeros(*s, device=dev, dtype=dt)
        self.tokb, self.slot, self.glen, self.rowid = z(B_max), z(B_max), z(B_max), z(B_max)
        self.done = z(B_max, dt=torch.bool)
        self.temp = torch.ones((), device=dev); self.greedy = torch.zeros((), device=dev, dtype=torch.bool)
        self.maxnew = torch.zeros((), device=dev, dtype=torch.long)
        self.out_tok = self.out_lp = self.out_w = None
        # stop rule == scripts/run_gsm.py (blank line or <|endoftext|>), as per-token tables
        self.eos = tok.convert_tokens_to_ids('<|endoftext|>')
        V = W['E'].shape[0]
        txt = [tok.decode([i]) for i in range(len(tok))] + [''] * max(0, V - len(tok))
        self.has_nn = torch.tensor(['\n\n' in t for t in txt], device=dev)
        self.ends_nl = torch.tensor([t.endswith('\n') for t in txt], device=dev)
        self.starts_nl = torch.tensor([t.startswith('\n') for t in txt], device=dev)
        self.graphs, self.pool = {}, None
        self.kern = kern
        if kern == 'triton':
            Bp = B_max + 1
            self.xb = torch.zeros(Bp, Q.D, device=dev, dtype=bf); self.hb = torch.zeros_like(self.xb)
            self.qb = torch.zeros(Bp, Q.H, Q.HD, device=dev, dtype=bf); self.ab = torch.zeros_like(self.qb)
            self.mp = torch.zeros(Bp * Q.H, device=dev); self.lpf = torch.zeros_like(self.mp)
            self.accp = torch.zeros(Bp * Q.H, Q.HD, device=dev)
            self.actb = torch.zeros(Bp, Q.FF, device=dev, dtype=bf)
            self.seed = torch.zeros((), device=dev, dtype=torch.int64)
            self.nxt_b = torch.zeros(Bp, device=dev, dtype=torch.long); self.lp_b = torch.zeros(Bp, device=dev)
        self.imp_on = True                                  # False: host only (implant + links off), separate graphs
        self.Dp = None          # trainable-writer path (W1/W4): fp32 (n_out, d) buffer; when set the end write is e @ Dp
        self.inwire = None      # learned host->implant veto (CW arms); see set_inwire
        self.partial = None     # partially broken calculator (CWinP); see set_partial
        self.stats = {}
        self.refresh()

    def set_inwire(self, iw):
        """iw: module with .L (layer read), .k (float), .b (param ()), .v (param (D,)). Per write burst (first token whose
        hand implant output is nonzero after a silent token) fire ~ Bernoulli(sigmoid(k*0.5 + b + v.LN(h_L[t-1]))) in
        sampling mode, p > 0.5 in greedy mode; a vetoed burst writes nothing (e masked to 0 for the whole burst).
        The decision's log-prob is added to the logprob of the token produced at that step."""
        assert self.kern == 'triton'
        self.inwire = iw; dev = self.dev; Bp = self.B_max + 1
        self.iw_h = torch.zeros(Bp, Q.D, device=dev, dtype=self.W['E'].dtype)
        self.iw_prev = torch.zeros(Bp, device=dev, dtype=torch.bool)
        self.iw_sup = torch.zeros(Bp, device=dev, dtype=torch.bool)
        # operand-size tracker (per row): digits in the current number, longest number in the current arithmetic span
        # (reset by any word token), snapshot at '=' = the largest operand of the equation about to be computed
        V = self.W['E'].shape[0]
        txt = [self.tok.decode([i]) for i in range(len(self.tok))] + [''] * max(0, V - len(self.tok))
        st_ = [t.strip() for t in txt]
        self.t_ndig = torch.tensor([len(t) if t.isdigit() else 0 for t in st_], device=dev, dtype=torch.int32)
        self.t_eq = torch.tensor([t == '=' for t in st_], device=dev)
        self.t_expr = torch.tensor([all(c in '0123456789+-*/x\u00d7\u00f7$.,()=' for c in t) for t in st_], device=dev)
        z = lambda dt: torch.zeros(Bp, device=dev, dtype=dt)
        self.iw_run, self.iw_max, self.iw_ops = z(torch.int32), z(torch.int32), z(torch.int32)
        self.iw_brk = z(torch.bool)
        # operator tracker: last operator in the current arithmetic span (1 '+', 2 '-', 3 '*', 4 '/'), snapshot at '='
        opmap = {'+': 1, '-': 2, '*': 3, 'x': 3, '\u00d7': 3, '/': 4, '\u00f7': 4}
        self.t_op = torch.tensor([opmap.get(t, 0) for t in st_], device=dev, dtype=torch.int32)
        self.iw_cop, self.iw_sop = z(torch.int32), z(torch.int32)
        self.partial = None; self.partial_ops = None
        self.graphs = {}; self._outshape = None

    def set_partial_ops(self, broken_ops, D_der):
        """partially broken calculator by OPERATION: calls whose operator is in broken_ops (codes 1 '+', 2 '-', 3 '*',
        4 '/') write deranged digits; others exact. Decided at the burst start, recorded as +4 in the decision code."""
        self.partial_ops = torch.tensor(sorted(broken_ops), device=self.dev, dtype=torch.int32)
        self.D_der_minus_D = (D_der - self.imp.D).float(); self.graphs = {}

    def set_partial(self, thresh, D_der):
        """partially broken calculator: a call whose largest operand has >= thresh digits writes with D_der (deranged
        digits); otherwise exact. Decided once per burst, recorded in the decision code (+4)."""
        self.partial = thresh; self.D_der_minus_D = (D_der - self.imp.D).float(); self.graphs = {}
        self.partial_t = torch.tensor(int(thresh), device=self.dev, dtype=torch.int32)   # in-graph: switch in place

    def _inwire_step(self, e, B, tokcur):
        iw = self.inwire
        nd = self.t_ndig[tokcur]; ise = self.t_eq[tokcur]; isx = self.t_expr[tokcur]
        run = torch.where(nd > 0, self.iw_run[:B] + nd, torch.zeros_like(nd))
        mx = torch.where(isx, torch.maximum(self.iw_max[:B], run), torch.zeros_like(run))
        ops = torch.where(ise, mx, self.iw_ops[:B])
        mx = torch.where(ise, torch.zeros_like(mx), mx)
        self.iw_run[:B] = run; self.iw_max[:B] = mx; self.iw_ops[:B] = ops
        to = self.t_op[tokcur]
        cop = torch.where(isx, torch.where(to > 0, to, self.iw_cop[:B]), torch.zeros_like(to))
        sop = torch.where(ise, cop, self.iw_sop[:B]); cop = torch.where(ise, torch.zeros_like(cop), cop)
        self.iw_cop[:B] = cop; self.iw_sop[:B] = sop
        nz = e.abs().sum(-1) > 0
        start = nz & ~self.iw_prev[:B]
        hp = self.iw_h[:B].float()
        hn = (hp - hp.mean(-1, keepdim=True)) * torch.rsqrt(hp.var(-1, unbiased=False, keepdim=True) + 1e-5)
        logit = iw.k * 0.5 + iw.b + hn @ iw.v
        u = torch.rand(B, device=self.dev)
        ie = float(getattr(iw, 'eps', 0.0))                 # exploration floor: p = ie + (1 - 2 ie) sigmoid(logit)
        pf = ie + (1 - 2 * ie) * torch.sigmoid(logit)
        fire = torch.where(self.greedy, logit > 0, u < pf)
        sup = torch.where(start, ~fire, self.iw_sup[:B]) & nz
        e = e * (~sup)[:, None].to(e.dtype)
        brk = torch.zeros_like(nz)
        if self.partial is not None or self.partial_ops is not None:
            bad = (ops >= self.partial_t) if self.partial is not None else torch.isin(sop, self.partial_ops)
            brk = torch.where(start, bad, self.iw_brk[:B]) & nz
            self.iw_brk[:B] = brk
        dec = torch.where(start, torch.where(fire, 1, 2) + 4 * brk.to(torch.int64), 0).to(torch.int8)
        lf, lv = ((F.logsigmoid(logit), F.logsigmoid(-logit)) if ie == 0.0 else (torch.log(pf), torch.log1p(-pf)))
        lpd = torch.where(start, torch.where(fire, lf, lv), torch.zeros_like(logit))
        self.iw_prev[:B] = nz; self.iw_sup[:B] = sup
        self._ops_now = torch.where(start, ops.clamp(max=15) + 16 * sop, torch.zeros_like(ops)).to(torch.int8)   # digits | op<<4
        self._iw_rec = (logit.float(), run.clamp(max=127).to(torch.int8), mx.clamp(max=127).to(torch.int8))
        self._brk_now = brk
        return e, dec, lpd

    # ---------------- weights ----------------
    @torch.no_grad()
    def refresh(self):
        kv = Q.prefix_kv_of(self.W, self.prefix_ids, self.cos_t, self.sin_t)
        for i, (k, v) in enumerate(kv):
            self.Kp[i].copy_(k); self.Vp[i].copy_(v)

    # ---------------- one decode step (graph body) ----------------
    def _step(self, B, S):
        if self.kern == 'triton':
            return self._step_triton(B, S)
        return self._step_torch(B, S)

    def _step_torch(self, B, S):
        W, tok = self.W, self.tokb[:B]
        slot = self.slot[:B]
        ar = torch.arange(B, device=self.dev)
        emb = W['E'][tok]
        feat = None; write = None
        imp = self.imp if self.imp_on else None
        links = self.links if self.imp_on else None
        if imp is not None:
            e = self.imp.step(emb, B, tok)
            feat = e
            if self.use_write: write = (e @ self.Dp) if self.Dp is not None else (self.imp.last_w if self.imp.fk is not None else e @ self.imp.D)
            wf = (e.abs().sum(-1) > 0).to(torch.int8)
        x = emb
        pos = self.Lp + slot
        cos, sin = self.cos_t[pos][:, None], self.sin_t[pos][:, None]           # (B,1,HD)
        smask = torch.arange(S, device=self.dev)[None] <= slot[:, None]        # (B,S)
        neg = torch.finfo(torch.float32).min
        for i, lw in enumerate(W['L']):
            if links is not None and i in links:
                x = x + links[i](feat, x).to(x.dtype)
            h = Q.rms(x, lw['ln1'])
            qkv = F.linear(h, lw['Wqkv'])
            q = Q.rms(qkv[:, :Q.H * Q.HD].view(B, Q.H, Q.HD), lw['qn'])
            k = Q.rms(qkv[:, Q.H * Q.HD:(Q.H + Q.KVH) * Q.HD].view(B, Q.KVH, Q.HD), lw['kn'])
            v = qkv[:, (Q.H + Q.KVH) * Q.HD:].view(B, Q.KVH, Q.HD)
            q = q * cos + Q.rot(q) * sin
            k = k * cos + Q.rot(k) * sin
            self.Kc[i][ar, :, slot] = k; self.Vc[i][ar, :, slot] = v
            qg = q.view(B, Q.KVH, 2, Q.HD)
            sc = 1.0 / (Q.HD ** 0.5)
            # prefix part: one GEMM per kv head over ALL rows (prefix is shared)
            sp = torch.matmul(qg.permute(1, 0, 2, 3).reshape(Q.KVH, B * 2, Q.HD), self.Kp[i].transpose(1, 2)).float() * sc
            sp = sp.view(Q.KVH, B, 2, self.Lp).permute(1, 0, 2, 3)                      # (B,KVH,2,Lp)
            Ks, Vs = self.Kc[i][:B, :, :S], self.Vc[i][:B, :, :S]
            ss = torch.matmul(qg, Ks.transpose(-1, -2)).float() * sc                     # (B,KVH,2,S)
            ss = ss.masked_fill(~smask[:, None, None, :], neg)
            m = torch.maximum(sp.amax(-1, keepdim=True), ss.amax(-1, keepdim=True))
            pp, ps = torch.exp(sp - m), torch.exp(ss - m)
            den = pp.sum(-1, keepdim=True) + ps.sum(-1, keepdim=True)
            pp, ps = (pp / den).to(x.dtype), (ps / den).to(x.dtype)
            op = torch.matmul(pp.permute(1, 0, 2, 3).reshape(Q.KVH, B * 2, self.Lp), self.Vp[i]).view(Q.KVH, B, 2, Q.HD).permute(1, 0, 2, 3)
            a = op + torch.matmul(ps, Vs)                                                # (B,KVH,2,HD)
            x = x + F.linear(a.reshape(B, Q.H * Q.HD), lw['Wo'])
            x = x + Q.mlp(lw, x)
        if links is not None and 'final' in links:
            x = x + links['final'](feat, x).to(x.dtype)
        if write is not None:
            x = x + write.to(x.dtype)
        logits = F.linear(Q.rms(x, W['lnf']), W['E']).float()
        z = logits / self.temp
        g = -torch.log(-torch.log(torch.rand_like(z).clamp_(1e-20, 1.0)))
        nxt = torch.where(self.greedy, z.argmax(-1), (z + g).argmax(-1))
        lp = torch.log_softmax(z, -1).gather(-1, nxt[:, None])[:, 0]
        self._book(B, tok, nxt, lp, wf if imp is not None else None, imp)

    def _book(self, B, tok, nxt, lp, wf, imp, dec=None, ops=None):
        slot = self.slot[:B]
        done = self.done[:B]
        nxt = torch.where(done, torch.full_like(nxt, self.eos), nxt)
        live = ~done
        r, gl = self.rowid[:B], self.glen[:B]
        self.out_tok[r, gl + 1] = torch.where(live, nxt, self.out_tok[r, gl + 1])
        self.out_lp[r, gl + 1] = torch.where(live, lp, self.out_lp[r, gl + 1])
        if wf is not None:
            self.out_w[r, gl + 1] = torch.where(live, wf, self.out_w[r, gl + 1])
        if dec is not None:
            self.out_dec[r, gl + 1] = torch.where(live, dec, self.out_dec[r, gl + 1])
        if ops is not None:
            self.out_ops[r, gl + 1] = torch.where(live, ops, self.out_ops[r, gl + 1])
            lg_, rn_, sp_ = self._iw_rec                      # per-token in-wire logit + tracker (interpretation)
            self.out_iwl[r, gl + 1] = torch.where(live, lg_, self.out_iwl[r, gl + 1])
            self.out_run[r, gl + 1] = torch.where(live, rn_, self.out_run[r, gl + 1])
            self.out_span[r, gl + 1] = torch.where(live, sp_, self.out_span[r, gl + 1])
        stop = (nxt == self.eos) | self.has_nn[nxt] | (self.ends_nl[tok] & self.starts_nl[nxt]) | (gl + 2 >= self.maxnew)
        self.glen[:B] += live.long()
        self.done[:B] |= stop
        self.slot[:B] = torch.clamp(slot + 1, max=self.S_max - 1)
        self.tokb[:B] = nxt


    def _step_triton(self, B, S):
        """Same step with the Triton kernels (hostkern.py): ~10 kernels per layer instead of ~170."""
        W, tok = self.W, self.tokb[:B]
        slot = self.slot[:B]
        imp = self.imp if self.imp_on else None
        links = self.links if self.imp_on else None
        emb = W['E'][tok]
        feat = write = wf = None; dec = lpd = None
        if imp is not None:
            e = imp.step(emb, B, tok)
            if self.inwire is not None: e, dec, lpd = self._inwire_step(e, B, tok)
            feat = e
            if self.use_write:
                if self.inwire is not None:            # the veto masks e, so the kernel's precomputed write is stale
                    write = e @ (self.Dp if self.Dp is not None else imp.D.float())
                    if self.partial is not None or self.partial_ops is not None:
                        write = write + self._brk_now[:, None].float() * (e @ self.D_der_minus_D)
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
            if i == iwL:                                         # in-wire read: layer-L input, before links
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
        self._book(B, tok, nxt, lp, wf, imp, dec, self._ops_now if dec is not None else None)

    def _graph(self, B, S):
        key = (B, S, self.imp_on)
        if key not in self.graphs:
            st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
            saved = [t.clone() for t in self._mut()]
            with torch.cuda.stream(st), torch.no_grad():
                for _ in range(2): self._step(B, S)
            torch.cuda.current_stream().wait_stream(st)
            g = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(g, pool=self.pool):
                self._step(B, S)
            if self.pool is None: self.pool = g.pool()
            for t, s in zip(self._mut(), saved): t.copy_(s)
            self.graphs[key] = g
        return self.graphs[key]

    def _mut(self):
        m = [self.tokb, self.slot, self.glen, self.done, self.out_tok, self.out_lp, self.out_w]
        if self.imp is not None and self.imp_on: m += [self.imp.sym_h, self.imp.S_h, self.imp.key_h]
        if self.inwire is not None: m += [self.iw_h, self.iw_prev, self.iw_sup, self.out_dec, self.out_ops, self.out_iwl,
                                          self.out_run, self.out_span,
                                          self.iw_run, self.iw_max, self.iw_ops, self.iw_brk, self.iw_cop, self.iw_sop]
        return m   # KV cache rows written during warm-up are overwritten before use (slots >= current)

    def capture_all(self, max_rows):
        for B in [b for b in B_BUCKETS if b <= max_rows]:
            for S in range(S_STEP, self.S_max + 1, S_STEP):
                self._graph(B, S)

    # ---------------- prefill of questions ----------------
    @torch.no_grad()
    def _prefill(self, qids, rows):
        """qids: list of question token lists (one per prompt); rows: list of LongTensors (G target rows per prompt).
        Computes suffix KV once per prompt and copies it to its G rows. Returns first-token logits (P, V) fp32."""
        P = len(qids); T = max(map(len, qids))
        ids = torch.tensor([q + [self.eos] * (T - len(q)) for q in qids], device=self.dev)
        lens = torch.tensor([len(q) for q in qids], device=self.dev)
        prefix_kv = [(self.Kp[i], self.Vp[i]) for i in range(Q.NL)]
        write = feat = None
        imp_on = self.imp is not None and self.imp_on
        # links also run with NO implant (host-only adapters, feat=None) so prefill matches decode and the update pass
        links = self.links if (imp_on or (self.imp is None and self.imp_on)) else None
        TAG = imp_on and getattr(self.imp, 'tagf', False)
        if imp_on:
            full = torch.cat([self.prefix_ids[None].expand(P, -1), ids], 1)
            tail = []
            KW = (self.imp.K + self.imp.W) * (4 if TAG else 1)              # calc_tag: room for skipped tag tokens
            for p in range(P):
                Lt = self.Lp + len(qids[p])
                tail.append(full[p, Lt - KW:Lt])
            tail = torch.stack(tail)
            if TAG:
                self.imp.Eref = self.W['E']
                e_last = self.imp.emit_ids(tail)[:, -1]
            else:
                emb_tail = self.W['E'][tail]
                e_last = self.imp.M.emit(emb_tail)[0][:, -1]                     # implant output at the last prompt token
            eq = torch.zeros(P, T, self.imp.n_out, device=self.dev); eq[torch.arange(P), lens - 1] = e_last
            feat = eq
            if links is not None:                                       # links need e at every question position
                Kt = (self.imp.K + self.imp.W) * (4 if TAG else 1)
                ef = self.imp.emit_ids(full[:, self.Lp - Kt:])[:, Kt:] if TAG else self.imp.full(self.W['E'][full[:, self.Lp - Kt:]])[:, Kt:]
                feat = ef
            if self.use_write: write = eq @ (self.Dp if self.Dp is not None else self.imp.D)
        PC = 64                                              # prefill in chunks of prompts (bounded memory)
        logits = torch.empty(P, self.W['E'].shape[0], device=self.dev)
        allrows = torch.cat(rows); src = torch.cat([torch.full((len(r),), p, device=self.dev) for p, r in enumerate(rows)])
        for c0 in range(0, P, PC):
            sl = slice(c0, c0 + PC)
            iwc = self.inwire.L if (self.inwire is not None and imp_on) else None
            r_ = Q.fwd_chunk(self.W, ids[sl], self.Lp, self.cos_t, self.sin_t, prefix_kv=prefix_kv, links=links,
                             feat=None if feat is None else feat[sl], final_write=None if write is None else write[sl],
                             return_kv=True, capture=iwc)
            hn, kvs = r_[0], r_[1]
            n = hn.shape[0]
            logits[c0:c0 + n] = F.linear(hn[torch.arange(n, device=self.dev), lens[sl] - 1], self.W['E']).float()
            sel = (src >= c0) & (src < c0 + n)
            rr, ss = allrows[sel], src[sel] - c0
            for i, (k, v) in enumerate(kvs):
                self.Kc[i][rr, :, :T] = k[ss]; self.Vc[i][rr, :, :T] = v[ss]
            if iwc is not None:
                hc = r_[2][torch.arange(n, device=self.dev), lens[sl] - 1]
                self.iw_h[rr] = hc[ss].to(self.iw_h.dtype)
            del hn, kvs, r_
        if imp_on:
            if TAG: self.imp.prime_rows_ids(tail[src], allrows)
            else: self.imp.prime_rows(emb_tail[src], allrows)
            if self.inwire is not None:
                self.iw_prev[allrows] = (e_last.abs().sum(-1) > 0)[src]; self.iw_sup[allrows] = False
                for t in (self.iw_run, self.iw_max, self.iw_ops, self.iw_cop, self.iw_sop): t[allrows] = 0
                self.iw_brk[allrows] = False
        return logits, lens, (e_last if imp_on else None)

    # ---------------- generate ----------------
    def _alloc_out(self, max_new):
        need = (self.B_max + 1, max_new + 2)                     # row B_max = dummy row for padding / stale rows
        if getattr(self, '_outshape', None) != need:
            self._outshape = need
            self.out_tok = torch.full(need, self.eos, device=self.dev, dtype=torch.long)
            self.out_lp = torch.zeros(need, device=self.dev)
            self.out_w = torch.zeros(need, device=self.dev, dtype=torch.int8)
            self.out_dec = torch.zeros(need, device=self.dev, dtype=torch.int8)
            self.out_ops = torch.zeros(need, device=self.dev, dtype=torch.int8)
            self.out_iwl = torch.zeros(need, device=self.dev)
            self.out_run = torch.zeros(need, device=self.dev, dtype=torch.int8)
            self.out_span = torch.zeros(need, device=self.dev, dtype=torch.int8)
            self.graphs = {}                                       # graphs captured against the old buffers

    @torch.no_grad()
    def generate(self, qids, G=1, T=1.0, max_new=320, greedy=False):
        """qids: list of question token lists (already include 'Question: ...\nAnswer:'). Returns per-row lists
        (row r = prompt r//G, sample r%G): tokens, logp (old policy at temperature T), wflag[t] = 1 iff the implant's
        output e was nonzero at the step that produced token t (a real write event)."""
        assert max(map(len, qids)) + max_new <= self.S_max
        self._alloc_out(max_new)
        N = len(qids) * G
        ppc = max(1, self.B_max // G)
        res_tok, res_lp, res_w, res_d, res_o = [], [], [], [], []; res_i = []
        torch.cuda.synchronize(); t0 = time.time(); steps = 0
        for p0 in range(0, len(qids), ppc):
            qs = qids[p0:p0 + ppc]; n = len(qs) * G
            steps += self._wave(qs, G, T, max_new, greedy)
            ot, ol, ow, gl = self.out_tok[:n].cpu(), self.out_lp[:n].cpu(), self.out_w[:n].cpu(), self._glen_final.cpu()
            od = self.out_dec[:n].cpu(); oo = self.out_ops[:n].cpu()
            oi, orn, osp = self.out_iwl[:n].cpu(), self.out_run[:n].cpu(), self.out_span[:n].cpu()
            for j in range(n):
                L = int(gl[j]) + 1
                res_tok.append(ot[j, :L].tolist()); res_lp.append(ol[j, :L].tolist()); res_w.append(ow[j, :L].tolist())
                res_d.append(od[j, :L].tolist()); res_o.append(oo[j, :L].tolist())
                if self.inwire is not None: res_i.append((oi[j, :L].tolist(), orn[j, :L].tolist(), osp[j, :L].tolist()))
        torch.cuda.synchronize(); dt = time.time() - t0
        ntok = sum(map(len, res_tok))
        self.stats = dict(seconds=dt, tokens=ntok, tok_s=ntok / dt, decode_steps=steps, rows=N)
        return dict(tokens=res_tok, logp=res_lp, wflag=res_w, dec=res_d, ops=res_o, iw=res_i, stats=self.stats)

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
        Bb = next(b for b in B_BUCKETS if b >= n)
        self.tokb[:n] = first; self.slot[:n] = lens.repeat_interleave(G); self.glen[:n] = 0
        self.rowid[:n] = torch.arange(n, device=dev)
        self.done[:n] = (first == self.eos) | self.has_nn[first] | (max_new <= 1)
        self.tokb[n:Bb] = self.eos; self.slot[n:Bb] = 0; self.glen[n:Bb] = 0; self.rowid[n:Bb] = DUMMY; self.done[n:Bb] = True
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
            if nb < Bb:                                           # compaction: live rows to the front
                idx = torch.nonzero(~d)[:, 0]; fin = torch.nonzero(d)[:, 0]
                glen_final[torch.clamp(self.rowid[fin], max=n)] = self.glen[fin]
                su = min(self.S_max, maxlen + step + 1)
                for i in range(Q.NL):
                    self.Kc[i][:n_live, :, :su] = self.Kc[i][idx, :, :su]
                    self.Vc[i][:n_live, :, :su] = self.Vc[i][idx, :, :su]
                for t in (self.tokb, self.slot, self.glen, self.rowid, self.done):
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
