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
                 kern='triton', rethink=None, soft_V=None, tool=None):
        self.W, self.tok, self.imp, self.links = W, tok, implant, links
        self.tool = tool                      # [35] arm X: text calculator tool (tooltext.py); forced flags go to out_w
        self.rt = rethink
        self.soft_V = soft_V                  # arm 4s: (K, D) static buffer of soft phrase embeddings (updated in place)
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
        self.prev_sidx = torch.full((B_max + 1,), -1, device=dev, dtype=torch.long)
        if tool is not None:                  # per (compact) row: paused at a '</calc>', forced-token queue, '/'-age
            self.FMAX = 24
            self.paused = z(B_max + 1, dt=torch.bool); self.fpos = z(B_max + 1); self.flen = z(B_max + 1)
            self.ftok = z(B_max + 1, self.FMAX); self.age = z(B_max + 1)
            self.gt_tab = torch.tensor(['>' in t for t in txt], device=dev)
            self.sl_tab = torch.tensor(['/' in t for t in txt], device=dev)
            self.tool_log = []
        self.imp_on = True                                  # False: host only (implant + links off), separate graphs
        self.stats = {}
        self.refresh()

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
            e = self.imp.step(emb, B)
            feat = e
            if self.use_write: write = self.imp.last_w if self.imp.fk is not None else e @ self.imp.D
            wf = (e.abs().sum(-1) > 0).to(torch.int8)
        x = emb
        pos = self.Lp + slot
        cos, sin = self.cos_t[pos][:, None], self.sin_t[pos][:, None]           # (B,1,HD)
        smask = torch.arange(S, device=self.dev)[None] <= slot[:, None]        # (B,S)
        neg = torch.finfo(torch.float32).min
        for i, lw in enumerate(W['L']):
            if links is not None and i in links:
                x = x + links[i](feat).to(x.dtype)
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
            x = x + links['final'](feat).to(x.dtype)
        if write is not None:
            x = x + write.to(x.dtype)
        logits = F.linear(Q.rms(x, W['lnf']), W['E']).float()
        z = logits / self.temp
        g = -torch.log(-torch.log(torch.rand_like(z).clamp_(1e-20, 1.0)))
        nxt = torch.where(self.greedy, z.argmax(-1), (z + g).argmax(-1))
        lp = torch.log_softmax(z, -1).gather(-1, nxt[:, None])[:, 0]
        self._book(B, tok, nxt, lp, wf if imp is not None else None, imp)

    def _book(self, B, tok, nxt, lp, wf, imp):
        if self.tool is not None:
            return self._book_tool(B, tok, nxt, lp)
        slot = self.slot[:B]
        done = self.done[:B]
        nxt = torch.where(done, torch.full_like(nxt, self.eos), nxt)
        live = ~done
        r, gl = self.rowid[:B], self.glen[:B]
        self.out_tok[r, gl + 1] = torch.where(live, nxt, self.out_tok[r, gl + 1])
        self.out_lp[r, gl + 1] = torch.where(live, lp, self.out_lp[r, gl + 1])
        if wf is not None:
            self.out_w[r, gl + 1] = torch.where(live, wf, self.out_w[r, gl + 1])
        stop = (nxt == self.eos) | self.has_nn[nxt] | (self.ends_nl[tok] & self.starts_nl[nxt]) | (gl + 2 >= self.maxnew)
        self.glen[:B] += live.long()
        self.done[:B] |= stop
        self.slot[:B] = torch.clamp(slot + 1, max=self.S_max - 1)
        self.tokb[:B] = nxt

    def _book_tool(self, B, tok, nxt, lp):
        """Bookkeeping with the text tool: queued tool tokens replace the sample (flag 1 in out_w, logp 0); a sampled '>'
        within 4 tokens of a '/' pauses the row (no advance) until the host has checked it for '</calc>'."""
        slot, done, paused = self.slot[:B], self.done[:B], self.paused[:B]
        fp, fl = self.fpos[:B], self.flen[:B]
        live = ~done & ~paused
        has_f = live & (fp < fl)
        fnxt = self.ftok[:B].gather(1, fp.clamp(max=self.FMAX - 1)[:, None])[:, 0]
        nxt = torch.where(done, torch.full_like(nxt, self.eos), torch.where(has_f, fnxt, nxt))
        lp = torch.where(has_f, torch.zeros_like(lp), lp)
        r, gl = self.rowid[:B], self.glen[:B]
        self.out_tok[r, gl + 1] = torch.where(live, nxt, self.out_tok[r, gl + 1])
        self.out_lp[r, gl + 1] = torch.where(live, lp, self.out_lp[r, gl + 1])
        self.out_w[r, gl + 1] = torch.where(live, has_f.to(torch.int8), self.out_w[r, gl + 1])
        age = torch.where(self.sl_tab[nxt], torch.zeros_like(gl), self.age[:B] + 1)
        stop_len = gl + 2 >= self.maxnew
        newpause = live & ~has_f & self.gt_tab[nxt] & (age <= 4) & ~stop_len
        stop = ((nxt == self.eos) | self.has_nn[nxt] | (self.ends_nl[tok] & self.starts_nl[nxt])) & ~newpause | stop_len
        self.glen[:B] += live.long()
        self.done[:B] |= stop & live
        self.slot[:B] = torch.where(live, torch.clamp(slot + 1, max=self.S_max - 1), slot)
        self.tokb[:B] = torch.where(live, nxt, tok)
        self.age[:B] = torch.where(live, age, self.age[:B])
        self.fpos[:B] += has_f.long()
        self.paused[:B] |= newpause

    def _tool_sync(self, live_n, wave_calls):
        """Host side of the tool, at a sync point: every paused row whose text now ends a '</calc>' gets the exact value
        of its expression queued as forced tokens; every paused row is released."""
        pz = torch.nonzero(self.paused[:live_n] & ~self.done[:live_n])[:, 0]
        if pz.numel() == 0:
            self.paused[:live_n] = False
            return
        from tooltext import calc_eval, OPEN, CLOSE, MAX_CALLS
        enc = lambda s_: self.tok(s_, add_special_tokens=False)['input_ids']
        rid, gl = self.rowid[pz], self.glen[pz]
        toks = self.out_tok[rid].cpu(); gls = gl.tolist(); rids = rid.tolist(); pzl = pz.tolist()
        q_p, q_ids, fix_p, fix_r, fix_g, fix_t = [], [], [], [], [], []
        for j, (p, r, g) in enumerate(zip(pzl, rids, gls)):
            ids = toks[j, :g + 1].tolist()
            t_all = self.tok.decode(ids); t_prev = self.tok.decode(ids[:-1])
            k = t_all.rfind(CLOSE)
            if k < 0 or k + len(CLOSE) <= len(t_prev) or wave_calls[r] >= MAX_CALLS:
                continue
            o = t_all.rfind(OPEN, 0, k)
            expr = t_all[o + len(OPEN):k] if o >= 0 else None
            res = calc_eval(expr)
            force = []
            if len(t_all) > k + len(CLOSE):              # the sampled token ran past '>': cut it at the tag (tool-written)
                tail = enc(t_all[len(t_prev):k + len(CLOSE)])
                if tail:
                    fix_p.append(p); fix_r.append(r); fix_g.append(g); fix_t.append(tail[0]); force += tail[1:]
            force += enc(res)
            wave_calls[r] += 1
            self.tool_log.append((expr, res))
            q_p.append(p); q_ids.append(force[:self.FMAX])
        if fix_p:
            fp_ = torch.tensor(fix_p, device=self.dev); fr = torch.tensor(fix_r, device=self.dev)
            fg = torch.tensor(fix_g, device=self.dev); ft = torch.tensor(fix_t, device=self.dev)
            self.out_tok[fr, fg] = ft; self.out_w[fr, fg] = 1; self.tokb[fp_] = ft
        if q_p:
            buf = torch.full((len(q_p), self.FMAX), self.eos, dtype=torch.long)
            for j, f in enumerate(q_ids): buf[j, :len(f)] = torch.tensor(f, dtype=torch.long)
            qp = torch.tensor(q_p, device=self.dev)
            self.ftok[qp] = buf.to(self.dev); self.fpos[qp] = 0
            self.flen[qp] = torch.tensor([len(f) for f in q_ids], device=self.dev)
        nocall = [p for p in pzl if p not in set(q_p)]       # released without a call: apply the blank-line stop it skipped
        if nocall:
            nc = torch.tensor(nocall, device=self.dev)
            self.done[nc] |= self.has_nn[self.tokb[nc]]
        self.paused[:live_n] = False

    def _step_triton(self, B, S):
        """Same step with the Triton kernels (hostkern.py): ~10 kernels per layer instead of ~170."""
        W, tok = self.W, self.tokb[:B]
        slot = self.slot[:B]
        imp = self.imp if self.imp_on else None
        links = self.links if self.imp_on else None
        emb = W['E'][tok]
        feat = write = wf = None
        if imp is not None:
            e = imp.step(emb, B)
            feat = e
            if self.use_write: write = imp.last_w if imp.fk is not None else e @ imp.D
            wf = (e.abs().sum(-1) > 0).to(torch.int8)
        x, h, q, att = self.xb[:B], self.hb[:B], self.qb[:B], self.ab[:B]
        if self.soft_V is not None and self.imp_on:      # soft rethink: the host reads the soft vector, not the token
            ps = self.prev_sidx[:B]
            x.copy_(torch.where((ps >= 0)[:, None], self.soft_V[ps.clamp(min=0, max=self.soft_V.shape[0] - 1)], emb))
        else:
            x.copy_(emb)
        rt = self.rt if (self.imp_on and self.rt is not None) else None
        feats = {}
        pos = self.Lp + slot
        ieee = x.dtype == torch.float32
        sc = 1.0 / (Q.HD ** 0.5)
        delta = None
        for i, lw in enumerate(W['L']):
            if links is not None and i in links:
                if delta is not None: x += delta; delta = None
                x += links[i](feat).to(x.dtype)
            HK.add_rms_kernel[(B,)](x, delta if delta is not None else x, lw['ln1'], h, D=Q.D, HAS_DELTA=delta is not None, eps=Q.EPS)
            if rt is not None and i in rt.det_layers: feats[i] = x.clone()     # hidden_states[i]
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
            x += links['final'](feat).to(x.dtype)
        if write is not None:
            x += write.to(x.dtype)
        HK.add_rms_kernel[(B,)](x, x, W['lnf'], h, D=Q.D, HAS_DELTA=False, eps=Q.EPS)
        logits = F.linear(h, W['E'])
        if rt is not None:                                # mistake detector + rethink, after the calculator
            wr = rt.step(tok, logits, feats, pos, B, rowid=self.rowid[:B], live=~self.done[:B])
            m = (wr != 0).any(-1)
            x2 = torch.where(m[:, None], (x.float() + wr).to(x.dtype), x)
            logits = torch.where(m[:, None], F.linear(Q.rms(x2, W['lnf']), W['E']), logits)
            rflag = m.to(torch.int8)
            sidx = rt.last_sidx
        nxt, lp = self.nxt_b[:B], self.lp_b[:B]
        HK.sample_kernel[(B,)](logits, self.temp, self.greedy, self.seed, nxt, lp, logits.shape[1], BV=2048, num_warps=8)
        self.seed += 1
        if rt is not None:
            r_, gl_ = self.rowid[:B], self.glen[:B]
            self.out_rt[r_, gl_ + 1] = torch.where(~self.done[:B], rflag, self.out_rt[r_, gl_ + 1])
            self.out_sidx[r_, gl_ + 1] = torch.where(~self.done[:B], sidx.to(torch.int8), self.out_sidx[r_, gl_ + 1])
            self.prev_sidx[:B] = sidx
        self._book(B, tok, nxt, lp, wf, imp)

    def _graph(self, B, S):
        key = (B, S, self.imp_on, bool(self.rt.passive) if self.rt is not None else False)
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
        if self.rt is not None and self.imp_on: m += [self.out_sidx, self.prev_sidx, self.rt.rec_feat, self.rt.rec_has, self.rt.rec_dB, self.rt.rec_dA, self.rt.rec_sB, self.rt.rec_s0, self.out_rt, self.rt.fl_tail, self.rt.s_first, self.rt.s_gate, self.rt.s_cand, self.rt.s_fire, self.rt.s_pf, self.rt.frow_tail]
        if self.imp is not None and self.imp_on: m += [self.imp.sym_h, self.imp.S_h, self.imp.key_h]
        if self.tool is not None: m += [self.paused, self.fpos, self.flen, self.ftok, self.age]
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
        links = self.links if imp_on else None
        if imp_on:
            full = torch.cat([self.prefix_ids[None].expand(P, -1), ids], 1)
            tail = []
            for p in range(P):
                Lt = self.Lp + len(qids[p])
                tail.append(full[p, Lt - (self.imp.K + self.imp.W):Lt])
            tail = torch.stack(tail)
            emb_tail = self.W['E'][tail]
            e_last = self.imp.M.emit(emb_tail)[0][:, -1]                     # implant output at the last prompt token
            eq = torch.zeros(P, T, self.imp.n_out, device=self.dev); eq[torch.arange(P), lens - 1] = e_last
            feat = eq
            if links is not None:                                       # links need e at every question position
                Kt = self.imp.K + self.imp.W
                ef = self.imp.full(self.W['E'][full[:, self.Lp - Kt:]])[:, Kt:]
                feat = ef
            if self.use_write: write = eq @ self.imp.D
        PC = 64                                              # prefill in chunks of prompts (bounded memory)
        logits = torch.empty(P, self.W['E'].shape[0], device=self.dev)
        allrows = torch.cat(rows); src = torch.cat([torch.full((len(r),), p, device=self.dev) for p, r in enumerate(rows)])
        for c0 in range(0, P, PC):
            sl = slice(c0, c0 + PC)
            hn, kvs = Q.fwd_chunk(self.W, ids[sl], self.Lp, self.cos_t, self.sin_t, prefix_kv=prefix_kv, links=links,
                                  feat=None if feat is None else feat[sl], final_write=None if write is None else write[sl],
                                  return_kv=True)
            n = hn.shape[0]
            logits[c0:c0 + n] = F.linear(hn[torch.arange(n, device=self.dev), lens[sl] - 1], self.W['E']).float()
            sel = (src >= c0) & (src < c0 + n)
            rr, ss = allrows[sel], src[sel] - c0
            for i, (k, v) in enumerate(kvs):
                self.Kc[i][rr, :, :T] = k[ss]; self.Vc[i][rr, :, :T] = v[ss]
            del hn, kvs
        if imp_on:
            self.imp.prime_rows(emb_tail[src], allrows)
        if self.rt is not None and self.imp_on:
            last2 = torch.stack([ids[torch.arange(P, device=self.dev), lens - 2], ids[torch.arange(P, device=self.dev), lens - 1]], 1)
            self.rt.reset_rows(allrows, last2[src])
        return logits, lens, (e_last if imp_on else None)

    # ---------------- generate ----------------
    def _alloc_out(self, max_new):
        need = (self.B_max + 1, max_new + 2)                     # row B_max = dummy row for padding / stale rows
        if getattr(self, '_outshape', None) != need:
            self._outshape = need
            self.out_tok = torch.full(need, self.eos, device=self.dev, dtype=torch.long)
            self.out_lp = torch.zeros(need, device=self.dev)
            self.out_w = torch.zeros(need, device=self.dev, dtype=torch.int8)
            self.out_rt = torch.zeros(need, device=self.dev, dtype=torch.int8)
            self.out_sidx = torch.full(need, -1, device=self.dev, dtype=torch.int8)
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
        res_tok, res_lp, res_w, res_rt, res_sidx = [], [], [], [], []
        res_calls = []
        rt_rec = []
        torch.cuda.synchronize(); t0 = time.time(); steps = 0
        for p0 in range(0, len(qids), ppc):
            qs = qids[p0:p0 + ppc]; n = len(qs) * G
            steps += self._wave(qs, G, T, max_new, greedy)
            res_calls += self._wave_calls
            ot, ol, ow, gl = self.out_tok[:n].cpu(), self.out_lp[:n].cpu(), self.out_w[:n].cpu(), self._glen_final.cpu()
            orr = self.out_rt[:n].cpu(); osd = self.out_sidx[:n].cpu()
            if self.rt is not None and self.imp_on:
                rt_rec.append(dict(feat=self.rt.rec_feat[:n].clone(), has=self.rt.rec_has[:n].clone(), dB=self.rt.rec_dB[:n].clone(),
                                   dA=self.rt.rec_dA[:n].clone(), sB=self.rt.rec_sB[:n].clone(), s0=self.rt.rec_s0[:n].clone()))
            for j in range(n):
                L = int(gl[j]) + 1
                res_tok.append(ot[j, :L].tolist()); res_lp.append(ol[j, :L].tolist()); res_w.append(ow[j, :L].tolist())
                res_rt.append(orr[j, :L].tolist()); res_sidx.append(osd[j, :L].tolist())
        torch.cuda.synchronize(); dt = time.time() - t0
        ntok = sum(map(len, res_tok))
        self.stats = dict(seconds=dt, tokens=ntok, tok_s=ntok / dt, decode_steps=steps, rows=N)
        if self.tool is not None:
            self.stats.update(tool_calls=sum(res_calls), forced_tokens=sum(map(sum, res_w)))
        rec = {k: torch.cat([d[k] for d in rt_rec]) for k in rt_rec[0]} if rt_rec else None
        return dict(tokens=res_tok, logp=res_lp, wflag=res_w, rtflag=res_rt, sidx=res_sidx, rt_rec=rec, stats=self.stats,
                    calls=res_calls)

    def _wave(self, qs, G, T, max_new, greedy):
        n = len(qs) * G; dev = self.dev; DUMMY = self.B_max
        self.out_tok.fill_(self.eos); self.out_lp.zero_(); self.out_w.zero_(); self.out_rt.zero_(); self.out_sidx.fill_(-1); self.prev_sidx.fill_(-1)
        if self.rt is not None: self.rt.reset_records()
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
        self._wave_calls = [0] * n
        if self.tool is not None:
            self.paused.zero_(); self.fpos.zero_(); self.flen.zero_(); self.age.fill_(99)
        maxlen = int(lens.max()); live_n = n; step = 0
        max_steps = max_new * (3 if self.tool is not None else 1)     # paused rows do not advance
        glen_final = torch.zeros(n + 1, dtype=torch.long, device=dev)
        while True:
            Bb = next(b for b in B_BUCKETS if b >= live_n)
            Sb = self.S_max if self.kern == 'triton' else min(self.S_max, ((maxlen + step + self.sync_every) // S_STEP + 1) * S_STEP)
            g = self._graph(Bb, Sb)
            for _ in range(self.sync_every):
                g.replay()
            step += self.sync_every
            if self.tool is not None:
                self._tool_sync(live_n, self._wave_calls)
            d = self.done[:live_n]
            n_live = int((~d).sum())
            if n_live == 0 or step >= max_steps:
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
                if self.tool is not None:
                    for t in (self.paused, self.fpos, self.flen, self.ftok, self.age):
                        t[:n_live] = t[idx].clone()
                    self.paused[n_live:nb] = False; self.flen[n_live:nb] = 0
                if self.imp is not None and self.imp_on: self.imp.gather(idx, n_live)
                if self.rt is not None and self.imp_on: self.rt.gather(idx, n_live)
                self.prev_sidx[:n_live] = self.prev_sidx[idx].clone(); self.prev_sidx[n_live:nb] = -1
                self.done[n_live:nb] = True; self.rowid[n_live:nb] = DUMMY
                live_n = n_live
        glen_final[torch.clamp(self.rowid[:live_n], max=n)] = self.glen[:live_n]
        self._glen_final = glen_final[:n]
        return step
