"""Fast rollout engine: shared-prefix KV, static preallocated suffix KV cache, the WHOLE decode step (implant + host +
sampling + logprob + stop rule + bookkeeping) captured as CUDA graphs bucketed by (live rows, cache length), finished-row
compaction, Gumbel-max sampling fused into the step. Host does ONE graph replay per token and syncs every `sync_every`.

Usage:
    eng = Engine(W_bf16, prefix_ids, implant=CalcImplant|None, links=None, B_max=512, S_max=544)
    eng.refresh()                      # after a weight update (prefix KV recomputed in place; graphs stay valid)
    out = eng.generate(list_of_question_id_lists, G=8, T=1.0, max_new=320, greedy=False)
"""
import os, time, torch, triton, torch.nn.functional as F
import fastqwen as Q
import hostkern as HK

B_BUCKETS = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
S_STEP = 64


class Engine:
    def __init__(self, W, prefix_ids, tok, implant=None, links=None, B_max=512, S_max=544, use_write=True, sync_every=16,
                 kern='triton', rethink=None, soft_V=None, tool=None, cap_extra=0, ci=None):
        self.W, self.tok, self.imp, self.links = W, tok, implant, links
        self.tool = tool                      # [35] arm X: text calculator tool (tooltext.py); forced flags go to out_w
        self.cap_extra = cap_extra            # [37a]/[38]: > 0: max_new caps MODEL tokens only; tool tokens get up to cap_extra more
        self.ci = ci                          # [38] arm I (armi.ArmI): the tool path's call events feed the exact core + wiring
        assert ci is None or (tool is not None and kern == 'triton')
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
            self.nf = z(B_max + 1)                    # forced (tool-written) tokens booked so far, per compact row
            self.maxtot = torch.zeros((), device=dev, dtype=torch.long)
            self.gt_tab = torch.tensor(['>' in t for t in txt], device=dev)
            self.sl_tab = torch.tensor(['/' in t for t in txt], device=dev)
            self.tool_log = []
        self.ci_L = None
        if ci is not None:                    # queue symbol per item (-1 = forced tail token), registers, in-wire state
            self.fsym = torch.full((B_max + 1, self.FMAX), -1, device=dev, dtype=torch.long); self.rbits = z(B_max + 1)
            self.ci_cache = {}; self.ci_core_calls = 0; self.ci_core_s = 0.0          # exact memo of core outputs (CI_MEMO=0: none)
            self.ci_memo = os.environ.get('CI_MEMO', '1') != '0'
            if ci.inwire is not None:
                self.ci_L = ci.inwire.L
                self.iw_cur = torch.zeros(B_max + 1, Q.D, device=dev, dtype=bf); self.iw_h = torch.zeros_like(self.iw_cur)
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
        r, gl = self.rowid[:B], self.glen[:B]
        if self.ci is not None:               # [38] arm I: symbol items are WRITTEN (host samples, flag 2); tail items forced (1)
            fsy = self.fsym[:B].gather(1, fp.clamp(max=self.FMAX - 1)[:, None])[:, 0]
            forced = has_f & (fsy < 0); wr = has_f & (fsy >= 0)
            self.out_sym[r, gl + 1] = torch.where(live, torch.where(wr, fsy, -1).to(torch.int8), self.out_sym[r, gl + 1])
            self.out_rb[r, gl + 1] = torch.where(live, torch.where(wr, self.rbits[:B], 0), self.out_rb[r, gl + 1])
            wflag = forced.to(torch.int8) + 2 * wr.to(torch.int8)
        else:
            forced = has_f; wflag = has_f.to(torch.int8)
        nxt = torch.where(done, torch.full_like(nxt, self.eos), torch.where(forced, fnxt, nxt))
        lp = torch.where(forced, torch.zeros_like(lp), lp)
        self.out_tok[r, gl + 1] = torch.where(live, nxt, self.out_tok[r, gl + 1])
        self.out_lp[r, gl + 1] = torch.where(live, lp, self.out_lp[r, gl + 1])
        self.out_w[r, gl + 1] = torch.where(live, wflag, self.out_w[r, gl + 1])
        age = torch.where(self.sl_tab[nxt], torch.zeros_like(gl), self.age[:B] + 1)
        if self.cap_extra:                                   # model tokens only (+ a hard bound on the buffer)
            nf = self.nf[:B] + has_f.long()
            stop_len = (gl + 2 - nf >= self.maxnew) | (gl + 2 >= self.maxtot)
            self.nf[:B] = torch.where(live, nf, self.nf[:B])
        else:
            stop_len = gl + 2 >= self.maxnew
        newpause = live & ~has_f & self.gt_tab[nxt] & (age <= 4) & ~stop_len
        if self.ci_L is not None:             # in-wire reads the layer-L residual of the step that sampled the '>'
            self.iw_h[:B] = torch.where(newpause[:, None], self.iw_cur[:B], self.iw_h[:B])
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
        if self.ci is not None:
            return self._ci_sync(live_n, wave_calls)
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

    def _ci_sync(self, live_n, wave_calls):
        """[38] arm I at a sync point: the same call events as the text tool (_tool_sync), but the EXACT CORE (GPU tensor
        program, no Python arithmetic) evaluates every firing row's expression; the in-wire admits or vetoes each call;
        admitted rows get [cut tail tokens (forced)] + [result symbols (written by the wiring, sampled by the host)]."""
        pz = torch.nonzero(self.paused[:live_n] & ~self.done[:live_n])[:, 0]
        if pz.numel() == 0:
            self.paused[:live_n] = False
            return
        from tooltext import OPEN, CLOSE, MAX_CALLS
        from armi import bits_of
        from core import sym_str
        enc = lambda s_: self.tok(s_, add_special_tokens=False)['input_ids']
        rid, gl = self.rowid[pz], self.glen[pz]
        toks = self.out_tok[rid].cpu(); gls = gl.tolist(); rids = rid.tolist(); pzl = pz.tolist()
        C_p, C_r, C_g, C_e, C_t = [], [], [], [], []
        for j, (p, r, g) in enumerate(zip(pzl, rids, gls)):
            ids = toks[j, :g + 1].tolist()
            t_all = self.tok.decode(ids); t_prev = self.tok.decode(ids[:-1])
            k = t_all.rfind(CLOSE)
            if k < 0 or k + len(CLOSE) <= len(t_prev) or wave_calls[r] >= MAX_CALLS:
                continue
            o = t_all.rfind(OPEN, 0, k)
            C_e.append(t_all[o + len(OPEN):k] if o >= 0 else None)
            C_t.append(enc(t_all[len(t_prev):k + len(CLOSE)]) if len(t_all) > k + len(CLOSE) else None)
            C_p.append(p); C_r.append(r); C_g.append(g); wave_calls[r] += 1
        admitted = set()
        if C_p:
            # the core is a pure function of the expression string: memoise its outputs (exact; only misses run the core)
            cache = self.ci_cache if self.ci_memo else {}
            miss = list(dict.fromkeys(e for e in C_e if e not in cache))
            if miss:
                core = self.ci.core; t0_ = time.time()
                cp, ln = core.encode(miss)
                sym_m, slen_m, regs_m, _ = core(cp, ln)
                ss_m = sym_m.tolist(); sl_m = slen_m.tolist(); rs_m = sym_str(sym_m, slen_m); rg_m = regs_m.float().cpu()
                self.ci_core_s += time.time() - t0_
                for k_, e in enumerate(miss):
                    cache[e] = (ss_m[k_][:sl_m[k_]], rs_m[k_], rg_m[k_])
                self.ci_core_calls += len(miss)
            hit = [cache[e] for e in C_e]
            regs = torch.stack([h_[2] for h_ in hit]).to(self.dev)
            rb = bits_of(regs)
            P_ = torch.tensor(C_p, device=self.dev); R_ = torch.tensor(C_r, device=self.dev); G_ = torch.tensor(C_g, device=self.dev)
            if self.ci.inwire is not None and self.imp_on:
                with torch.no_grad():
                    lg = self.ci.inwire.logit(self.iw_h[P_])
                    adm = self.ci.inwire.decide(lg, bool(self.greedy))
                    lpd = self.ci.inwire.logp(lg, adm)
                self.out_dec[R_, G_] = torch.where(adm, 1, 2).to(torch.int8); self.out_lpd[R_, G_] = lpd.float()
            else:
                adm = torch.ones(len(C_p), dtype=torch.bool, device=self.dev)
                self.out_dec[R_, G_] = 1
            if not self.imp_on: adm = torch.zeros_like(adm)             # host-only eval: calls go unanswered
            adl = adm.tolist(); syl = [h_[0] for h_ in hit]; res_s = [h_[1] for h_ in hit]
            regl = regs.cpu()
            fix_p, fix_r, fix_g, fix_t, q_p, q_tok, q_sym, q_rb = [], [], [], [], [], [], [], []
            for j, (p, r, g) in enumerate(zip(C_p, C_r, C_g)):
                rg = regl[j]
                md = rg[31:39].nonzero()                          # registers: max operand digits one-hot (31..38), opset (39..44)
                self.tool_log.append((C_e[j], res_s[j], bool(adl[j]), int(md[0, 0]) + 1 if md.numel() else 0,
                                      ''.join(c for c, b in zip('+-*/(^', rg[39:45].tolist()) if b > 0.5)))
                if not adl[j]: continue
                admitted.add(p)
                ft, fs = [], []
                tail = C_t[j]
                if tail:                                          # cut the sampled token at the tag (forced, as arm T)
                    fix_p.append(p); fix_r.append(r); fix_g.append(g); fix_t.append(tail[0])
                    ft += tail[1:]; fs += [-1] * len(tail[1:])
                ss = syl[j]
                ft += [int(self.ci.sym_tok[s]) for s in ss]; fs += ss
                q_p.append(p); q_tok.append(ft[:self.FMAX]); q_sym.append(fs[:self.FMAX]); q_rb.append(j)
            if fix_p:
                fp_ = torch.tensor(fix_p, device=self.dev); fr = torch.tensor(fix_r, device=self.dev)
                fg = torch.tensor(fix_g, device=self.dev); ftt = torch.tensor(fix_t, device=self.dev)
                self.out_tok[fr, fg] = ftt; self.out_w[fr, fg] = 1; self.tokb[fp_] = ftt
            if q_p:
                bt = torch.full((len(q_p), self.FMAX), self.eos, dtype=torch.long)
                bs = torch.full((len(q_p), self.FMAX), -1, dtype=torch.long)
                for j, (f_, s_) in enumerate(zip(q_tok, q_sym)):
                    bt[j, :len(f_)] = torch.tensor(f_, dtype=torch.long); bs[j, :len(s_)] = torch.tensor(s_, dtype=torch.long)
                qp = torch.tensor(q_p, device=self.dev)
                self.ftok[qp] = bt.to(self.dev); self.fsym[qp] = bs.to(self.dev); self.fpos[qp] = 0
                self.flen[qp] = torch.tensor([len(f) for f in q_tok], device=self.dev)
                self.rbits[qp] = rb[torch.tensor(q_rb, device=self.dev)]
        nocall = [p for p in pzl if p not in admitted]       # released without a write: apply the blank-line stop it skipped
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
        if self.ci is not None and self.imp_on:          # [38] arm I: feature of the symbol being written (zeros elsewhere)
            import armi as AI
            fp = self.fpos[:B]
            hq = ~self.done[:B] & ~self.paused[:B] & (fp < self.flen[:B])
            s_ = self.fsym[:B].gather(1, fp.clamp(max=self.FMAX - 1)[:, None])[:, 0]
            feat = AI.feat_of(torch.where(hq, s_, torch.full_like(s_, -1)), self.rbits[:B])
            links = self.ci.links
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
                x += links[i](feat, x).to(x.dtype)
            HK.add_rms_kernel[(B,)](x, delta if delta is not None else x, lw['ln1'], h, D=Q.D, HAS_DELTA=delta is not None, eps=Q.EPS)
            if self.ci_L is not None and i == self.ci_L: self.iw_cur[:B].copy_(x)
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
            x += links['final'](feat, x).to(x.dtype)
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
        if self.tool is not None: m += [self.paused, self.fpos, self.flen, self.ftok, self.age, self.nf]
        if self.ci is not None:
            m += [self.out_sym, self.out_rb, self.fsym, self.rbits]
            if self.ci_L is not None: m += [self.iw_cur, self.iw_h]
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
            self.out_sym = torch.full(need, -1, device=self.dev, dtype=torch.int8)        # [38] written symbol per token
            self.out_rb = torch.zeros(need, device=self.dev, dtype=torch.long)            # its registers (bitmask)
            self.out_dec = torch.zeros(need, device=self.dev, dtype=torch.int8)           # in-wire decision at the '>' token
            self.out_lpd = torch.zeros(need, device=self.dev)                             # its log-prob
            self.graphs = {}                                       # graphs captured against the old buffers

    @torch.no_grad()
    def generate(self, qids, G=1, T=1.0, max_new=320, greedy=False):
        """qids: list of question token lists (already include 'Question: ...\nAnswer:'). Returns per-row lists
        (row r = prompt r//G, sample r%G): tokens, logp (old policy at temperature T), wflag[t] = 1 iff the implant's
        output e was nonzero at the step that produced token t (a real write event)."""
        assert max(map(len, qids)) + max_new + self.cap_extra <= self.S_max
        self._alloc_out(max_new + self.cap_extra)
        N = len(qids) * G
        ppc = max(1, self.B_max // G)
        res_tok, res_lp, res_w, res_rt, res_sidx = [], [], [], [], []
        res_calls = []
        res_sym, res_rb, res_dec, res_lpd = [], [], [], []
        rt_rec = []
        torch.cuda.synchronize(); t0 = time.time(); steps = 0
        for p0 in range(0, len(qids), ppc):
            qs = qids[p0:p0 + ppc]; n = len(qs) * G
            steps += self._wave(qs, G, T, max_new, greedy)
            res_calls += self._wave_calls
            ot, ol, ow, gl = self.out_tok[:n].cpu(), self.out_lp[:n].cpu(), self.out_w[:n].cpu(), self._glen_final.cpu()
            orr = self.out_rt[:n].cpu(); osd = self.out_sidx[:n].cpu()
            if self.ci is not None:
                osy, orb, odc, olp = self.out_sym[:n].cpu(), self.out_rb[:n].cpu(), self.out_dec[:n].cpu(), self.out_lpd[:n].cpu()
            if self.rt is not None and self.imp_on:
                rt_rec.append(dict(feat=self.rt.rec_feat[:n].clone(), has=self.rt.rec_has[:n].clone(), dB=self.rt.rec_dB[:n].clone(),
                                   dA=self.rt.rec_dA[:n].clone(), sB=self.rt.rec_sB[:n].clone(), s0=self.rt.rec_s0[:n].clone()))
            for j in range(n):
                L = int(gl[j]) + 1
                res_tok.append(ot[j, :L].tolist()); res_lp.append(ol[j, :L].tolist()); res_w.append(ow[j, :L].tolist())
                res_rt.append(orr[j, :L].tolist()); res_sidx.append(osd[j, :L].tolist())
                if self.ci is not None:
                    res_sym.append(osy[j, :L].tolist()); res_rb.append(orb[j, :L].tolist())
                    res_dec.append(odc[j, :L].tolist()); res_lpd.append(olp[j, :L].tolist())
        torch.cuda.synchronize(); dt = time.time() - t0
        ntok = sum(map(len, res_tok))
        self.stats = dict(seconds=dt, tokens=ntok, tok_s=ntok / dt, decode_steps=steps, rows=N)
        if self.tool is not None:
            self.stats.update(tool_calls=sum(res_calls), forced_tokens=sum(sum(1 for x in w if x == 1) for w in res_w),
                              written_tokens=sum(sum(1 for x in w if x == 2) for w in res_w))
        rec = {k: torch.cat([d[k] for d in rt_rec]) for k in rt_rec[0]} if rt_rec else None
        return dict(tokens=res_tok, logp=res_lp, wflag=res_w, rtflag=res_rt, sidx=res_sidx, rt_rec=rec, stats=self.stats,
                    calls=res_calls, sym=res_sym, rb=res_rb, dec=res_dec, lpd=res_lpd)

    def _wave(self, qs, G, T, max_new, greedy):
        n = len(qs) * G; dev = self.dev; DUMMY = self.B_max
        self.out_tok.fill_(self.eos); self.out_lp.zero_(); self.out_w.zero_(); self.out_rt.zero_(); self.out_sidx.fill_(-1); self.prev_sidx.fill_(-1)
        if self.rt is not None: self.rt.reset_records()
        rows = [torch.arange(p * G, (p + 1) * G, device=dev) for p in range(len(qs))]
        logits, lens, e_last = self._prefill(qs, rows)
        self.temp.fill_(T); self.greedy.fill_(greedy); self.maxnew.fill_(max_new)
        if self.tool is not None: self.maxtot.fill_(max_new + self.cap_extra)
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
            self.paused.zero_(); self.fpos.zero_(); self.flen.zero_(); self.age.fill_(99); self.nf.zero_()
        self.out_sym.fill_(-1); self.out_rb.zero_(); self.out_dec.zero_(); self.out_lpd.zero_()
        if self.ci is not None:
            self.fsym.fill_(-1); self.rbits.zero_()
        maxlen = int(lens.max()); live_n = n; step = 0
        max_steps = (max_new + self.cap_extra) * (3 if self.tool is not None else 1)     # paused rows do not advance
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
                    for t in (self.paused, self.fpos, self.flen, self.ftok, self.age, self.nf):
                        t[:n_live] = t[idx].clone()
                if self.ci is not None:
                    for t in (self.fsym, self.rbits) + ((self.iw_h,) if self.ci_L is not None else ()):
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
