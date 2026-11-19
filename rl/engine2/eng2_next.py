"""engine2 decode engine (stage 2a): the old LibEngine with
  * all 9 micromachines (calc_s4 + 8 BBH) stepped by ONE tools-megakernel launch per token (tk3_kernel.cu) on a side
    stream, overlapping host layers 0..13 (joined at the in-wire layer), instead of ~50 torch subgraphs;
  * prompt priming / prompt-position features by a scan of the same kernel over the prompt (no CSR teacher forcing);
  * lazy IFEval write (rows that cannot write skipped; see IFLazy) [optional, stage-2a step 2];
  * host layers, in-wires, adapters, sampling, bookkeeping: the old engine's kernels, unchanged (bit-identical host).
"""
import torch, torch.nn.functional as F, triton
import fastqwen as Q
import hostkern as HK
from lib_engine import LibEngine, _lnorm
from engine import B_BUCKETS
import tk


class LibEngine2(LibEngine):
    def __init__(self, *a, tk_cls='ToolsKernel3', tk_scratch=0, **kw):
        super().__init__(*a, **kw)
        machines = [('calc', self.imp.M, self.W['E'])] + [(t.name, t.M, self.E0) for t in self.tools]
        self.TK = getattr(tk, tk_cls)(machines, self.B_max, **({'scratch': tk_scratch} if tk_scratch else {}))
        self.Kt_max = max(M.K + M.W for _, M, _ in machines)
        self._tk_side = torch.cuda.Stream(priority=-5)   # highest priority: tools are on the critical path (joined at L14)
        TKr = self.TK
        self.imp.gather = lambda idx, n: TKr.gather(idx, n)           # the old _wave's compaction calls these
        for t in self.tools: t.gather = lambda idx, n: None
        self.graphs = {}

    # ------------------------------------------------------------ recording of raw tool outputs (for the trainer)
    def _alloc_out(self, max_new):
        need = (self.B_max + 1, max_new + 2)
        if getattr(self, '_outshape', None) != need:
            super()._alloc_out(max_new)
            self.out_E = torch.zeros(*need, self.TK.nout, device=self.dev, dtype=torch.float32)   # raw values (not 0/1 beyond design range)
            self.efin = torch.zeros(self.B_max + 1, device=self.dev, dtype=torch.bool)
            self._outshape = need

    def _mut(self):
        m = super()._mut() + list(self.TK._state_ids)
        if hasattr(self, 'out_E'): m += [self.out_E, self.efin]
        return m

    def _book_lib(self, B, tok, nxt, lp, wf, dec):
        if self.imp_on and hasattr(self, 'out_E'):
            done = self.done[:B]; r, gl = self.rowid[:B], self.glen[:B]
            wE = (~done) | (~self.efin[:B])          # live rows + one extra step for a just-finished row (its final token)
            self.out_E[r, gl + 1] = torch.where(wE[:, None], self._Eraw, self.out_E[r, gl + 1])
            self.efin[:B] |= done
        super()._book_lib(B, tok, nxt, lp, wf, dec)

    @torch.no_grad()
    def generate(self, qids, G=1, T=1.0, max_new=320, greedy=False):
        """old LibEngine.generate + per-row raw tool outputs: res['E'][j] (len+1, nout) int8, index k >= 1 = outputs of
        the step that consumed response token k-1 (index len = the final token's step), index 0 = last prompt position;
        res['Eq'][j] (len(q), nout) fp32 = outputs at the question positions (raw values; int8 would wrap anything not in {0,1})."""
        import time
        assert max(map(len, qids)) + max_new <= self.S_max
        self._alloc_out(max_new)
        ppc = max(1, self.B_max // G)
        res = dict(tokens=[], logp=[], wflag=[], dec=[], wt=[], ifw=[], if_lat=[], if_lat_prompt=[], if_ans=[], E=[], Eq=[])
        torch.cuda.synchronize(); t0 = time.time(); steps = 0
        for p0 in range(0, len(qids), ppc):
            qs = qids[p0:p0 + ppc]; n = len(qs) * G
            self.out_E.zero_(); self.efin.zero_()
            steps += self._wave(qs, G, T, max_new, greedy)
            ot, ol, ow, gl = self.out_tok[:n].cpu(), self.out_lp[:n].cpu(), self.out_w[:n].cpu(), self._glen_final.cpu()
            od, owt = self.out_dec3[:n].cpu(), self.out_wt[:n].cpu()
            oE = self.out_E[:n].cpu() if self.imp_on else None
            Eq = self._Eq_wave.cpu() if self.imp_on else None
            for j in range(n):
                L = int(gl[j]) + 1
                res['tokens'].append(ot[j, :L].tolist()); res['logp'].append(ol[j, :L].tolist()); res['wflag'].append(ow[j, :L].tolist())
                res['dec'].append(od[j, :L].clone()); res['wt'].append(owt[j, :L].clone())
                if self.imp_on:
                    res['E'].append(oE[j, :L + 1].clone()); res['Eq'].append(Eq[j // G, :len(qs[j // G])].clone())
                if self.iftool is not None and self.imp_on:
                    w_ = self.out_ifw[j, :L]
                    nzp = (w_.abs().sum(-1) > 0).nonzero()[:, 0]
                    res['ifw'].append((nzp.cpu(), w_[nzp].cpu()))
                    res['if_lat'].append(bool(self._lat_final[j])); res['if_lat_prompt'].append(bool(self._latp_final[j]))
                    res['if_ans'].append(bool(self._ans_final[j]))
        torch.cuda.synchronize(); dt = time.time() - t0
        ntok = sum(map(len, res['tokens']))
        self.stats = dict(seconds=dt, tokens=ntok, tok_s=ntok / dt, decode_steps=steps, rows=len(res['tokens']))
        res['stats'] = self.stats
        return res

    # ------------------------------------------------------------ weights
    @torch.no_grad()
    def refresh(self):
        super().refresh()
        if hasattr(self, 'TK'): self.TK.update_symtab(0, self.W['E'])   # calc reads the LIVE host embedding

    # ------------------------------------------------------------ decode step (old _step_triton, tools replaced)
    def _step_triton(self, B, S):
        W, tok = self.W, self.tokb[:B]
        slot = self.slot[:B]
        on = self.imp_on
        links = self.links if on else None
        emb = W['E'][tok]
        feat = None; write = None; wf = None; dec = None; lpd = None; hn = None
        BM = self.B_max
        cur = torch.cuda.current_stream()
        side_if = self._streams()[2]
        if on:
            hn = _lnorm(self.iw_h[:B])
            tks = cur if getattr(self, 'tk_serial', False) else self._tk_side
            tks.wait_stream(cur)
            with torch.cuda.stream(tks):
                Eall = self.TK.out_e if getattr(self, 'diag_skip_tk', False) else self.TK.step(self.tokb, B)   # diag: timing only
                self._Eraw = Eall[:B]
                es_all = [Eall[:B, o:o + n] * self.tool_on[i] for i, (o, n) in enumerate(self.TK.outoff)]
            if self.iftool is not None:
                side_if.wait_stream(cur)
                with torch.cuda.stream(side_if): self._if_side(B)
        x, h, q, att = self.xb[:B], self.hb[:B], self.qb[:B], self.ab[:B]
        x.copy_(emb)
        pos = self.Lp + slot
        ieee = x.dtype == torch.float32
        sc = 1.0 / (Q.HD ** 0.5)
        delta = None
        iwL = self.mwire.L if (self.mwire is not None and on) else -1
        for i, lw in enumerate(W['L']):
            if i == iwL:
                if delta is not None: x += delta; delta = None
                self.iw_h[:B].copy_(x)
                if not getattr(self, 'tk_serial', False): cur.wait_stream(self._tk_side)
                es, dec, lpd, wf = self._decide(es_all, hn, B)
                feat = {'calc': es[0]}
                for j, t in enumerate(self.tools):
                    feat[t.name] = es[j + 1]
                    w_ = es[j + 1] @ t.D
                    write = w_ if write is None else write + w_
                if self.iftool is not None: feat['ifeval_prev'] = self.if_prev_f[:B]
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
        if on and self.iftool is not None:
            self._if_write_step(B, hn, feat, cur, side_if)
            write = write + self._wif
            dec = torch.cat([dec, self._deci[:, None]], -1)
            wf = torch.cat([wf, self._wfi[:, None]], -1)
            lpd = lpd + self._lpdi
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
        self._book_lib(B, tok, nxt, lp, wf, dec)

    def _if_side(self, B):
        self.iftool.if_step(self.tokb[:self.B_max])

    def _if_write_step(self, B, hn, feat, cur, side_if):
        """IFEval write + its in-wire decision: the old code (eager on all B_max rows)."""
        BM = self.B_max; mw = self.mwire; ie = float(mw.eps)
        cur.wait_stream(side_if)
        wif = self.iftool.if_write(self.xb[:BM])[:B] * self.tool_on[-1]
        self._if_tail(B, hn, feat, wif)

    def _if_tail(self, B, hn, feat, wif):
        mw = self.mwire; ie = float(mw.eps)
        self.if_lat[:B] |= self.iftool.latched()[:B]
        self.if_ans[:B] |= (self.iftool.c.ans[:B] > 0.5)
        nzi = wif.abs().sum(-1) > 0
        j = self.NT - 1
        st_i = nzi & ~self.iw_prev[:B, j]
        lg_i = mw.k * 0.5 + mw.b[j] + hn @ mw.v[j]
        pfi = ie + (1 - 2 * ie) * torch.sigmoid(lg_i)
        fi = torch.where(self.greedy, lg_i > 0, torch.rand(B, device=self.dev) < pfi)
        supi = torch.where(st_i, ~fi, self.iw_sup[:B, j]) & nzi
        self._deci = torch.where(st_i, torch.where(fi, 1, 2), 0).to(torch.int8)
        self._lpdi = torch.where(st_i, torch.where(fi, torch.log(pfi), torch.log1p(-pfi)), torch.zeros_like(lg_i))
        self.iw_prev[:B, j] = nzi; self.iw_sup[:B, j] = supi
        wif = wif * (~supi)[:, None].to(wif.dtype)
        self.out_ifw_cur = wif
        fcur = wif / wif.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        feat['ifeval'] = fcur
        self.if_prev_f[:B].copy_(fcur)
        self._wif = wif; self._wfi = nzi & ~supi

    # ------------------------------------------------------------ prefill: tools by a kernel scan over the prompts
    @torch.no_grad()
    def _tk_prime(self, qids, rows):
        """Scan the tools kernel over [left pad][256 eos + prefix][question] for P prompt rows (TK rows 0..P-1),
        return per-tool features at question positions (P, Tq, n_out), then broadcast the states to the G rows."""
        P = len(qids); Tq = max(map(len, qids)); dev = self.dev
        Ls = Tq + self.Kt_max
        ids = torch.full((P, Ls), self.eos, dtype=torch.long, device=dev)
        tp = self.tool_prefix
        for p, q in enumerate(qids):
            s = torch.cat([tp, torch.tensor(q, device=dev)])[-(len(q) + self.Kt_max):]
            ids[p, Ls - len(s):] = s
        TK = self.TK
        TK.reset_rows(torch.arange(TK.Bp, device=dev))
        feats = torch.zeros(P, Tq, TK.nout, device=dev)
        lens = [len(q) for q in qids]
        for t in range(Ls):
            E = TK.step(ids[:, t], P)
            k = t - (Ls - Tq)                       # right-aligned question index; row p's question starts at Tq - len
            if k >= 0: feats[:, k] = E[:P]
        # left-align (old layout: q right-padded) -> features at question positions 0..len-1
        out = torch.zeros_like(feats)
        for p, n in enumerate(lens): out[p, :n] = feats[p, Tq - n:]
        src = torch.cat([torch.full((len(r),), p, device=dev, dtype=torch.long) for p, r in enumerate(rows)])
        allrows = torch.cat(rows)
        n = int(allrows.max()) + 1
        idx = torch.zeros(n, dtype=torch.long, device=dev); idx[allrows] = src
        TK.gather(idx, n)
        return out

    @torch.no_grad()
    def _prefill(self, qids, rows):
        P = len(qids); T = max(map(len, qids))
        ids = torch.tensor([q + [self.eos] * (T - len(q)) for q in qids], device=self.dev)
        lens = torch.tensor([len(q) for q in qids], device=self.dev)
        prefix_kv = [(self.Kp[i], self.Vp[i]) for i in range(Q.NL)]
        on = self.imp_on
        links = self.links if on else None
        feat = write = None; e_last_all = None
        allrows = torch.cat(rows); src = torch.cat([torch.full((len(r),), p, device=self.dev) for p, r in enumerate(rows)])
        if on:
            fz = self._tk_prime(qids, rows); self._Eq_wave = fz
            tl_on = self.tool_on
            fs_ = [fz[:, :, o:o + n] * tl_on[i] for i, (o, n) in enumerate(self.TK.outoff)]
            feat = {'calc': fs_[0]}
            for j, t in enumerate(self.tools): feat[t.name] = fs_[j + 1]
            ar = torch.arange(P, device=self.dev)
            e_last = [f[ar, lens - 1] for f in fs_]
            write = torch.zeros(P, T, Q.D, device=self.dev)
            for i, t in enumerate(self.tools):
                write[ar, lens - 1] += e_last[i + 1] @ t.D
            if self.iftool is not None:
                feat['ifeval_prev'] = torch.zeros(P, T, Q.D, device=self.dev)
                feat['ifeval'] = feat['ifeval_prev']
            e_last_all = torch.stack([e.abs().sum(-1) > 0 for e in e_last], -1)
        PC = getattr(self, 'prefill_chunk', 64)
        logits = torch.empty(P, self.W['E'].shape[0], device=self.dev)
        iwc = self.mwire.L if (self.mwire is not None and on) else None
        for c0 in range(0, P, PC):
            sl = slice(c0, c0 + PC)
            fs = None if feat is None else {k: v[sl] for k, v in feat.items()}
            r_ = Q.fwd_chunk(self.W, ids[sl], self.Lp, self.cos_t, self.sin_t, prefix_kv=prefix_kv, links=links,
                             feat=fs, final_write=None if write is None else write[sl], return_kv=True, capture=iwc)
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
        if on:
            if self.mwire is not None:
                NTb = 1 + len(self.tools)
                self.iw_prev[allrows] = False; self.iw_sup[allrows] = False
                self.iw_prev[allrows, :NTb] = e_last_all[src]
            if self.iftool is not None:
                self._if_prime(qids, rows)
                self.if_prev_f.zero_(); self.if_lat.zero_(); self.if_ans.zero_()
                self.if_lat[:self.B_max] |= self.if_lat_prompt
        self._e_last_all = e_last_all
        return logits, lens, (None if e_last_all is None else e_last_all[:, 0].float())


from torch.utils._pytree import tree_flatten, tree_unflatten


def if_can_write(c):
    """rows whose IF write can be nonzero this step: any latch, hforced, or a running emitter program."""
    Lst = torch.stack([x for x in c.L.values()], -1)
    return (Lst > 0).any(-1) | (c.prog != 0) | (c.hforced != 0)


def if_write_subset(c, v, h, sel, C, bind):
    """IF write on rows sel (C of the B rows bound in views v); scatters the subset's new state back into v, applies
    the write's only silent-row state effect (pre/suf compaction, idempotent) to all rows; returns the (C, D) write."""
    sub = {a: ([t[sel] if bm else t for t, bm in zip(lv, msk)], spec, msk) for a, (lv, spec, msk) in v.items()}
    bind(sub, C)
    ws = c.write(h[sel]).float()
    todo = []
    for a, (lv, spec, msk) in v.items():
        cl, cspec = tree_flatten(getattr(c, a)); assert cspec == spec
        for t, s, s0, bm in zip(lv, cl, sub[a][0], msk):
            if not torch.is_tensor(t) or s is s0: continue
            todo.append((t, s.clone(), bm))
    for t, s, bm in todo:
        if bm: t[sel] = s
        else: t.copy_(s)
    B = h.shape[0]
    bind(v, B)
    for nm, ln in (('suf', 'suf_len'), ('pre', 'pre_len')):
        x_ = getattr(c, nm); order = torch.sort((x_ < 0).long(), dim=1, stable=True).indices
        x2 = x_.gather(1, order); x_.copy_(x2); getattr(c, ln).copy_((x2 >= 0).sum(1))
    return ws


class LibEngine2Lazy(LibEngine2):
    """+ lazy IFEval: the IF token step runs on the live bucket's B rows (not B_max); the IF write runs on a capacity-C
    subset of rows that contains every row that can write this step (any latch, hforced, or a running emitter
    program; for every other row the write is exactly zero and its only state effect, the pre/suf compaction, is applied
    to all rows). If more than C rows could write, a device flag is raised; the host then rolls the 16-step block
    back (snapshot of all mutable state) and replays it with C = B. Rows are independent in the circuit, so any subset
    gives the same per-row result as the full batch."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.ifC = None
        self.if_over = torch.zeros((), dtype=torch.bool, device=self.dev)
        self.n_rollback = 0; self.ifC_hist = []

    def _graph(self, B, S):
        key = (B, S, self.imp_on, self.ifC if self.ifC is None else min(self.ifC, B))
        if key not in self.graphs:
            st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
            saved = [t.clone() for t in self._mut()]
            ov = self.if_over.clone()
            with torch.cuda.stream(st), torch.no_grad():
                for _ in range(2): self._step(B, S)
            torch.cuda.current_stream().wait_stream(st)
            g = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(g, pool=self.pool):
                self._step(B, S)
            if self.pool is None: self.pool = g.pool()
            for t, s in zip(self._mut(), saved): t.copy_(s)
            self.if_over.copy_(ov)
            self.graphs[key] = g
        return self.graphs[key]

    # ---- IF state views on the first B rows (batch leaves = dim 0 == B_max in the static state)
    def _if_views(self, B):
        it = self.iftool; BM = self.B_max
        return {a: ([t[:B] if (torch.is_tensor(t) and t.dim() > 0 and t.shape[0] == BM) else t for t in leaves], spec,
                    [bool(torch.is_tensor(t) and t.dim() > 0 and t.shape[0] == BM) for t in leaves])
                for a, (leaves, spec) in it.static.items()}

    def _if_bind(self, views, Bn):
        c = self.iftool.c
        for a, (lv, spec, _) in views.items(): setattr(c, a, tree_unflatten(lv, spec))
        c.B = Bn

    def _if_store(self, views):
        c = self.iftool.c
        sid = {id(t) for lv, _, _ in views.values() for t in lv if torch.is_tensor(t)}
        todo = []
        for a, (lv, spec, _) in views.items():
            cl, cspec = tree_flatten(getattr(c, a)); assert cspec == spec
            for t, s in zip(lv, cl):
                if torch.is_tensor(t) and s is not t: todo.append((t, s.clone() if id(s) in sid else s))
        for t, s in todo: t.copy_(s)

    def _step_triton(self, B, S):
        self._curB = B
        return super()._step_triton(B, S)

    def _if_side(self, B):
        if getattr(self, 'diag_skip_if', False): return
        v = self._if_views(B); self._if_bind(v, B); self.iftool.c.step(self.tokb[:B]); self._if_store(v)
        self._if_bind(v, B)

    def _if_write_step(self, B, hn, feat, cur, side_if):
        it = self.iftool; c = it.c; BM = self.B_max
        cur.wait_stream(side_if)
        if getattr(self, 'diag_skip_if', False):                  # diag: timing only
            self._if_tail2(B, hn, feat, torch.zeros(B, Q.D, device=self.dev)); return
        v = self._if_views(B); self._if_bind(v, B)
        C = B if self.ifC is None else min(self.ifC, B)
        h = self.xb[:B]
        if C >= B:
            wif = c.write(h).float(); self._if_store(v); self._if_bind(v, B)
        else:
            can = if_can_write(c)
            self.if_over |= can.sum() > C
            sel = torch.argsort((~can).to(torch.int8), stable=True)[:C]
            ws = if_write_subset(c, v, h, sel, C, self._if_bind)
            wif = torch.zeros(B, Q.D, device=self.dev); wif[sel] = ws
        if it.perm is not None: wif = wif[:, it.perm]
        wif = wif * self.tool_on[-1]
        # latched()/ans read the bound B-row views
        mw = self.mwire; ie = float(mw.eps)
        self.if_lat[:B] |= (torch.stack([x for x in c.L.values()], -1).amax(-1) > 0.5)
        self.if_ans[:B] |= (c.ans > 0.5)
        self._if_tail2(B, hn, feat, wif)

    def _if_tail2(self, B, hn, feat, wif):
        mw = self.mwire; ie = float(mw.eps)
        nzi = wif.abs().sum(-1) > 0
        j = self.NT - 1
        st_i = nzi & ~self.iw_prev[:B, j]
        lg_i = mw.k * 0.5 + mw.b[j] + hn @ mw.v[j]
        pfi = ie + (1 - 2 * ie) * torch.sigmoid(lg_i)
        fi = torch.where(self.greedy, lg_i > 0, torch.rand(B, device=self.dev) < pfi)
        supi = torch.where(st_i, ~fi, self.iw_sup[:B, j]) & nzi
        self._deci = torch.where(st_i, torch.where(fi, 1, 2), 0).to(torch.int8)
        self._lpdi = torch.where(st_i, torch.where(fi, torch.log(pfi), torch.log1p(-pfi)), torch.zeros_like(lg_i))
        self.iw_prev[:B, j] = nzi; self.iw_sup[:B, j] = supi
        wif = wif * (~supi)[:, None].to(wif.dtype)
        self.out_ifw_cur = wif
        fcur = wif / wif.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        feat['ifeval'] = fcur
        self.if_prev_f[:B].copy_(fcur)
        self._wif = wif; self._wfi = nzi & ~supi

    @torch.no_grad()
    def _final_E(self, live_n):
        """rows that finished at the block's last step have not consumed their final token: run ONE tools step on a
        snapshot of the tools state (restored afterwards) and record its outputs (the trainer's last-position feature)."""
        need = self.done[:live_n] & ~self.efin[:live_n]
        if not bool(need.any()): return
        snap = [t.clone() for t in self.TK._state_ids]
        E = self.TK.step(self.tokb, live_n)[:live_n]
        r, gl = self.rowid[:live_n], self.glen[:live_n]
        self.out_E[r[need], gl[need] + 1] = E[need]
        self.efin[:live_n] |= need
        for t, s_ in zip(self.TK._state_ids, snap): t.copy_(s_)

    # ---- capacity choice + block rollback
    IF_MARGIN = 8

    def _if_can_count(self, n):
        it = self.iftool; it._set(); c = it.c
        Lst = torch.stack([x for x in c.L.values()], -1)[:n]
        can = (Lst > 0).any(-1) | (c.prog[:n] != 0) | (c.hforced[:n] != 0)
        return int(can.sum())

    def _pick_C(self, n_can, B):
        want = n_can + self.IF_MARGIN
        C = next((b for b in B_BUCKETS if b >= want), B)
        return None if C >= B else C

    def _snap_list(self):
        outs = {id(t) for t in (self.out_tok, self.out_lp, self.out_w, self.out_wt, self.out_dec3, getattr(self, 'out_ifw', None)) if t is not None}
        return [t for t in self._mut() if id(t) not in outs]

    def _wave(self, qs, G, T, max_new, greedy):
        if self.iftool is None or not self.imp_on:
            self.ifC = None
            return super()._wave(qs, G, T, max_new, greedy)
        n = len(qs) * G; dev = self.dev; DUMMY = self.B_max
        for t in (self.out_lp, self.out_w, self.out_wt, self.out_dec3): t.zero_()
        self.out_tok.fill_(self.eos); self.out_ifw.zero_()
        rows = [torch.arange(p * G, (p + 1) * G, device=dev) for p in range(len(qs))]
        logits, lens, e_last = self._prefill(qs, rows)
        self.temp.fill_(T); self.greedy.fill_(greedy); self.maxnew.fill_(max_new)
        z = (logits / T).repeat_interleave(G, 0)
        first = z.argmax(-1) if greedy else (z - torch.log(-torch.log(torch.rand_like(z).clamp_(1e-20, 1.0)))).argmax(-1)
        self.out_tok[:n, 0] = first
        self.out_lp[:n, 0] = torch.log_softmax(z, -1).gather(-1, first[:, None])[:, 0]
        if self._e_last_all is not None:
            NTb = self._e_last_all.shape[1]
            self.out_wt[:n, 0, :NTb] = self._e_last_all.to(torch.int8).repeat_interleave(G, 0)
            self.out_w[:n, 0] = self.out_wt[:n, 0, 0]
        del z
        Bb = next(b for b in B_BUCKETS if b >= n)
        self.tokb[:n] = first; self.slot[:n] = lens.repeat_interleave(G); self.glen[:n] = 0
        self.rowid[:n] = torch.arange(n, device=dev)
        self.done[:n] = (first == self.eos) | ((first == 151645) if getattr(self, 'chat_stop', False) else self.has_nn[first]) | (max_new <= 1)
        self.tokb[n:Bb] = self.eos; self.slot[n:Bb] = 0; self.glen[n:Bb] = 0; self.rowid[n:Bb] = DUMMY; self.done[n:Bb] = True
        self.efin[:n] = False; self.efin[n:Bb] = True
        self.out_E[:n, 0] = self._Eq_wave[torch.arange(len(qs), device=dev).repeat_interleave(G), lens.repeat_interleave(G) - 1]
        maxlen = int(lens.max()); live_n = n; step = 0
        glen_final = torch.zeros(n + 1, dtype=torch.long, device=dev)
        lat_final = torch.zeros(n + 1, dtype=torch.bool, device=dev); ans_final = torch.zeros_like(lat_final)
        latp_final = torch.zeros_like(lat_final); latp_final[:n] = self.if_lat_prompt[:n]
        self.ifC = self._pick_C(self._if_can_count(live_n), Bb)
        self.if_over.zero_()
        while True:
            Bb = next(b for b in B_BUCKETS if b >= live_n)
            if self.ifC is not None and self.ifC >= Bb: self.ifC = None
            snap = [t.clone() for t in self._snap_list()]; seed0 = self.seed.clone()
            g = self._graph(Bb, self.S_max)
            for _ in range(self.sync_every): g.replay()
            if self.ifC is not None and bool(self.if_over):
                self.n_rollback += 1
                for t, s in zip(self._snap_list(), snap): t.copy_(s)
                self.iftool._set()
                C0 = self.ifC; self.ifC = None
                g = self._graph(Bb, self.S_max)
                for _ in range(self.sync_every): g.replay()
                self.if_over.zero_()
                self.ifC = self._pick_C(self._if_can_count(live_n), Bb)
            del snap
            self.ifC_hist.append(self.ifC or Bb)
            step += self.sync_every
            d = self.done[:live_n]
            n_live = int((~d).sum())
            if n_live == 0 or step >= max_new:
                self._final_E(live_n); break
            nb = next(b for b in B_BUCKETS if b >= n_live)
            if nb < Bb:
                self._final_E(live_n)
                idx = torch.nonzero(~d)[:, 0]; fin = torch.nonzero(d)[:, 0]
                rf = torch.clamp(self.rowid[fin], max=n)
                glen_final[rf] = self.glen[fin]
                lat_final[rf] = self.if_lat[fin]; ans_final[rf] = self.if_ans[fin]
                su = min(self.S_max, maxlen + step + 1)
                for i in range(Q.NL):
                    self.Kc[i][:n_live, :, :su] = self.Kc[i][idx, :, :su]
                    self.Vc[i][:n_live, :, :su] = self.Vc[i][idx, :, :su]
                for t in (self.tokb, self.slot, self.glen, self.rowid, self.done, self.efin):
                    t[:n_live] = t[idx].clone()
                self.TK.gather(idx, n_live)
                for t in (self.iw_h, self.iw_prev, self.iw_sup): t[:n_live] = t[idx].clone()
                for t in (self.if_prev_f, self.if_lat, self.if_ans): t[:n_live] = t[idx].clone()
                idx_full = torch.arange(self.B_max, device=dev); idx_full[:n_live] = idx
                self.iftool.gather(idx_full)
                self.done[n_live:nb] = True; self.rowid[n_live:nb] = DUMMY
                live_n = n_live
                self.ifC = self._pick_C(self._if_can_count(live_n), nb)
        rf = torch.clamp(self.rowid[:live_n], max=n)
        glen_final[rf] = self.glen[:live_n]
        lat_final[rf] = self.if_lat[:live_n]; ans_final[rf] = self.if_ans[:live_n]
        self._glen_final = glen_final[:n]; self._lat_final = lat_final[:n].cpu(); self._ans_final = ans_final[:n].cpu()
        self._latp_final = latp_final[:n].cpu()
        return step



class LibEngine2Refill(LibEngine2Lazy):
    """+ slot refill (continuous batching): one wave for the whole generate call; whenever >= refill_min rows are free,
    finished rows are compacted away and queued prompts are prefilled into the free slots.
    Priming of new rows (other rows keep decoding state untouched):
      * tools: cached tools state after the shared [256 x eos + chat prefix] stream, copied into scratch rows, then a
        scan over the question tokens only (rows sorted by length, the active set shrinks) -> states copied to the
        prompt's G rows. Same windows as the old prime (the receptive field K+W <= 192 lies inside the shared stream).
      * IFEval: the circuit has ONE step counter for all rows (prompt memory column = counter). New rows are primed in
        a temporary state whose counter starts at t_now - (Lp + Tq), so they end aligned with the live rows (the circuit
        is time-shift invariant: test_if_shift.py). The first group is primed the same way, from a base offset.
    Rows of one generate call therefore see the IF circuit at shifted absolute times; greedy outputs equal the
    non-refill engine only if the circuit is shift invariant (tested) and the host is batch-invariant (it is not:
    cuBLAS depends on the bucket size) -> refill is validated per component, not by whole-token equality."""
    T_BASE = 1024

    def __init__(self, *a, refill_min=32, **kw):
        G_max = kw.pop('G_max', 8)
        super().__init__(*a, tk_scratch=kw.pop('tk_scratch', 64), **kw)
        self.refill_min = refill_min; self.n_refills = 0

    # ---------------- tools priming for a subset of rows
    @torch.no_grad()
    def _tk_prefix_state(self):
        TK = self.TK; s0 = TK.scr0; dev = self.dev
        TK.reset_rows(torch.arange(s0, s0 + 1, device=dev))
        tb = torch.full((TK.Bp,), self.eos, dtype=torch.long, device=dev)
        for t in self.tool_prefix.tolist():
            tb[s0] = t; TK.step(tb, 1, r0=s0)
        self._tkp = [x.view(-1, TK.Bp)[:, s0].clone() for x in TK._state_ids]

    @torch.no_grad()
    def _tk_prime_rows(self, qids, rows):
        TK = self.TK; s0 = TK.scr0; dev = self.dev; P = len(qids)
        assert P <= TK.n_scr
        order = sorted(range(P), key=lambda i: -len(qids[i])); lens = [len(qids[i]) for i in order]
        for x, v in zip(TK._state_ids, self._tkp): x.view(-1, TK.Bp)[:, s0:s0 + P] = v[:, None]
        Tq = lens[0]
        feats = torch.zeros(P, Tq, TK.nout, device=dev)
        tb = torch.full((TK.Bp,), self.eos, dtype=torch.long, device=dev)
        qt = torch.full((P, Tq), self.eos, dtype=torch.long, device=dev)
        for k, i in enumerate(order): qt[k, :len(qids[i])] = torch.tensor(qids[i], device=dev)
        perm = torch.tensor(order, device=dev)
        for j in range(Tq):
            Bj = sum(1 for n in lens if n > j)
            tb[s0:s0 + Bj] = qt[:Bj, j]
            E = TK.step(tb, Bj, r0=s0)
            feats[perm[:Bj], j] = E[s0:s0 + Bj]
        src = torch.cat([torch.full((len(r),), s0 + order.index(p), device=dev, dtype=torch.long) for p, r in enumerate(rows)])
        TK.copy_rows(src, torch.cat(rows))
        return feats

    # ---------------- IF priming for a subset of rows (time-aligned)
    @torch.no_grad()
    def _if_prime_rows(self, qids, rows):
        it = self.iftool; c = it.c; BM = self.B_max; dev = self.dev; P = len(qids)
        it._set()
        t_now = int(c.t)
        Tq = max(map(len, qids)); t0 = t_now - self.Lp - Tq
        assert t0 >= 0, (t_now, Tq)
        Pb = next(b for b in (8, 16, 32, 64, 128, 256) if b >= P)
        c.reset(Pb); c.t.fill_(t0)
        for t in self.prefix_ids.tolist(): c.step(torch.full((Pb,), t, device=dev))      # eager (35 steps)
        ids = torch.full((Pb, Tq), self.eos, dtype=torch.long, device=dev)
        for p, q in enumerate(qids): ids[p, Tq - len(q):] = torch.tensor(q, device=dev)
        st, g, ibuf = self._if_tmp_graph(Pb)
        cur = {a: tree_flatten(getattr(c, a)) for a in it.state_names}
        for a, (leaves, spec) in st.items():                                           # load the state into the graph's leaves
            cl, cspec = cur[a]; assert cspec == spec, f'IF state structure differs for {a}'
            for x, y in zip(leaves, cl):
                if torch.is_tensor(x): x.copy_(y)
        for k in range(Tq): ibuf.copy_(ids[:, k]); g.replay()
        for a, (leaves, spec) in st.items(): setattr(c, a, tree_unflatten(leaves, spec))
        c.B = Pb
        assert int(c.t) == t_now
        lat = (torch.stack([v for v in c.L.values()], -1).amax(-1) > 0.5)[:P]
        tmp = {a: tree_flatten(getattr(c, a)) for a in it.state_names}
        src = torch.cat([torch.full((len(r),), p, device=dev, dtype=torch.long) for p, r in enumerate(rows)])
        dst = torch.cat(rows)
        for a, (leaves, spec) in it.static.items():
            tl, tspec = tmp[a]
            assert tspec == spec, f'IF state structure differs for {a}'
            for x, y in zip(leaves, tl):
                if torch.is_tensor(x) and x.dim() > 0 and x.shape[0] == BM: x[dst] = y[src]
        it._set()
        return lat[src]

    @torch.no_grad()
    def _if_tmp_graph(self, Pb):
        """CUDA graph of one IF circuit step on a Pb-row temporary state (captured on the post-prefix structure, which
        is stable from there on: the old engine's _if_prime relies on the same). Returns (static leaves, graph, ids)."""
        if not hasattr(self, '_if_tmp'): self._if_tmp = {}
        if Pb in self._if_tmp: return self._if_tmp[Pb]
        it = self.iftool; c = it.c
        st = {a: ([x.clone() if torch.is_tensor(x) else x for x in leaves], spec)
              for a, (leaves, spec) in ((a, tree_flatten(getattr(c, a))) for a in it.state_names)}
        ibuf = torch.full((Pb,), self.eos, dtype=torch.long, device=self.dev)
        sid = {id(x) for lv, _ in st.values() for x in lv if torch.is_tensor(x)}
        def body():
            for a, (lv, spec) in st.items(): setattr(c, a, tree_unflatten(lv, spec))
            c.B = Pb; c.step(ibuf)
            todo = []
            for a, (lv, spec) in st.items():
                cl, cspec = tree_flatten(getattr(c, a)); assert cspec == spec
                for x, y in zip(lv, cl):
                    if torch.is_tensor(x) and y is not x: todo.append((x, y.clone() if id(y) in sid else y))
            for x, y in todo: x.copy_(y)
        save = [x.clone() for lv, _ in st.values() for x in lv if torch.is_tensor(x)]
        s_ = torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s_):
            for _ in range(2): body()
        torch.cuda.current_stream().wait_stream(s_)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): body()
        for x, v in zip([x for lv, _ in st.values() for x in lv if torch.is_tensor(x)], save): x.copy_(v)
        self._if_tmp[Pb] = (st, g, ibuf)
        return self._if_tmp[Pb]

    @torch.no_grad()
    def _prefill_rows(self, qids, rows):
        """host prefill of new prompts into rows (KV, iw_h, first-token logits) + tools / IF priming of those rows only."""
        P = len(qids); T = max(map(len, qids)); dev = self.dev
        ids = torch.tensor([q + [self.eos] * (T - len(q)) for q in qids], device=dev)
        lens = torch.tensor([len(q) for q in qids], device=dev)
        prefix_kv = [(self.Kp[i], self.Vp[i]) for i in range(Q.NL)]
        links = self.links
        allrows = torch.cat(rows); src = torch.cat([torch.full((len(r),), p, device=dev) for p, r in enumerate(rows)])
        fz = self._tk_prime_rows(qids, rows)
        fs_ = [fz[:, :, o:o + n] * self.tool_on[i] for i, (o, n) in enumerate(self.TK.outoff)]
        if fz.shape[1] < T: pass
        feat = {'calc': fs_[0]}
        for j, t in enumerate(self.tools): feat[t.name] = fs_[j + 1]
        ar = torch.arange(P, device=dev)
        e_last = [f[ar, lens - 1] for f in fs_]
        write = torch.zeros(P, T, Q.D, device=dev)
        for i, t in enumerate(self.tools): write[ar, lens - 1] += e_last[i + 1] @ t.D
        feat['ifeval_prev'] = torch.zeros(P, T, Q.D, device=dev); feat['ifeval'] = feat['ifeval_prev']
        e_last_all = torch.stack([e.abs().sum(-1) > 0 for e in e_last], -1)
        logits = torch.empty(P, self.W['E'].shape[0], device=dev)
        iwc = self.mwire.L
        PC = getattr(self, 'prefill_chunk', 64)
        for c0 in range(0, P, PC):
            sl = slice(c0, c0 + PC)
            r_ = Q.fwd_chunk(self.W, ids[sl], self.Lp, self.cos_t, self.sin_t, prefix_kv=prefix_kv, links=links,
                             feat={k: v[sl] for k, v in feat.items()}, final_write=write[sl], return_kv=True, capture=iwc)
            hn, kvs = r_[0], r_[1]; n = hn.shape[0]
            logits[c0:c0 + n] = F.linear(hn[torch.arange(n, device=dev), lens[sl] - 1], self.W['E']).float()
            sel = (src >= c0) & (src < c0 + n); rr, ss = allrows[sel], src[sel] - c0
            for i, (k, v) in enumerate(kvs):
                self.Kc[i][rr, :, :T] = k[ss]; self.Vc[i][rr, :, :T] = v[ss]
            hc = r_[2][torch.arange(n, device=dev), lens[sl] - 1]
            self.iw_h[rr] = hc[ss].to(self.iw_h.dtype)
            del hn, kvs, r_
        NTb = 1 + len(self.tools)
        self.iw_prev[allrows] = False; self.iw_sup[allrows] = False
        self.iw_prev[allrows, :NTb] = e_last_all[src]
        latp = self._if_prime_rows(qids, rows)
        self.if_prev_f[allrows] = 0; self.if_lat[allrows] = latp; self.if_ans[allrows] = False
        return logits, lens, e_last_all, fz, latp

    # ---------------- outputs sized by the whole call
    def _alloc_out_n(self, n_rows, max_new):
        need = (n_rows + 1, max_new + 2)
        if getattr(self, '_outshape_n', None) != need:
            z = lambda *s, dt: torch.zeros(*s, device=self.dev, dtype=dt)
            self.out_tok = z(*need, dt=torch.long); self.out_lp = z(*need, dt=torch.float32); self.out_w = z(*need, dt=torch.int8)
            self.out_wt = z(*need, self.NT, dt=torch.int8); self.out_dec3 = z(*need, self.NT, dt=torch.int8)
            self.out_ifw = z(*need, Q.D, dt=torch.float32); self.out_E = z(*need, self.TK.nout, dt=torch.float32)
            self.efin = torch.zeros(self.B_max + 1, device=self.dev, dtype=torch.bool)
            self._outshape_n = need; self._outshape = None; self.graphs = {}; self.pool = None

    @torch.no_grad()
    def generate(self, qids, G=1, T=1.0, max_new=320, greedy=False):
        import time
        if not self.imp_on: return super().generate(qids, G, T, max_new, greedy)
        assert max(map(len, qids)) + max_new <= self.S_max
        dev = self.dev; NP = len(qids); n_tot = NP * G; DUMMY = n_tot
        self._alloc_out_n(n_tot, max_new)
        for t_ in (self.out_lp, self.out_w, self.out_wt, self.out_dec3, self.out_ifw, self.out_E): t_.zero_()
        self.out_tok.fill_(self.eos)
        torch.cuda.synchronize(); t0w = time.time()
        self.temp.fill_(T); self.greedy.fill_(greedy); self.maxnew.fill_(max_new)
        self._tk_prefix_state()
        it = self.iftool
        if self.if_prefix_state is None: self._if_prefix()
        # static state = the prefix-stepped state expanded to B_max rows (same pytree structure as after steps)
        c = it.c
        for a, (leaves, spec) in self.if_prefix_state.items():
            setattr(c, a, tree_unflatten([l.clone() if torch.is_tensor(l) else l for l in leaves], spec))
        c.B = 1
        from lib_engine import gather_tree
        idx0 = torch.zeros(self.B_max, dtype=torch.long, device=dev)
        for a in it.state_names: setattr(c, a, gather_tree(getattr(c, a), idx0, 1))
        c.B = self.B_max; it.to_static()
        it._set(); it.c.t.fill_(self.T_BASE + self.Lp + max(map(len, qids)))
        Eq_all = [None] * NP; latp_all = torch.zeros(n_tot + 1, dtype=torch.bool, device=dev)
        glen_final = torch.zeros(n_tot + 1, dtype=torch.long, device=dev)
        lat_final = torch.zeros(n_tot + 1, dtype=torch.bool, device=dev); ans_final = torch.zeros_like(lat_final)
        queue = list(range(NP)); live_n = 0; steps = 0; self.n_refills = 0
        cap = (self.B_max // G)

        def admit(n_free):
            nonlocal live_n
            k = min(len(queue), n_free // G, self.TK.n_scr)
            if k == 0: return
            ps = [queue.pop(0) for _ in range(k)]
            qs = [qids[p] for p in ps]
            rows = [torch.arange(live_n + i * G, live_n + (i + 1) * G, device=dev) for i in range(k)]
            logits, lens, e_last, fz, latp = self._prefill_rows(qs, rows)
            z = (logits / T).repeat_interleave(G, 0)
            first = z.argmax(-1) if greedy else (z - torch.log(-torch.log(torch.rand_like(z).clamp_(1e-20, 1.0)))).argmax(-1)
            gid = torch.tensor([p * G + g for p in ps for g in range(G)], device=dev)
            sl = slice(live_n, live_n + k * G)
            self.out_tok[gid, 0] = first
            self.out_lp[gid, 0] = torch.log_softmax(z, -1).gather(-1, first[:, None])[:, 0]
            NTb = e_last.shape[1]
            self.out_wt[gid, 0, :NTb] = e_last.to(torch.int8).repeat_interleave(G, 0)
            self.out_w[gid, 0] = self.out_wt[gid, 0, 0]
            self.out_E[gid, 0] = fz[torch.arange(k, device=dev), lens - 1].repeat_interleave(G, 0)
            for i, p in enumerate(ps): Eq_all[p] = fz[i, :len(qids[p])].cpu()
            latp_all[gid] = latp
            self.tokb[sl] = first; self.slot[sl] = lens.repeat_interleave(G); self.glen[sl] = 0; self.rowid[sl] = gid
            self.done[sl] = (first == self.eos) | ((first == 151645) if getattr(self, 'chat_stop', False) else self.has_nn[first]) | (max_new <= 1)
            self.efin[sl] = False
            live_n += k * G; self.n_refills += 1

        admit(cap * G)
        self.if_over.zero_()
        while True:
            Bb = next(b for b in B_BUCKETS if b >= live_n)
            self.tokb[live_n:Bb] = self.eos; self.slot[live_n:Bb] = 0; self.glen[live_n:Bb] = 0
            self.rowid[live_n:Bb] = DUMMY; self.done[live_n:Bb] = True; self.efin[live_n:Bb] = True
            self.ifC = self._pick_C(self._if_can_count(live_n), Bb)
            if self.ifC is not None and self.ifC >= Bb: self.ifC = None
            snap = [t.clone() for t in self._snap_list()]
            g = self._graph(Bb, self.S_max)
            for _ in range(self.sync_every): g.replay()
            if self.ifC is not None and bool(self.if_over):
                self.n_rollback += 1
                for t, s_ in zip(self._snap_list(), snap): t.copy_(s_)
                self.iftool._set(); self.ifC = None
                g = self._graph(Bb, self.S_max)
                for _ in range(self.sync_every): g.replay()
                self.if_over.zero_()
            del snap
            steps += self.sync_every
            d = self.done[:live_n]
            n_alive = int((~d).sum())
            free = self.B_max - n_alive
            if n_alive == 0 and not queue: self._final_E(live_n); fin_all = True
            else: fin_all = False
            need_compact = fin_all or (queue and free >= max(self.refill_min, G)) or \
                (next(b for b in B_BUCKETS if b >= max(n_alive, 1)) < Bb)
            if need_compact or fin_all:
                self._final_E(live_n)
                idx = torch.nonzero(~d)[:, 0]; fin = torch.nonzero(d)[:, 0]
                rf = torch.clamp(self.rowid[fin], max=n_tot)
                glen_final[rf] = self.glen[fin]; lat_final[rf] = self.if_lat[fin]; ans_final[rf] = self.if_ans[fin]
                if fin_all: break
                su = int(self.slot[:live_n].max()) + 1
                for i in range(Q.NL):
                    self.Kc[i][:n_alive, :, :su] = self.Kc[i][idx, :, :su]
                    self.Vc[i][:n_alive, :, :su] = self.Vc[i][idx, :, :su]
                for t in (self.tokb, self.slot, self.glen, self.rowid, self.done, self.efin):
                    t[:n_alive] = t[idx].clone()
                self.TK.gather(idx, n_alive)
                for t in (self.iw_h, self.iw_prev, self.iw_sup, self.if_prev_f, self.if_lat, self.if_ans): t[:n_alive] = t[idx].clone()
                idx_full = torch.arange(self.B_max, device=dev); idx_full[:n_alive] = idx
                self.iftool.gather(idx_full)
                live_n = n_alive
                if queue and self.B_max - live_n >= G: admit(self.B_max - live_n)
            if steps > 100000: raise RuntimeError('runaway')
        res = dict(tokens=[], logp=[], wflag=[], dec=[], wt=[], ifw=[], if_lat=[], if_lat_prompt=[], if_ans=[], E=[], Eq=[])
        gl = glen_final.cpu(); ot, ol, ow = self.out_tok[:n_tot].cpu(), self.out_lp[:n_tot].cpu(), self.out_w[:n_tot].cpu()
        od, owt, oE = self.out_dec3[:n_tot].cpu(), self.out_wt[:n_tot].cpu(), self.out_E[:n_tot].cpu()
        latc, ansc, latpc = lat_final.cpu(), ans_final.cpu(), latp_all.cpu()
        for j in range(n_tot):
            L = int(gl[j]) + 1
            res['tokens'].append(ot[j, :L].tolist()); res['logp'].append(ol[j, :L].tolist()); res['wflag'].append(ow[j, :L].tolist())
            res['dec'].append(od[j, :L].clone()); res['wt'].append(owt[j, :L].clone())
            res['E'].append(oE[j, :L + 1].clone()); res['Eq'].append(Eq_all[j // G])
            w_ = self.out_ifw[j, :L]; nzp = (w_.abs().sum(-1) > 0).nonzero()[:, 0]
            res['ifw'].append((nzp.cpu(), w_[nzp].cpu()))
            res['if_lat'].append(bool(latc[j])); res['if_lat_prompt'].append(bool(latpc[j])); res['if_ans'].append(bool(ansc[j]))
        torch.cuda.synchronize(); dt = time.time() - t0w
        ntok = sum(map(len, res['tokens']))
        self.stats = dict(seconds=dt, tokens=ntok, tok_s=ntok / dt, decode_steps=steps, rows=n_tot, refills=self.n_refills)
        res['stats'] = self.stats
        return res


def install_ln_memo(G):
    """Rollout only (no grad): the 10 CWOut adapters of a layer each compute LN(h) of the SAME h; compute it once.
    Same op on the same input -> bitwise identical. Under autograd the original per-wire LN is used (a shared node
    would sum the 10 gradients before the LN backward: different fp32 order)."""
    if G.get('_ln_memo_installed'): return
    orig = G['_ln']; cache = {}

    def _ln(h):
        if torch.is_grad_enabled(): return orig(h)
        key = (h.data_ptr(), h._version, tuple(h.shape), h.dtype, torch.cuda.current_stream().cuda_stream)
        v = cache.get(key)
        if v is None:
            if len(cache) > 8: cache.clear()
            v = cache[key] = orig(h)
        return v
    G['_ln'] = _ln; G['_ln_memo_installed'] = True
