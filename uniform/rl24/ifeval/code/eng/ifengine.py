"""IFEval rollout engine (rl24/ifeval): the v5 fast decode step (engine/rel/v5/wiring/engine.py, Triton host kernels, shared-
prefix KV + static per-slot suffix KV, the WHOLE decode step incl. the implant captured as one CUDA graph per live-row bucket,
Gumbel-max sampler fused) + continuous batching (fastroll-style: finished rows harvested every `sync_every` steps, live rows
compacted to the front, queued prompts admitted into the freed slots with an exact prefill and a per-slot implant reset).

Implant (optional, GExec over the full 14-rule export): per token, in the order of f14/run_c.py --
  prompt:  gstep over every prompt token (incl. the chat-template tokens, no padding: masked per row);
           gwrite on the last prompt position's pre-norm residual -> widx0; logits of the first answer token from x + F[widx0]
  decode:  gstep(previous token) ; host layers ; gwrite(x) -> widx ; x += F[widx] (bf16) ; final norm ; LM head ; sample.
The write vector F[widx] is a row of a fixed table added to the residual stream; there is no host-side decision anywhere: the
sampler sees the logits of the host with the implant's residual write, nothing else.

Recorded per generated token t (row-major lists): tokens (incl. the stop token), logp (behaviour policy at temperature T),
widx (the implant's write row at the step that produced t; V = no write) -> the trainer adds F[widx] at the same positions.
"""
import time, torch, triton, torch.nn.functional as F
import fastqwen as Q
import hostkern as HK

B_BUCKETS = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 320, 384, 448, 512, 640, 768, 1024]
STOPS = (151645, 151643)                    # <|im_end|>, <|endoftext|>


class IFEngine:
    def __init__(self, W, tok, prefix_ids, imp=None, B_max=512, S_max=1536, N_cap=1024, max_new_cap=2048, sync_every=16,
                 admit_min=8):
        self.W, self.tok, self.imp = W, tok, imp
        dev = W['E'].device; self.dev = dev; bf = W['E'].dtype
        self.B_max = next(b for b in B_BUCKETS if b >= B_max)
        self.S_max, self.N_cap, self.sync_every, self.admit_min = S_max, N_cap, sync_every, admit_min
        self.prefix_ids = torch.tensor(prefix_ids, device=dev); self.Lp = len(prefix_ids)
        self.cos_t, self.sin_t = Q.rope_tables(self.Lp + S_max + 8, dev, bf)
        Bp = self.B_max + 1
        self.Kc = [torch.zeros(Bp, Q.KVH, S_max, Q.HD, device=dev, dtype=bf) for _ in range(Q.NL)]
        self.Vc = [torch.zeros(Bp, Q.KVH, S_max, Q.HD, device=dev, dtype=bf) for _ in range(Q.NL)]
        self.Kp = torch.zeros(Q.NL, Q.KVH, self.Lp, Q.HD, device=dev, dtype=bf); self.Vp = torch.zeros_like(self.Kp)
        z = lambda *s, dt=torch.long: torch.zeros(*s, device=dev, dtype=dt)
        self.tokb, self.slot, self.glen, self.rowid = z(Bp), z(Bp), z(Bp), z(Bp)
        self.done = z(Bp, dt=torch.bool)
        self.temp = torch.ones((), device=dev); self.greedy = torch.zeros((), device=dev, dtype=torch.bool)
        self.maxnew = torch.zeros((), device=dev, dtype=torch.long)
        self.eos = STOPS[1]; self.stops = torch.tensor(STOPS, device=dev)
        self.xb = torch.zeros(Bp, Q.D, device=dev, dtype=bf); self.hb = torch.zeros_like(self.xb)
        self.qb = torch.zeros(Bp, Q.H, Q.HD, device=dev, dtype=bf); self.ab = torch.zeros_like(self.qb)
        self.mp = torch.zeros(Bp * Q.H, device=dev); self.lpf = torch.zeros_like(self.mp)
        self.accp = torch.zeros(Bp * Q.H, Q.HD, device=dev)
        self.actb = torch.zeros(Bp, Q.FF, device=dev, dtype=bf)
        self.seed = torch.zeros((), device=dev, dtype=torch.int64)
        self.nxt_b = torch.zeros(Bp, device=dev, dtype=torch.long); self.lp_b = torch.zeros(Bp, device=dev)
        need = (N_cap + 1, max_new_cap + 2)                       # row N_cap = dummy row for padding / stale rows
        self.out_tok = torch.full(need, self.eos, device=dev, dtype=torch.long)
        self.out_lp = torch.zeros(need, device=dev)
        self.V = W['E'].shape[0]
        self.out_wid = torch.full(need, self.V, device=dev, dtype=torch.int32)
        self.max_new_cap = max_new_cap
        if imp is not None:
            self.st, self.mem = imp.new_state(Bp)
        self.imp_on = imp is not None
        self.graphs, self.pool = {}, None
        self.stats = {}; self.f_on_start = None
        self.refresh()

    def refresh(self):
        kv = Q.prefix_kv_of(self.W, self.prefix_ids, self.cos_t, self.sin_t)
        for i, (k, v) in enumerate(kv):
            self.Kp[i].copy_(k); self.Vp[i].copy_(v)

    # ------------------------------------------------------------------------------------------- decode step (graph body)
    def _step(self, B):
        W, tok, slot = self.W, self.tokb[:B], self.slot[:B]
        imp = self.imp if self.imp_on else None
        if imp is not None: imp.gstep(self.st[:B], self.mem[:B], tok)
        x, h, q, att = self.xb[:B], self.hb[:B], self.qb[:B], self.ab[:B]
        x.copy_(W['E'][tok])
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
        if imp is not None:
            widx, _ = imp.gwrite(self.st[:B], self.mem[:B], x)
            x += imp.F[widx].to(x.dtype)
        else:
            widx = None
        HK.add_rms_kernel[(B,)](x, x, W['lnf'], h, D=Q.D, HAS_DELTA=False, eps=Q.EPS)
        logits = F.linear(h, W['E'])
        nxt, lp = self.nxt_b[:B], self.lp_b[:B]
        HK.sample_kernel[(B,)](logits, self.temp, self.greedy, self.seed, nxt, lp, logits.shape[1], BV=2048, num_warps=8)
        self.seed += 1
        self._book(B, nxt, lp, widx)

    def _book(self, B, nxt, lp, widx):
        done = self.done[:B]
        nxt = torch.where(done, torch.full_like(nxt, self.eos), nxt)
        live = ~done
        r, gl = self.rowid[:B], self.glen[:B]
        self.out_tok[r, gl + 1] = torch.where(live, nxt, self.out_tok[r, gl + 1])
        self.out_lp[r, gl + 1] = torch.where(live, lp, self.out_lp[r, gl + 1])
        if widx is not None:
            self.out_wid[r, gl + 1] = torch.where(live, widx.to(torch.int32), self.out_wid[r, gl + 1])
        stop = (nxt == STOPS[0]) | (nxt == STOPS[1]) | (gl + 2 >= self.maxnew)
        self.glen[:B] += live.long()
        self.done[:B] |= stop
        self.slot[:B] = torch.clamp(self.slot[:B] + 1, max=self.S_max - 1)
        self.tokb[:B] = nxt

    def _mut(self):
        m = [self.tokb, self.slot, self.glen, self.done, self.out_tok, self.out_lp, self.out_wid, self.seed]
        if self.imp is not None: m += [self.st, self.mem]
        return m

    def _graph(self, B):
        key = (B, self.imp_on)
        if key not in self.graphs:
            st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
            saved = [t.clone() for t in self._mut()]
            with torch.cuda.stream(st), torch.no_grad():
                for _ in range(2): self._step(B)
            torch.cuda.current_stream().wait_stream(st)
            g = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(g, pool=self.pool):
                self._step(B)
            if self.pool is None: self.pool = g.pool()
            for t, s in zip(self._mut(), saved): t.copy_(s)
            self.graphs[key] = g
        return self.graphs[key]

    # ------------------------------------------------------------------------------------------------- prompt states
    @torch.no_grad()
    def prompt_states(self, full_ids):
        """implant registers / memory after stepping each prompt's full token list (template included), masked per row."""
        P = len(full_ids); T = max(map(len, full_ids))
        st, mem = self.imp.new_state(P)
        ids = torch.tensor([q + [self.eos] * (T - len(q)) for q in full_ids], device=self.dev)
        lens = torch.tensor([len(q) for q in full_ids], device=self.dev)
        for t in range(T):
            self.imp.gstep(st, mem, ids[:, t], active=lens > t)
        return st, mem

    # ---------------------------------------------------------------------------------------------------- admission
    @torch.no_grad()
    def _admit(self, plist, base, G, qids, pst, pmem, T, greedy, gen):
        """prompts plist (indices into qids) -> slot rows base.. (G per prompt, prompt-major); output rows rowid = p*G+g."""
        P = len(plist); dev = self.dev
        sq = [qids[p] for p in plist]; Ts = max(map(len, sq))
        ids = torch.tensor([q + [self.eos] * (Ts - len(q)) for q in sq], device=dev)
        lens = torch.tensor([len(q) for q in sq], device=dev)
        pkv = [(self.Kp[i], self.Vp[i]) for i in range(Q.NL)]
        x, kvs = Q.fwd_chunk(self.W, ids, self.Lp, self.cos_t, self.sin_t, prefix_kv=pkv, return_kv=True, return_pre=True)
        ar = torch.arange(P, device=dev)
        xl = x[ar, lens - 1]
        rows = torch.arange(base, base + P * G, device=dev); src = ar.repeat_interleave(G)
        for i, (k, v) in enumerate(kvs):
            self.Kc[i][rows, :, :Ts] = k[src]; self.Vc[i][rows, :, :Ts] = v[src]
        del x, kvs
        widx = None
        if self.imp_on:
            pi = torch.tensor(plist, device=dev)
            st, mem = pst[pi].clone(), pmem[pi].clone()
            widx, _ = self.imp.gwrite(st, mem, xl)
            xl = xl + self.imp.F[widx].to(xl.dtype)
            self.st[rows] = st[src]; self.mem[rows] = mem[src]
        logits = F.linear(Q.rms(xl, self.W['lnf']), self.W['E']).float()
        z = (logits / T)[src]
        if greedy: first = z.argmax(-1)
        else:
            u = torch.rand(z.shape, device=dev, generator=gen).clamp_(1e-20, 1.0)
            first = (z - torch.log(-torch.log(u))).argmax(-1)
        lp = torch.log_softmax(z, -1).gather(-1, first[:, None])[:, 0]
        rid = (torch.tensor(plist, device=dev) * G)[:, None] + torch.arange(G, device=dev)[None]
        rid = rid.reshape(-1)
        self.out_tok[rid] = self.eos; self.out_lp[rid] = 0; self.out_wid[rid] = self.V
        self.out_tok[rid, 0] = first; self.out_lp[rid, 0] = lp
        if widx is not None: self.out_wid[rid, 0] = widx[src].to(torch.int32)
        self.tokb[rows] = first; self.slot[rows] = lens[src]; self.glen[rows] = 0; self.rowid[rows] = rid
        self.done[rows] = (first == STOPS[0]) | (first == STOPS[1]) | (self.maxnew <= 1)

    def _compact(self, live_n):
        d = self.done[:live_n]
        idx = torch.nonzero(~d)[:, 0]; fin = torch.nonzero(d)[:, 0]
        n = idx.numel()
        self._glen_final[torch.clamp(self.rowid[fin], max=self.N_cap)] = self.glen[fin]
        if n and not torch.equal(idx, torch.arange(n, device=self.dev)):
            su = int(self.slot[idx].max()) + 1
            for i in range(Q.NL):
                self.Kc[i][:n, :, :su] = self.Kc[i][idx, :, :su]
                self.Vc[i][:n, :, :su] = self.Vc[i][idx, :, :su]
            for t in (self.tokb, self.slot, self.glen, self.rowid, self.done):
                t[:n] = t[idx].clone()
            if self.imp is not None:
                self.st[:n] = self.st[idx].clone(); self.mem[:n] = self.mem[idx].clone()
        return n

    def _pad_rows(self, n, nb):
        """bucket padding rows [n, nb): finished dummies with a valid (reset) implant state."""
        if nb <= n: return
        self.done[n:nb] = True; self.rowid[n:nb] = self.N_cap; self.tokb[n:nb] = self.eos; self.slot[n:nb] = 0; self.glen[n:nb] = 0
        if self.imp is not None:
            self.st[n:nb] = self.imp.init; self.mem[n:nb] = 0

    # ----------------------------------------------------------------------------------------------------- generate
    @torch.no_grad()
    def generate(self, qids, G=1, T=1.0, max_new=1024, greedy=False, full_ids=None, seed=0):
        """qids: per prompt the token ids AFTER the shared prefix; full_ids: prefix + qids (only for the implant's prompt
        steps; default prefix_ids + q). Returns per row r = prompt r//G, sample r%G: tokens (incl. stop token), logp, widx."""
        dev = self.dev; N = len(qids) * G
        assert N <= self.N_cap and max_new <= self.max_new_cap
        assert max(map(len, qids)) + max_new <= self.S_max, (max(map(len, qids)), max_new, self.S_max)
        torch.cuda.synchronize(); t0 = time.time()
        if self.f_on_start is not None: self.f_on_start()
        gen = torch.Generator(device=dev); gen.manual_seed(seed); self.seed.fill_(seed * 1000003)
        self.temp.fill_(T); self.greedy.fill_(greedy); self.maxnew.fill_(max_new)
        pst = pmem = None
        if self.imp_on:
            pre = self.prefix_ids.tolist()
            fi = full_ids if full_ids is not None else [pre + q for q in qids]
            pst, pmem = self.prompt_states(fi)
        t_ps = time.time() - t0
        self._glen_final = torch.zeros(self.N_cap + 1, dtype=torch.long, device=dev)
        queue = list(range(len(qids))); live_n = 0; steps = 0; admits = 0; nsync = 0
        maxp = max(1, self.B_max // G)
        while True:
            free = (self.B_max - live_n) // G
            if queue and free > 0 and (free >= min(self.admit_min, len(queue)) or live_n == 0):
                k = min(free, len(queue), maxp); pl = queue[:k]; queue = queue[k:]
                self._admit(pl, live_n, G, qids, pst, pmem, T, greedy, gen); live_n += k * G; admits += 1
            if live_n == 0: break
            Bb = next(b for b in B_BUCKETS if b >= live_n)
            self._pad_rows(live_n, Bb)
            g = self._graph(Bb)
            for _ in range(self.sync_every): g.replay()
            steps += self.sync_every; nsync += 1
            if bool(self.done[:live_n].any()):
                live_n = self._compact(live_n)
        torch.cuda.synchronize(); dt = time.time() - t0
        gl = self._glen_final[:N].cpu(); ot = self.out_tok[:N].cpu(); ol = self.out_lp[:N].cpu(); ow = self.out_wid[:N].cpu()
        res_tok, res_lp, res_w = [], [], []
        for j in range(N):
            L = int(gl[j]) + 1
            res_tok.append(ot[j, :L].tolist()); res_lp.append(ol[j, :L].tolist()); res_w.append(ow[j, :L].tolist())
        ntok = sum(map(len, res_tok))
        self.stats = dict(seconds=dt, tokens=ntok, tok_s=ntok / dt, decode_steps=steps, rows=N, admits=admits,
                          prompt_state_s=t_ps)
        return dict(tokens=res_tok, logp=res_lp, widx=res_w, stats=self.stats)
