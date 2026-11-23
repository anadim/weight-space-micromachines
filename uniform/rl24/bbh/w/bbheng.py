"""BBH fast rollout engine (rl24/bbh, task [24b]/[26]): Qwen3-0.6B host + the 8 frozen BBH micromachines (integ.TOOLS_V2,
calc veto) + optional compiled call gate, all inside ONE per-token CUDA graph, with continuous batching.

Same decoding as the BBH-27 harness (callgate/code/cg_gate_gen.generate, mode 'none' / cg_compiled 'compiled'):
  prompt = the task's 3-shot CoT + "Q: ..\nA: Let's think step by step." (no shared prefix: Lp = 0, whole prompt per row),
  eos masked for the first `min_tokens` tokens, stop at eos, at the budget, or when the output length is a multiple of 6
  and '\n\nQ' is in the decode of its last 12 tokens (host-side stop rule of the harness, applied at the sync points:
  the row is cut at that length; tokens generated after it are discarded).
Tools: Micromachine.step_sym math (lib_engine.BBHTool), reading a FROZEN snapshot E0 of the input embedding (the tensor
their readers were solved on), stepped on side streams concurrently with the host layers; their writes
sum_i (e_i @ out_dirs_i) * gain are added to the residual right before the final norm (= micromachines.Attached).
Call gate: cg_compiled.CompiledCallGate math on the LM head output (fixed tensors + a 1-float latch register per row).
Nothing on the host side decides anything but sampling / budget / stop.

Continuous batching: a queue of rows (prompt p, sample g); free slots are refilled (prefill of the prompt into that slot's
KV rows, per-slot reset of every register: tool states primed from the prompt tail, latch 0); finished slots are
harvested and parked; host sync every K graph replays; optional compaction to smaller buckets when the queue is empty.
Prefill is per prompt (batch 1, no padding), so a row's prefill never depends on the batch it is admitted with.
"""
import time, torch, triton, torch.nn.functional as F
import fastqwen as Q
import hostkern as HK

B_BUCKETS = [8, 16, 32, 64, 96, 128, 192, 256, 384, 512]
TC = 151657                                   # <tool_call>
K_STEP = float(2 ** 24); BIGM = 1.0e6; NEG = -1.0e9


class DenseStack(torch.nn.Module):
    """lib_engine.DenseStack: a deep narrow compiled ReLUNet applied with dense per-layer matrices on a transposed wire
    buffer (graph friendly); same integer arithmetic as the reference (fp64 for teamc_tools.F64 else fp32)."""
    def __init__(self, ref):
        super().__init__()
        self.ref = ref
        net = getattr(ref, 'net', ref); self.dt = torch.float64 if hasattr(ref, 'net') else torch.float32
        self.L = net.L; self.layers = []
        dev = net.b0.device
        for l in range(net.L + 1):
            idx = getattr(net, f'i{l}').to(dev)
            sh = getattr(net, f'shape{l}')
            Wd = torch.zeros(sh, dtype=self.dt, device=dev)
            wi = getattr(net, f'wi{l}'); Wd[wi[0], wi[1]] = getattr(net, f'wv{l}').to(self.dt)
            self.layers.append((idx, Wd.contiguous(), getattr(net, f'b{l}').to(self.dt)[:, None].contiguous()))
        self.n_units = net.n_units

    def forward(self, x):
        sh = x.shape; xf = x.reshape(-1, sh[-1]).to(self.dt)
        B, n_in = xf.shape
        wires = torch.empty(n_in + self.n_units, B, dtype=self.dt, device=xf.device)
        wires[:n_in] = xf.T
        p = n_in
        for idx, Wd, b in self.layers[:-1]:
            k = Wd.shape[0]
            torch.addmm(b, Wd, wires.index_select(0, idx), out=wires[p:p + k])
            wires[p:p + k].relu_()
            p += k
        idx, Wd, b = self.layers[-1]
        out = torch.addmm(b, Wd, wires.index_select(0, idx))
        return out.T.to(torch.float32).reshape(*sh[:-1], -1)


class BBHTool:
    """One frozen micromachine with static per-slot state (Micromachine.step_sym math)."""
    def __init__(self, name, M, B_max):
        self.name, self.M = name, M
        dev = M.reader_w.device
        self.K, self.Wn, self.NS, self.key = M.K, M.W, M.NS, M.key
        self.Kt = M.K + M.W
        with torch.no_grad():
            self.dS = M.stack1(torch.zeros(1, M.K * M.NS, device=dev)).shape[-1]
        self.out_dirs = M.out_dirs.float().contiguous(); self.gain = M.gain
        self.Dnz = (self.out_dirs.abs().sum(-1) > 0).float()       # output rows with a nonzero write
        Bp = B_max + 1
        self.sym_h = torch.zeros(Bp, M.K, M.NS, device=dev)
        self.S_h = torch.zeros(Bp, M.W, self.dS, device=dev)
        self.key_h = torch.zeros(Bp, M.W, device=dev)
        self.dist = torch.arange(M.W - 1, -1, -1, device=dev).float()
        self.fast1 = None
        net1 = getattr(M.stack1, 'net', M.stack1)
        if net1.L >= 200:
            self.fast1 = DenseStack(M.stack1)
            g = torch.Generator(device=dev).manual_seed(7)
            x = (torch.rand(512, 1, M.K * M.NS, device=dev, generator=g) < 0.15).float()
            self.fast1_check_rand = bool(torch.equal(M.stack1(x).float(), self.fast1(x)))

    @torch.no_grad()
    def check_exact(self, emb_seq):
        if self.fast1 is None: return True
        win = self.M.windows(self.M.symbols(emb_seq))
        return bool(torch.equal(self.M.stack1(win).float(), self.fast1(win)))

    def stack1(self, win):
        return self.fast1(win) if self.fast1 is not None else self.M.stack1(win).float()

    def step(self, emb, B):
        from micromachines.machine import BIG, GAMMA
        M = self.M
        s = M.symbols(emb)
        sym_h = torch.cat([self.sym_h[:B, 1:], s[:, None]], 1)
        win = sym_h.flip(1).reshape(B, 1, M.K * M.NS)
        S_h = torch.cat([self.S_h[:B, 1:], self.stack1(win)], 1)
        key_h = torch.cat([self.key_h[:B, 1:], s[:, M.key:M.key + 1]], 1)
        self.sym_h[:B].copy_(sym_h); self.S_h[:B].copy_(S_h); self.key_h[:B].copy_(key_h)
        C = torch.softmax(BIG * key_h - GAMMA * self.dist, -1)[:, None, :] @ S_h
        return M.stack2(torch.cat([win, C], -1))[:, 0].float()

    @torch.no_grad()
    def prime_many(self, emb_tail):
        """emb_tail (n, Kt, d) -> (last outputs e (n, n_out), state tuple) = Micromachine.prime_sym math with the
        DenseStack stack1 (integer circuit: identical values), batched over prompts, without touching any slot."""
        M = self.M
        sym = M.symbols(emb_tail)
        win = M.windows(sym)
        S = self.stack1(win)
        C = M.attend(S, sym[..., M.key])
        e = M.stack2(torch.cat([win, C], -1))[:, -1].float()
        return e, (sym[:, -M.K:].float(), S[:, -M.W:].float(), sym[:, -M.W:, M.key].float())

    def set_rows(self, rows, st):
        self.sym_h[rows] = st[0]; self.S_h[rows] = st[1]; self.key_h[rows] = st[2]

    def move(self, src, dst):
        for t in (self.sym_h, self.S_h, self.key_h): t[dst] = t[src]


class BBHEngine:
    def __init__(self, W, tok, tools=(), E0=None, gate=None, B_max=256, S_max=1792, K=8, min_tokens=4, compact=True):
        self.W, self.tok = W, tok
        dev = W['E'].device; self.dev = dev
        self.B_max = next(b for b in B_BUCKETS if b >= B_max); B_max = self.B_max
        self.S_max, self.K, self.min_tokens, self.compact = S_max, K, min_tokens, compact
        bf = W['E'].dtype
        self.cos_t, self.sin_t = Q.rope_tables(S_max + 8, dev, bf)
        Bp = B_max + 1                                         # row B_max: dummy output row of parked slots
        self.Kc = [torch.zeros(B_max, Q.KVH, S_max, Q.HD, device=dev, dtype=bf) for _ in range(Q.NL)]
        self.Vc = [torch.zeros(B_max, Q.KVH, S_max, Q.HD, device=dev, dtype=bf) for _ in range(Q.NL)]
        z = lambda *s, dt=torch.long: torch.zeros(*s, device=dev, dtype=dt)
        self.tokb, self.slot, self.glen, self.rowid = z(Bp), z(Bp), z(Bp), z(Bp)
        self.done = z(Bp, dt=torch.bool); self.latch = torch.zeros(Bp, device=dev)
        self.temp = torch.ones((), device=dev); self.greedy = torch.zeros((), device=dev, dtype=torch.bool)
        self.maxnew = torch.zeros((), device=dev, dtype=torch.long)
        self.eos = tok.convert_tokens_to_ids('<|endoftext|>')
        self.xb = torch.zeros(Bp, Q.D, device=dev, dtype=bf); self.hb = torch.zeros_like(self.xb)
        self.qb = torch.zeros(Bp, Q.H, Q.HD, device=dev, dtype=bf); self.ab = torch.zeros_like(self.qb)
        self.mp = torch.full((Bp * Q.H,), float('-inf'), device=dev)     # no shared prefix: empty online-softmax state
        self.lpf = torch.zeros(Bp * Q.H, device=dev); self.accp = torch.zeros(Bp * Q.H, Q.HD, device=dev)
        self.actb = torch.zeros(Bp, Q.FF, device=dev, dtype=bf)
        self.seed = torch.zeros((), device=dev, dtype=torch.int64)
        self.nxt_b = torch.zeros(Bp, device=dev, dtype=torch.long); self.lp_b = torch.zeros(Bp, device=dev)
        self.tools = list(tools); self.E0 = E0
        self.tools_on = len(self.tools) > 0
        if self.tools_on:
            deep = [i for i, t in enumerate(self.tools) if t.fast1 is not None]
            rest = [i for i in range(len(self.tools)) if i not in deep]
            self._groups = [[i] for i in deep] + ([rest] if rest else [])
            self._side = [torch.cuda.Stream() for _ in self._groups]
            self.Kt = max(t.Kt for t in self.tools)
        self.gate_on = gate is not None
        if self.gate_on:
            V = W['E'].shape[0]
            self.gw = gate['w'].to(dev).float(); self.gb = torch.tensor(float(gate['b']), device=dev)
            self.m_eos = torch.zeros(V, device=dev); self.m_eos[self.eos] = NEG
        self.graphs, self.pool = {}, None
        self._outshape = None
        # stop rule tables: a window can only contain '\n\nQ' if one of its tokens contains the byte 'Q'
        V = W['E'].shape[0]
        self.hasQ = [False] * V
        for i in range(len(tok)):
            if 'Q' in tok.convert_ids_to_tokens(i): self.hasQ[i] = True
        self.stats = {}; self.t_prime = 0.0; self.t_host_pre = 0.0

    # ------------------------------------------------------------------ decode step (graph body)
    def _gate(self, h, lg, B):
        """cg_compiled.CompiledCallGate._hook on fp32 logits lg (B, V) with LM-head input h (B, D); updates the latch."""
        s = h.float() @ self.gw + self.gb
        neg = torch.clamp(F.relu(-K_STEP * s), max=1.0)
        top = ((lg + self.m_eos).argmax(-1) == TC).float()
        ev = F.relu(top + neg - 1.0)
        lat = 1.0 - F.relu(1.0 - self.latch[:B] - ev)
        self.latch[:B] = lat
        x = lg[:, TC]; off = BIGM * (1.0 - lat)
        lg[:, TC] = x - (F.relu(x - off) - F.relu(-x - off)) + NEG * lat
        return lg

    def _tool_write(self, outs):
        write = wf = None
        for i, t in enumerate(self.tools):
            e = outs[i]
            w_ = (e @ t.out_dirs) * t.gain
            write = w_ if write is None else write + w_
            nz = (e.abs() @ t.Dnz) > 0
            wf = nz if wf is None else (wf | nz)
        return write, wf

    def _step(self, B):
        W, tok = self.W, self.tokb[:B]
        slot = self.slot[:B]
        emb = W['E'][tok]
        outs = {}
        if self.tools_on:
            emb0 = self.E0[tok]
            cur = torch.cuda.current_stream()
            for st_, grp in zip(self._side, self._groups):
                st_.wait_stream(cur)
                with torch.cuda.stream(st_):
                    for i in grp: outs[i] = self.tools[i].step(emb0, B)
        x, h, q, att = self.xb[:B], self.hb[:B], self.qb[:B], self.ab[:B]
        x.copy_(emb)
        pos = slot
        ieee = x.dtype == torch.float32
        sc = 1.0 / (Q.HD ** 0.5)
        delta = None
        for i, lw in enumerate(W['L']):
            HK.add_rms_kernel[(B,)](x, delta if delta is not None else x, lw['ln1'], h, D=Q.D, HAS_DELTA=delta is not None, eps=Q.EPS)
            qkv = F.linear(h, lw['Wqkv'])
            Kc, Vc = self.Kc[i], self.Vc[i]
            HK.qk_rope_kernel[(B,)](qkv, lw['qn'], lw['kn'], self.cos_t, self.sin_t, pos, q, Kc, Vc, slot,
                                    Kc.stride(0), Kc.stride(1), Kc.stride(2), H=Q.H, KVH=Q.KVH, HD=Q.HD, eps=Q.EPS)
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
        wf = None
        if self.tools_on:
            for st_ in self._side: cur.wait_stream(st_)
            write, wf = self._tool_write(outs)
            x += write.to(x.dtype)
        HK.add_rms_kernel[(B,)](x, x, W['lnf'], h, D=Q.D, HAS_DELTA=False, eps=Q.EPS)
        logits = F.linear(h, W['E'])
        if self.gate_on: logits = self._gate(h, logits.float(), B)
        em = (self.glen[:B] + 1) < self.min_tokens               # token index glen+1 is produced now
        logits[:, self.eos] = torch.where(em, torch.full_like(logits[:, self.eos], NEG), logits[:, self.eos])
        nxt, lp = self.nxt_b[:B], self.lp_b[:B]
        HK.sample_kernel[(B,)](logits, self.temp, self.greedy, self.seed, nxt, lp, logits.shape[1], BV=2048, num_warps=8)
        self.seed += 1
        self._book(B, nxt, lp, wf)

    def _book(self, B, nxt, lp, wf):
        slot = self.slot[:B]; done = self.done[:B]
        nxt = torch.where(done, torch.full_like(nxt, self.eos), nxt)
        live = ~done
        r, gl = self.rowid[:B], self.glen[:B]
        c = gl + 1
        self.out_tok[r, c] = torch.where(live, nxt, self.out_tok[r, c])
        self.out_lp[r, c] = torch.where(live, lp, self.out_lp[r, c])
        if wf is not None: self.out_w[r, c] = torch.where(live, wf.to(torch.int8), self.out_w[r, c])
        if self.gate_on: self.out_lat[r, c] = torch.where(live, (self.latch[:B] > 0.5).to(torch.int8), self.out_lat[r, c])
        stop = (nxt == self.eos) | (gl + 2 >= self.maxnew)
        self.glen[:B] += live.long()
        self.done[:B] |= stop
        self.slot[:B] = torch.where(live, torch.clamp(slot + 1, max=self.S_max - 1), slot)
        self.tokb[:B] = nxt

    def _mut(self):
        m = [self.tokb, self.slot, self.glen, self.done, self.latch, self.seed, self.out_tok, self.out_lp, self.out_w, self.out_lat]
        for t in self.tools: m += [t.sym_h, t.S_h, t.key_h]
        return m

    def _graph(self, B):
        key = B
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

    def _alloc_out(self, max_new):
        need = (self.B_max + 1, max_new + 2)
        if self._outshape != need:
            self._outshape = need
            self.out_tok = torch.full(need, self.eos, device=self.dev, dtype=torch.long)
            self.out_lp = torch.zeros(need, device=self.dev)
            self.out_w = torch.zeros(need, device=self.dev, dtype=torch.int8)
            self.out_lat = torch.zeros(need, device=self.dev, dtype=torch.int8)
            self.graphs = {}; self.pool = None

    # ------------------------------------------------------------------ prefill (host: one prompt at batch 1; tools batched)
    @torch.no_grad()
    def prefill_many(self, idss):
        """-> list of dict(kvs, T, logits (V,) fp32 raw, h (D,) LM-head input, tools (per-tool state), wf0)."""
        n = len(idss); ta = time.time()
        W0 = wf0 = None; sts = None
        if self.tools_on:
            es, sts = [], []
            for t in self.tools:
                tail = torch.tensor([([self.eos] * t.Kt + list(ids))[-t.Kt:] for ids in idss], device=self.dev)
                e, st = t.prime_many(self.E0[tail]); es.append(e); sts.append(st)
            W0, wf0 = self._tool_write(dict(enumerate(es)))
            wf0 = wf0.tolist()
        tb = time.time(); self.t_prime += tb - ta
        res = []
        for j, ids in enumerate(idss):
            T = len(ids)
            x = torch.tensor([ids], device=self.dev)
            write = None
            if self.tools_on:
                write = torch.zeros(1, T, Q.D, device=self.dev); write[0, T - 1] = W0[j]
            hn, kvs = Q.fwd_chunk(self.W, x, 0, self.cos_t, self.sin_t, final_write=write, return_kv=True)
            h = hn[0, T - 1]
            logits = F.linear(h[None], self.W['E']).float()[0]
            tl = [(st[0][j], st[1][j], st[2][j]) for st in sts] if self.tools_on else None
            res.append(dict(kvs=[(k[0], v[0]) for k, v in kvs], T=T, logits=logits, h=h, tools=tl,
                            wf0=bool(wf0[j]) if self.tools_on else False))
        self.t_host_pre += time.time() - tb
        return res

    def prefill_one(self, ids): return self.prefill_many([ids])[0]

    @torch.no_grad()
    def _admit(self, slots, pre, T_, greedy):
        """slots: list of slot ids all for ONE prompt (pre = its prefill_one result)."""
        dev = self.dev
        S = torch.tensor(slots, device=dev, dtype=torch.long); n = len(slots); T = pre['T']
        for i, (k, v) in enumerate(pre['kvs']):
            self.Kc[i][S, :, :T] = k; self.Vc[i][S, :, :T] = v
        if self.tools_on:
            for t, st in zip(self.tools, pre['tools']): t.set_rows(S, st)
        R = S                                                    # output row = slot at admission
        self.out_tok[R] = self.eos; self.out_lp[R] = 0; self.out_w[R] = 0; self.out_lat[R] = 0
        lg = pre['logits'][None].expand(n, -1).clone()
        self.latch[S] = 0.0
        if self.gate_on:
            hh = pre['h'][None].expand(n, -1)
            lat0 = self.latch[S].clone()
            s = hh.float() @ self.gw + self.gb
            neg = torch.clamp(F.relu(-K_STEP * s), max=1.0)
            top = ((lg + self.m_eos).argmax(-1) == TC).float()
            ev = F.relu(top + neg - 1.0)
            lat = 1.0 - F.relu(1.0 - lat0 - ev)
            self.latch[S] = lat
            x = lg[:, TC]; off = BIGM * (1.0 - lat)
            lg[:, TC] = x - (F.relu(x - off) - F.relu(-x - off)) + NEG * lat
            self.out_lat[R, 0] = (lat > 0.5).to(torch.int8)
        if self.min_tokens > 0: lg[:, self.eos] = NEG
        z = lg / T_
        first = z.argmax(-1) if greedy else (z - torch.log(-torch.log(torch.rand_like(z).clamp_(1e-20, 1.0)))).argmax(-1)
        self.out_tok[R, 0] = first
        self.out_lp[R, 0] = torch.log_softmax(z, -1).gather(-1, first[:, None])[:, 0]
        if pre['wf0']: self.out_w[R, 0] = 1
        self.tokb[S] = first; self.slot[S] = T; self.glen[S] = 0; self.rowid[S] = R
        self.done[S] = (first == self.eos) | (int(self.maxnew) <= 1)

    def _move(self, src, dst):
        dev = self.dev
        S = torch.tensor(src, device=dev, dtype=torch.long); D = torch.tensor(dst, device=dev, dtype=torch.long)
        L = int(self.slot[S].max()) + 1
        for i in range(Q.NL):
            self.Kc[i][D, :, :L] = self.Kc[i][S, :, :L]; self.Vc[i][D, :, :L] = self.Vc[i][S, :, :L]
        for t in (self.tokb, self.slot, self.glen, self.done, self.latch, self.rowid): t[D] = t[S]
        for t in self.tools: t.move(S, D)
        self.done[S] = True; self.rowid[S] = self.B_max; self.slot[S] = 0

    # ------------------------------------------------------------------ generate
    @torch.no_grad()
    def generate(self, qids, G=1, T=1.0, max_new=600, greedy=False, wave=False, seed=None, B_use=None):
        """qids: list of prompt token lists. Returns per-row (row = prompt r//G, sample r%G) tokens, logp (old policy at
        temperature T, with the gate / eos masks applied), wflag (1 = a tool wrote at the step producing the token),
        lat (1 = gate latched at that step). wave=True: admit only into an empty batch (machinery check)."""
        assert max(map(len, qids)) + max_new <= self.S_max, (max(map(len, qids)), max_new, self.S_max)
        self._alloc_out(max_new)
        if seed is not None: torch.manual_seed(seed); self.seed.fill_(seed)
        self.temp.fill_(T); self.greedy.fill_(bool(greedy)); self.maxnew.fill_(max_new)
        dev = self.dev; Bb = self.B_max if B_use is None else next(b for b in B_BUCKETS if b >= B_use); DUMMY = self.B_max
        self.done[:] = True; self.slot[:] = 0; self.glen[:] = 0; self.rowid[:] = DUMMY; self.tokb[:] = self.eos
        n = len(qids) * G
        occ = [-1] * Bb; res = [None] * n; plen = {}; checked = {}
        pc = {}                                                # prompt -> prefill result (until all its rows admitted)
        left = {}
        self.t_prime = 0.0; self.t_host_pre = 0.0
        torch.cuda.synchronize(); t0 = time.time()
        nxt = 0; steps = 0; Bc = Bb; busy = 0; t_pre = 0.0; n_cmp = 0; bh = {}
        adm_min = max(1, Bb // 16)
        while True:
            free = [s for s in range(Bc) if occ[s] < 0]
            if nxt < n and free:
                k = (min(len(free), n - nxt) if len(free) == Bb else 0) if wave else \
                    (min(len(free), n - nxt) if (len(free) >= adm_min or n - nxt <= len(free)) else 0)
                if k:
                    ta = time.time()
                    sl = free[:k]; rows = list(range(nxt, nxt + k)); nxt += k
                    byp = {}
                    for s, r in zip(sl, rows): byp.setdefault(r // G, []).append(s); occ[s] = r
                    newp = [p for p in byp if p not in pc]
                    if newp:
                        for p, pre in zip(newp, self.prefill_many([qids[p] for p in newp])):
                            pc[p] = pre; left[p] = G; plen[p] = len(qids[p])
                    for p, ss in byp.items():
                        self._admit(ss, pc[p], T, greedy)
                        left[p] -= len(ss)
                        if left[p] == 0: del pc[p]
                    t_pre += time.time() - ta
            live = [s for s in range(Bc) if occ[s] >= 0]
            if not live and nxt >= n: break
            if self.compact and not wave and nxt >= n:
                nb = next(b for b in B_BUCKETS if b >= len(live))
                if nb < Bc:
                    src = [s for s in live if s >= nb]; dst = [s for s in range(nb) if occ[s] < 0][:len(src)]
                    if src:
                        self._move(src, dst)
                        for s, d in zip(src, dst): occ[d] = occ[s]; occ[s] = -1; checked[d] = checked.pop(s, 0)
                    Bc = nb; n_cmp += 1
            g = self._graph(Bc)
            for _ in range(self.K): g.replay()
            steps += self.K; busy += self.K * len(live); bh[Bc] = bh.get(Bc, 0) + self.K
            # host-side harness stop rule at the sync point: '\n\nQ' in the last 12 tokens at a length multiple of 6
            gl = self.glen[:Bc].tolist(); dl = self.done[:Bc].tolist(); rw = self.rowid[:Bc].tolist()
            ot = None; fin = []
            for s in range(Bc):
                if occ[s] < 0: continue
                L = gl[s] + 1                                   # tokens so far (incl. a final eos)
                c0 = checked.get(s, 0)
                cut = None
                lens = list(range(max(6, (c0 // 6 + 1) * 6), L + 1, 6))
                if lens:
                    if ot is None: ot = self.out_tok.cpu().numpy()
                    row = ot[rw[s]]
                    for m in lens:
                        ids = row[max(0, m - 12):m].tolist()
                        if any(self.hasQ[i] for i in ids) and '\n\nQ' in self.tok.decode(ids):
                            cut = m; break
                    checked[s] = lens[-1]
                if cut is not None: fin.append((s, cut))
                elif dl[s]: fin.append((s, L))
            if fin: self._harvest(fin, occ, res, checked)
        torch.cuda.synchronize(); dt = time.time() - t0
        tok_ = [r[0] for r in res]
        ntok = sum(map(len, tok_))
        self.stats = dict(seconds=dt, tokens=ntok, tok_s=ntok / dt, decode_steps=steps, rows=n, prefill_s=t_pre,
                          occupancy=busy / max(1, steps * Bb), compactions=n_cmp, bucket_steps=bh,
                          prime_s=self.t_prime, host_prefill_s=self.t_host_pre, graphs=len(self.graphs))
        return dict(tokens=tok_, logp=[r[1] for r in res], wflag=[r[2] for r in res], lat=[r[3] for r in res], stats=self.stats)

    def _harvest(self, fin, occ, res, checked):
        S = torch.tensor([s for s, _ in fin], device=self.dev, dtype=torch.long)
        R = self.rowid[S]
        ot, ol, ow, ola = self.out_tok[R].cpu(), self.out_lp[R].cpu(), self.out_w[R].cpu(), self.out_lat[R].cpu()
        for j, (s, L) in enumerate(fin):
            res[occ[s]] = (ot[j, :L].tolist(), ol[j, :L].tolist(), ow[j, :L].tolist(), ola[j, :L].tolist())
            occ[s] = -1; checked.pop(s, None)
        self.done[S] = True; self.slot[S] = 0; self.rowid[S] = self.B_max
