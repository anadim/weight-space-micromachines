"""Tool-library engine: rl_wiring Engine + 8 frozen BBH micromachines + the frozen IFEval 14-rule circuit, all stepped
INSIDE the engine's per-token CUDA graph, each tool behind its own learned in-wire veto (REINFORCE, exploration floor).

Tool index 0 = calc_s4 (the rl_wiring CalcImplant, fused kernel, reads the live host embedding exactly as CW-lean);
tools 1..8 = integ.TOOLS_V2 machines (read a FROZEN snapshot of the base input embedding, the tensor they were solved
against); last tool = IFEval circuit (circuit_g.Circuit; reads a FROZEN LM-head snapshot; its write needs the host's
final pre-norm state at the same position, so it is computed after the layers; its L14 adapter sees the PREVIOUS
position's write, the final adapter the same position's -- disclosed asymmetry, as in rl_envs).

Per tool i: burst start = output nonzero after a silent step; fire ~ Bernoulli(eps + (1-2eps) sigmoid(k/2 + b_i +
v_i . LN(h14[t-1]))); a vetoed burst writes nothing and feeds zeros to its adapters. Decision codes per tool
(1 fire / 2 veto) recorded in out_dec[r, t, i]; the summed decision log-prob is added to the token logprob.
`tool_on[i]` (0/1 buffer read inside the graphs) detaches tool i (leave-one-out) without recapture.
"""
import torch, triton, torch.nn.functional as F
import fastqwen as Q
import hostkern as HK
from engine import Engine, B_BUCKETS, S_STEP
from torch.utils._pytree import tree_flatten, tree_unflatten


def gather_tree(x, idx, B):
    if torch.is_tensor(x): return x.index_select(0, idx) if (x.dim() > 0 and x.shape[0] == B) else x
    if isinstance(x, tuple): return tuple(gather_tree(t, idx, B) for t in x)
    if isinstance(x, list): return [gather_tree(t, idx, B) for t in x]
    if isinstance(x, dict): return {k: gather_tree(v, idx, B) for k, v in x.items()}
    return x


def _lnorm(hp):
    hp = hp.float()
    return (hp - hp.mean(-1, keepdim=True)) * torch.rsqrt(hp.var(-1, unbiased=False, keepdim=True) + 1e-5)


class DenseStack(torch.nn.Module):
    """Speed only: a compiled ReLUNet (deep, narrow: expr_gen 667 / bool_gen 850 layers) applied with DENSE per-layer
    matrices on a transposed wire buffer (wires (N, B); each layer = index_select + addmm(out=rows) + relu_), CUDA-graph
    friendly, instead of cuSPARSE CSR products. Same integer arithmetic (dtype of the reference: fp64 for teamc_tools.F64,
    else fp32, no TF32); checked torch.equal against the reference stack at build time."""
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

    def forward(self, x):                      # x (B, 1, n_in) or (B, n_in)
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
    """One frozen micromachine with static per-row state (same math as Micromachine.step_sym)."""
    def __init__(self, name, M, B_max, perm=None):
        self.name, self.M = name, M
        dev = M.reader_w.device
        self.K, self.Wn, self.NS, self.key = M.K, M.W, M.NS, M.key
        self.Kt = M.K + M.W
        with torch.no_grad():
            self.dS = M.stack1(torch.zeros(1, M.K * M.NS, device=dev)).shape[-1]
        self.n_out = M.out_dirs.shape[0]
        D = (M.out_dirs * M.gain).float()
        self.perm = perm
        if perm is not None: D = D[perm]                    # scrambled control: output rows deranged
        self.D = D.contiguous()
        self.sym_h = torch.zeros(B_max + 1, M.K, M.NS, device=dev)
        self.S_h = torch.zeros(B_max + 1, M.W, self.dS, device=dev)
        self.key_h = torch.zeros(B_max + 1, M.W, device=dev)
        self.dist = torch.arange(M.W - 1, -1, -1, device=dev).float()
        self.fast1 = None
        net1 = getattr(M.stack1, 'net', M.stack1)
        if net1.L >= 200:                                   # deep narrow stack: dense graph-friendly path (exactness checked)
            self.fast1 = DenseStack(M.stack1)
            g = torch.Generator(device=dev).manual_seed(7)
            x = (torch.rand(512, 1, M.K * M.NS, device=dev, generator=g) < 0.15).float()
            ref = M.stack1(x).float(); got = self.fast1(x)
            self.fast1_check_rand = bool(torch.equal(ref, got))     # random 0/1 windows (may exceed design ranges)

    @torch.no_grad()
    def check_exact(self, emb_seq):
        """hard check on real windows: emb_seq (1, T, d) -> torch.equal(dense, reference) on every position"""
        if self.fast1 is None: return True
        win = self.M.windows(self.M.symbols(emb_seq))
        ok = bool(torch.equal(self.M.stack1(win).float(), self.fast1(win)))
        assert ok, f'DenseStack mismatch on real windows: {self.name}'
        return ok

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
    def prime_rows(self, emb_tail, rows):
        M = self.M
        e = M.prime_sym(M.symbols(emb_tail))
        sym, S, key = M.state
        self.sym_h[rows] = sym.float(); self.S_h[rows] = S.float(); self.key_h[rows] = key.float()
        return e[:, -1].float()

    @torch.no_grad()
    def full(self, emb):
        e, _ = self.M.emit(emb)
        return e.float()

    def gather(self, idx, n):
        for t in (self.sym_h, self.S_h, self.key_h):
            t[:n] = t[idx]


class IFTool:
    """Team D's frozen IFEval circuit (circuit_g, graph-safe), state held at B_max rows in static leaves; stepped for all
    B_max rows inside every bucket graph (rows >= B are dead and ignored)."""
    def __init__(self, circ, B_max, d, perm=None):
        self.c, self.B_max, self.name, self.perm = circ, B_max, 'ifeval', perm
        self.n_out = d
        before = set(vars(circ)); circ.reset(3)
        self.state_names = [k for k in vars(circ) if k not in before and k != 'B']
        self.static = None

    def _flat(self):
        return {a: tree_flatten(getattr(self.c, a)) for a in self.state_names}

    def _set(self):
        for a, (leaves, spec) in self.static.items(): setattr(self.c, a, tree_unflatten(leaves, spec))

    def _copy_back(self):
        cur = self._flat()
        sid = {id(t) for leaves, _ in self.static.values() for t in leaves if torch.is_tensor(t)}
        todo = []
        for a, (leaves, spec) in self.static.items():
            cl, cspec = cur[a]; assert cspec == spec, f'circuit state structure changed for {a}'
            for t, s in zip(leaves, cl):
                if torch.is_tensor(t) and s is not t: todo.append((t, s.clone() if id(s) in sid else s))
        for t, s in todo: t.copy_(s)

    @torch.no_grad()
    def reset_eager(self, B):
        self.c.reset(B)

    @torch.no_grad()
    def eager_step(self, ids):
        self.c.step(ids)

    @torch.no_grad()
    def to_static(self):
        """load the circuit's current eager state (B_max rows) into the static leaves (allocated on first call)."""
        cur = self._flat()
        if self.static is None:
            self.static = {a: ([t.clone() for t in leaves], spec) for a, (leaves, spec) in cur.items()}
        else:
            for a, (leaves, spec) in self.static.items():
                cl, cspec = cur[a]; assert cspec == spec
                for t, v in zip(leaves, cl):
                    if v is not t: t.copy_(v)
        self._set(); self.c.B = self.B_max

    def if_step(self, ids):
        self._set(); self.c.step(ids)

    def if_write(self, h):
        w = self.c.write(h).float(); self._copy_back()
        if self.perm is not None: w = w[:, self.perm]
        return w

    def step_write(self, ids, h):
        """graph body: ids (B_max,), h (B_max, d) pre-norm final state -> write (B_max, d) fp32"""
        self._set(); self.c.step(ids); w = self.c.write(h).float(); self._copy_back()
        if self.perm is not None: w = w[:, self.perm]
        return w

    def latched(self):
        L = self.c.L
        return torch.stack([v for v in L.values()], -1).amax(-1) > 0.5          # (B_max,) any rule latched

    @torch.no_grad()
    def gather(self, idx_full):
        """idx_full: LongTensor (B_max,) source row for every destination row."""
        for a, (leaves, _) in self.static.items():
            for t in leaves:
                if torch.is_tensor(t) and t.dim() > 0 and t.shape[0] == self.B_max:
                    t.copy_(t[idx_full])
        self._set()


class LibEngine(Engine):
    def __init__(self, W, prefix_ids, tok, implant, tools, iftool, E0, links=None, B_max=128, S_max=544):
        self.tools, self.iftool, self.E0 = tools, iftool, E0
        super().__init__(W, prefix_ids, tok, implant=implant, links=links, B_max=B_max, S_max=S_max, use_write=False)
        self.NT = 1 + len(tools) + (1 if iftool is not None else 0)
        dev = self.dev
        self.tool_on = torch.ones(self.NT, device=dev)
        self.mwire = None
        self.if_prev_f = torch.zeros(self.B_max + 1, Q.D, device=dev)       # previous position's (vetoed) normalized IF write
        self.if_lat = torch.zeros(self.B_max + 1, device=dev, dtype=torch.bool)
        self.if_ans = torch.zeros(self.B_max + 1, device=dev, dtype=torch.bool)
        self.if_prefix_state = None
        # tool/calc features see [256 x <|endoftext|>] + prefix (a short chat prefix is shorter than a tool's K+W window;
        # decode priming and teacher forcing use the same padded stream, so they agree)
        self.tool_prefix = torch.cat([torch.full((512,), self.eos, device=self.dev, dtype=torch.long), self.prefix_ids])
        self.Lt = len(self.tool_prefix)
        self.exact_checks = {t.name: (t.check_exact(E0[self.prefix_ids][None]), getattr(t, 'fast1_check_rand', None))
                             for t in tools if t.fast1 is not None}
        print('DenseStack exactness (real prefix windows, random windows):', self.exact_checks, flush=True)

    def set_mwire(self, mw):
        """mw: MultiInWire (.L, .k, .b (NT,), .v (NT, D), .eps)."""
        self.mwire = mw; self.inwire = mw; dev = self.dev; Bp = self.B_max + 1
        self.iw_h = torch.zeros(Bp, Q.D, device=dev, dtype=self.W['E'].dtype)
        self.iw_prev = torch.zeros(Bp, self.NT, device=dev, dtype=torch.bool)
        self.iw_sup = torch.zeros(Bp, self.NT, device=dev, dtype=torch.bool)
        self.graphs = {}; self._outshape = None

    # ------------------------------------------------------------ in-wire for all tools at once
    def _mw_decide(self, nz, B, hn):
        """nz (B, NT) bool tool output nonzero. Returns keep (B,NT) bool, dec (B,NT) int8, lpd (B,)"""
        mw = self.mwire
        start = nz & ~self.iw_prev[:B]
        logit = mw.k * 0.5 + mw.b[None] + hn @ mw.v.T                     # (B, NT)
        u = torch.rand(B, self.NT, device=self.dev)
        ie = float(mw.eps)
        pf = ie + (1 - 2 * ie) * torch.sigmoid(logit)
        fire = torch.where(self.greedy, logit > 0, u < pf)
        sup = torch.where(start, ~fire, self.iw_sup[:B]) & nz
        dec = torch.where(start, torch.where(fire, 1, 2), 0).to(torch.int8)
        lpd = torch.where(start, torch.where(fire, torch.log(pf), torch.log1p(-pf)), torch.zeros_like(logit)).sum(-1)
        self.iw_prev[:B] = nz; self.iw_sup[:B] = sup
        return ~sup, dec, lpd, logit

    def _streams(self):
        if not hasattr(self, '_side'):
            deep = [i for i, t in enumerate(self.tools) if t.fast1 is not None]
            rest = [i for i in range(len(self.tools)) if i not in deep]
            self._groups = [[i] for i in deep] + ([rest] if rest else [])
            self._side = [torch.cuda.Stream() for _ in self._groups]
            self._side_if = torch.cuda.Stream()
        return self._groups, self._side, self._side_if

    def _decide(self, es, hn, B):
        """in-wire decisions for calc + BBH tools; returns masked es, dec, lpd, wf"""
        NTb = len(es)
        mw = self.mwire; ie = float(mw.eps)
        nz = torch.stack([e.abs().sum(-1) > 0 for e in es], -1)
        start = nz & ~self.iw_prev[:B, :NTb]
        logit = mw.k * 0.5 + mw.b[None, :NTb] + hn @ mw.v[:NTb].T
        u = torch.rand(B, NTb, device=self.dev)
        pf = ie + (1 - 2 * ie) * torch.sigmoid(logit)
        fire = torch.where(self.greedy, logit > 0, u < pf)
        sup = torch.where(start, ~fire, self.iw_sup[:B, :NTb]) & nz
        dec = torch.where(start, torch.where(fire, 1, 2), 0).to(torch.int8)
        lpd = torch.where(start, torch.where(fire, torch.log(pf), torch.log1p(-pf)), torch.zeros_like(logit)).sum(-1)
        self.iw_prev[:B, :NTb] = nz; self.iw_sup[:B, :NTb] = sup
        es = [e * (~sup[:, i:i + 1]).to(e.dtype) for i, e in enumerate(es)]
        wf = nz & ~sup
        return es, dec, lpd, wf

    def _step_triton(self, B, S):
        """calc on the main stream first (persistent kernel: never concurrent); then the BBH tools (deep stacks each on
        their own stream) and the IF circuit's token step run concurrently with host layers 0..13; join at layer 14."""
        W, tok = self.W, self.tokb[:B]
        slot = self.slot[:B]
        on = self.imp_on
        links = self.links if on else None
        emb = W['E'][tok]
        feat = None; write = None; wf = None; dec = None; lpd = None; outs = {}; hn = None
        BM = self.B_max
        if on:
            e0 = self.imp.step(emb, B, tok) * self.tool_on[0]
            emb0 = self.E0[tok]
            hn = _lnorm(self.iw_h[:B])                                       # previous position's layer-L input
            groups, side, side_if = self._streams()
            cur = torch.cuda.current_stream()
            for st_, grp in zip(side, groups):
                st_.wait_stream(cur)
                with torch.cuda.stream(st_):
                    for i in grp: outs[i] = self.tools[i].step(emb0, B) * self.tool_on[1 + i]
            if self.iftool is not None:
                side_if.wait_stream(cur)
                with torch.cuda.stream(side_if): self.iftool.if_step(self.tokb[:BM])
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
                for st_ in side: cur.wait_stream(st_)
                es = [e0] + [outs[j] for j in range(len(self.tools))]
                es, dec, lpd, wf = self._decide(es, hn, B)
                feat = {'calc': es[0]}
                if getattr(self, 'calc_D', None) is not None:      # arm C: calc writes through its own (soft, trainable) D
                    write = es[0] @ self.calc_D
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
            mw = self.mwire; ie = float(mw.eps)
            cur.wait_stream(side_if)
            wif = self.iftool.if_write(self.xb[:BM])[:B] * self.tool_on[-1]
            self.if_lat[:B] |= self.iftool.latched()[:B]
            self.if_ans[:B] |= (self.iftool.c.ans[:B] > 0.5)
            nzi = wif.abs().sum(-1) > 0
            j = self.NT - 1
            st_i = nzi & ~self.iw_prev[:B, j]
            lg_i = mw.k * 0.5 + mw.b[j] + hn @ mw.v[j]
            pfi = ie + (1 - 2 * ie) * torch.sigmoid(lg_i)
            fi = torch.where(self.greedy, lg_i > 0, torch.rand(B, device=self.dev) < pfi)
            supi = torch.where(st_i, ~fi, self.iw_sup[:B, j]) & nzi
            deci = torch.where(st_i, torch.where(fi, 1, 2), 0).to(torch.int8)
            lpd = lpd + torch.where(st_i, torch.where(fi, torch.log(pfi), torch.log1p(-pfi)), torch.zeros_like(lg_i))
            self.iw_prev[:B, j] = nzi; self.iw_sup[:B, j] = supi
            wif = wif * (~supi)[:, None].to(wif.dtype)
            self.out_ifw_cur = wif
            fcur = wif / wif.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            feat['ifeval'] = fcur
            self.if_prev_f[:B].copy_(fcur)
            write = write + wif
            dec = torch.cat([dec, deci[:, None]], -1)
            wf = torch.cat([wf, (nzi & ~supi)[:, None]], -1)
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

    def _book_lib(self, B, tok, nxt, lp, wf, dec):
        slot = self.slot[:B]; done = self.done[:B]
        nxt = torch.where(done, torch.full_like(nxt, self.eos), nxt)
        live = ~done
        r, gl = self.rowid[:B], self.glen[:B]
        self.out_tok[r, gl + 1] = torch.where(live, nxt, self.out_tok[r, gl + 1])
        self.out_lp[r, gl + 1] = torch.where(live, lp, self.out_lp[r, gl + 1])
        if wf is not None:
            self.out_wt[r, gl + 1] = torch.where(live[:, None], wf.to(torch.int8), self.out_wt[r, gl + 1])
            self.out_w[r, gl + 1] = torch.where(live, wf[:, 0].to(torch.int8), self.out_w[r, gl + 1])
            self.out_dec3[r, gl + 1] = torch.where(live[:, None], dec, self.out_dec3[r, gl + 1])
            if self.iftool is not None:
                self.out_ifw[r, gl + 1] = torch.where(live[:, None], self.out_ifw_cur, self.out_ifw[r, gl + 1])
        if getattr(self, 'chat_stop', False):            # chat format: stop only at <|im_end|> / <|endoftext|>
            stop = (nxt == self.eos) | (nxt == 151645) | (gl + 2 >= self.maxnew)
        else:
            stop = (nxt == self.eos) | self.has_nn[nxt] | (self.ends_nl[tok] & self.starts_nl[nxt]) | (gl + 2 >= self.maxnew)
        self.glen[:B] += live.long()
        self.done[:B] |= stop
        self.slot[:B] = torch.clamp(slot + 1, max=self.S_max - 1)
        self.tokb[:B] = nxt

    def _mut(self):
        m = [self.tokb, self.slot, self.glen, self.done, self.out_tok, self.out_lp, self.out_w, self.out_wt, self.out_dec3]
        if self.imp_on:
            m += [self.imp.sym_h, self.imp.S_h, self.imp.key_h] + ([self.imp.fst] if hasattr(self.imp, 'fst') else [])   # calc_tag filter state
            for t in self.tools: m += [t.sym_h, t.S_h, t.key_h]
            if self.mwire is not None: m += [self.iw_h, self.iw_prev, self.iw_sup]
            if self.iftool is not None:
                m += [self.out_ifw, self.if_prev_f, self.if_lat, self.if_ans]
                m += [t for leaves, _ in self.iftool.static.values() for t in leaves if torch.is_tensor(t)]
        return m

    def _alloc_out(self, max_new):
        need = (self.B_max + 1, max_new + 2)
        if getattr(self, '_outshape', None) != need:
            super()._alloc_out(max_new)
            self.out_wt = torch.zeros(*need, self.NT, device=self.dev, dtype=torch.int8)
            self.out_dec3 = torch.zeros(*need, self.NT, device=self.dev, dtype=torch.int8)
            if self.iftool is not None:
                self.out_ifw = torch.zeros(*need, Q.D, device=self.dev)
            self.graphs = {}; self.pool = None          # graphs dropped -> their private pool is gone too

    # ------------------------------------------------------------ prefill
    @torch.no_grad()
    def tool_feats(self, full_ids, Lq_start):
        """full_ids (P, Lp + T) prefix+suffix ids. Returns dict name -> e (P, T, n_out) at suffix positions (unvetoed)."""
        out = {}
        for t in self.tools:
            sl = full_ids[:, Lq_start - t.Kt:]
            ch = [t.full(self.E0[sl[c:c + 8]])[:, t.Kt:] for c in range(0, sl.shape[0], 8)]     # bounded memory
            out[t.name] = torch.cat(ch) * self.tool_on[1 + self.tools.index(t)]
        return out

    @torch.no_grad()
    def _if_prefix(self):
        """step the IF circuit once over the shared prefix (B=1), keep that state."""
        c = self.iftool.c
        c.reset(1)
        for t in range(self.Lp): c.step(self.prefix_ids[t:t + 1])
        self.if_prefix_state = {a: tree_flatten(getattr(c, a)) for a in self.iftool.state_names}

    @torch.no_grad()
    def _if_prime(self, qids, rows_list):
        """IF circuit state for B_max rows: prefix state expanded, then the (left-padded) question tokens stepped."""
        it = self.iftool; c = it.c; BM = self.B_max
        if self.if_prefix_state is None: self._if_prefix()
        for a, (leaves, spec) in self.if_prefix_state.items():
            setattr(c, a, tree_unflatten([l.clone() if torch.is_tensor(l) else l for l in leaves], spec))
        c.B = 1
        idx = torch.zeros(BM, dtype=torch.long, device=self.dev)
        for a in it.state_names:
            setattr(c, a, gather_tree(getattr(c, a), idx, 1))
        c.B = BM
        T = max(map(len, qids))
        ids = torch.full((BM, T), self.eos, dtype=torch.long, device=self.dev)
        for p, rr in enumerate(rows_list):
            q = torch.tensor(qids[p], device=self.dev)
            ids[rr, T - len(q):] = q
        it.to_static()
        if not hasattr(self, '_if_gstep'):
            self._if_ids = torch.zeros(BM, dtype=torch.long, device=self.dev)
            def body(): it._set(); c.step(self._if_ids); it._copy_back()
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            save = [t.clone() for leaves, _ in it.static.values() for t in leaves if torch.is_tensor(t)]
            with torch.cuda.stream(s):
                for _ in range(2): body()
            torch.cuda.current_stream().wait_stream(s)
            self._if_gstep = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._if_gstep): body()
            for t, v in zip([t for leaves, _ in it.static.values() for t in leaves if torch.is_tensor(t)], save): t.copy_(v)
        for t in range(T):
            self._if_ids.copy_(ids[:, t]); self._if_gstep.replay()
        it._set()
        self.if_lat_prompt = it.latched().clone()

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
            full = torch.cat([self.tool_prefix[None].expand(P, -1), ids], 1)
            TAG = getattr(self.imp, 'tagf', False)
            Kc_ = (self.imp.K + self.imp.W) * (4 if TAG else 1)          # calc_tag: room for the skipped tag tokens
            tail = torch.stack([full[p, self.Lt + len(qids[p]) - Kc_:self.Lt + len(qids[p])] for p in range(P)])
            emb_tail = self.W['E'][tail]
            if TAG: ef = self.imp.emit_ids(full[:, self.Lt - Kc_:])[:, Kc_:] * self.tool_on[0]
            else: ef = self.imp.full(self.W['E'][full[:, self.Lt - Kc_:]])[:, Kc_:] * self.tool_on[0]
            feat = {'calc': ef}
            tf = self.tool_feats(full, self.Lt)
            feat.update(tf)
            ar = torch.arange(P, device=self.dev)
            e_last = [ef[ar, lens - 1]] + [tf[t.name][ar, lens - 1] for t in self.tools]
            write = torch.zeros(P, T, Q.D, device=self.dev)
            for i, t in enumerate(self.tools):
                write[ar, lens - 1] += e_last[i + 1] @ t.D
            if getattr(self, 'calc_D', None) is not None: write[ar, lens - 1] += e_last[0] @ self.calc_D
            if self.iftool is not None:
                feat['ifeval_prev'] = torch.zeros(P, T, Q.D, device=self.dev)
                feat['ifeval'] = feat['ifeval_prev']
            e_last_all = torch.stack([e.abs().sum(-1) > 0 for e in e_last], -1)       # (P, NTb)
        PC = 64
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
            if TAG: self.imp.prime_rows_ids(tail[src], allrows)
            else: self.imp.prime_rows(emb_tail[src], allrows)
            full0 = full
            for t in self.tools:
                tl = torch.stack([full0[p, self.Lt + len(qids[p]) - t.Kt:self.Lt + len(qids[p])] for p in range(P)])
                t.prime_rows(self.E0[tl][src], allrows)
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

    # ------------------------------------------------------------ generate / wave with multi-tool records
    @torch.no_grad()
    def generate(self, qids, G=1, T=1.0, max_new=320, greedy=False):
        assert max(map(len, qids)) + max_new <= self.S_max
        self._alloc_out(max_new)
        ppc = max(1, self.B_max // G)
        res = dict(tokens=[], logp=[], wflag=[], dec=[], wt=[], ifw=[], if_lat=[], if_lat_prompt=[], if_ans=[])
        torch.cuda.synchronize(); import time; t0 = time.time(); steps = 0
        for p0 in range(0, len(qids), ppc):
            qs = qids[p0:p0 + ppc]; n = len(qs) * G
            steps += self._wave(qs, G, T, max_new, greedy)
            ot, ol, ow, gl = self.out_tok[:n].cpu(), self.out_lp[:n].cpu(), self.out_w[:n].cpu(), self._glen_final.cpu()
            od, owt = self.out_dec3[:n].cpu(), self.out_wt[:n].cpu()
            for j in range(n):
                L = int(gl[j]) + 1
                res['tokens'].append(ot[j, :L].tolist()); res['logp'].append(ol[j, :L].tolist()); res['wflag'].append(ow[j, :L].tolist())
                res['dec'].append(od[j, :L].clone()); res['wt'].append(owt[j, :L].clone())
                if self.iftool is not None and self.imp_on:
                    w_ = self.out_ifw[j, :L]
                    nzp = (w_.abs().sum(-1) > 0).nonzero()[:, 0]
                    res['ifw'].append((nzp.cpu(), w_[nzp].cpu()))            # sparse: positions + vectors
                    res['if_lat'].append(bool(self._lat_final[j])); res['if_lat_prompt'].append(bool(self._latp_final[j]))
                    res['if_ans'].append(bool(self._ans_final[j]))
        torch.cuda.synchronize(); dt = time.time() - t0
        ntok = sum(map(len, res['tokens']))
        self.stats = dict(seconds=dt, tokens=ntok, tok_s=ntok / dt, decode_steps=steps, rows=len(res['tokens']))
        res['stats'] = self.stats
        return res

    def _wave(self, qs, G, T, max_new, greedy):
        n = len(qs) * G; dev = self.dev; DUMMY = self.B_max
        for t in (self.out_lp, self.out_w, self.out_wt, self.out_dec3): t.zero_()
        self.out_tok.fill_(self.eos)
        if self.iftool is not None: self.out_ifw.zero_()
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
        maxlen = int(lens.max()); live_n = n; step = 0
        glen_final = torch.zeros(n + 1, dtype=torch.long, device=dev)
        lat_final = torch.zeros(n + 1, dtype=torch.bool, device=dev); ans_final = torch.zeros_like(lat_final)
        latp_final = torch.zeros_like(lat_final)
        if self.iftool is not None and self.imp_on: latp_final[:n] = self.if_lat_prompt[:n]
        while True:
            Bb = next(b for b in B_BUCKETS if b >= live_n)
            g = self._graph(Bb, self.S_max)
            for _ in range(self.sync_every): g.replay()
            step += self.sync_every
            d = self.done[:live_n]
            n_live = int((~d).sum())
            if n_live == 0 or step >= max_new: break
            nb = next(b for b in B_BUCKETS if b >= n_live)
            if nb < Bb:
                idx = torch.nonzero(~d)[:, 0]; fin = torch.nonzero(d)[:, 0]
                rf = torch.clamp(self.rowid[fin], max=n)
                glen_final[rf] = self.glen[fin]
                if self.iftool is not None and self.imp_on: lat_final[rf] = self.if_lat[fin]; ans_final[rf] = self.if_ans[fin]
                su = min(self.S_max, maxlen + step + 1)
                for i in range(Q.NL):
                    self.Kc[i][:n_live, :, :su] = self.Kc[i][idx, :, :su]
                    self.Vc[i][:n_live, :, :su] = self.Vc[i][idx, :, :su]
                for t in (self.tokb, self.slot, self.glen, self.rowid, self.done):
                    t[:n_live] = t[idx].clone()
                if self.imp_on:
                    self.imp.gather(idx, n_live)
                    for tl in self.tools: tl.gather(idx, n_live)
                    if self.mwire is not None:
                        for t in (self.iw_h, self.iw_prev, self.iw_sup): t[:n_live] = t[idx].clone()
                    if self.iftool is not None:
                        for t in (self.if_prev_f, self.if_lat, self.if_ans): t[:n_live] = t[idx].clone()
                        idx_full = torch.arange(self.B_max, device=dev); idx_full[:n_live] = idx
                        self.iftool.gather(idx_full)
                self.done[n_live:nb] = True; self.rowid[n_live:nb] = DUMMY
                live_n = n_live
        rf = torch.clamp(self.rowid[:live_n], max=n)
        glen_final[rf] = self.glen[:live_n]
        if self.iftool is not None and self.imp_on: lat_final[rf] = self.if_lat[:live_n]; ans_final[rf] = self.if_ans[:live_n]
        self._glen_final = glen_final[:n]; self._lat_final = lat_final[:n].cpu(); self._ans_final = ans_final[:n].cpu()
        self._latp_final = latp_final[:n].cpu()
        return step
