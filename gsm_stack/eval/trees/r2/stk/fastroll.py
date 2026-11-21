"""Fast rollout loop for the serial-stack RL engines (comp_serial.SerialWD, stackh.StackHD): continuous batching.

The math of one decode step is NOT changed: the same captured CUDA graph (host + implants + side state + sampler) runs
on a fixed bucket of B slots, exactly as in C2Engine._wave with C2_NOCOMPACT=1.  Only the generator loop changes:

  old (C2Engine._wave, calc v2 python hook off): rows run in waves of B_max rows; every wave lasts as long as its
      longest row; finished rows stay in the batch and keep attending over a slot pointer that still advances to
      S_max (a finished row costs a full-length attention every step); one host sync per decode step.
  new: a queue of all rows; whenever slots are free, the next rows are admitted into them (prefill of their question
      into that slot's KV rows, per-slot reset of every register to its wave-start value, calc_s4 / chain2 primed);
      a finished slot is harvested and its slot pointer parked at 0 (cost 1 key); host sync every K steps.

Each row's own computation is the same per-row function as before (fixed bucket -> same GEMM shapes; every kernel and
register op is row-local).  The question prefill is batch-variant in bf16 (padding / GEMM shape), so by default
(FAST_EXACT_PREFILL=1) a row's question is prefilled inside its OLD wave group (same prompts, same padding, same call),
writing only the admitted rows: greedy outputs are then token-identical to the old loop.  Sampling uses the same kernel
and the same distribution (another Philox stream assignment).

FAST_WAVE=1 reproduces the old wave schedule exactly (admit only into an empty batch, old wave order, K=1): with the
same seed it must give token-identical outputs to the old loop, sampled or greedy (machinery check).
"""
import os, time, torch
import engine as EM
from engine import B_BUCKETS


def _tkd_cls():
    import tkdual
    return tkdual.TKDual


def make_fast(cls):
    class Fast(cls):
        def fast_config(self, K=8, admit=None, wave=False, exact=True):
            self.f_K = int(K); self.f_wave = bool(wave); self.f_exact = bool(exact)
            self.f_time = os.environ.get('FAST_TIME', '0') == '1'
            self.f_compact = os.environ.get('FAST_COMPACT', '0') == '1'   # bucket compaction after the queue drains
            self.f_admit = int(admit) if admit else max(1, self.B_max // 16)
            return self

        # ------------------------------------------------------------------ helpers
        def _f_regs(self):
            """every per-row register (first dim B_max + 1, all rows equal right after the class's wave-start reset),
            except the output buffers: (name, tensor, wave-start row value).  Tables (rows differ) and per-step scratch
            (overwritten before use) are left alone; s_qlen is set at admission."""
            seen, regs = set(), []
            for k, t in vars(self).items():
                if not isinstance(t, torch.Tensor) or t.dim() < 1 or t.shape[0] != self.B_max + 1: continue
                if k.startswith('out_') or id(t) in seen: continue
                seen.add(id(t))
                if not bool((t == t[:1]).all()): continue
                regs.append((k, t, t[0].clone()))
            return regs

        def _f_stash(self, qs, rws):
            """TKDual priming reads token ids: the same last K+W prefix+question tokens Engine._prefill embeds."""
            if not isinstance(self.imp, _tkd_cls()): return
            KW = self.imp.K + self.imp.W; pre = self.prefix_ids.tolist(); tails = []
            for q, r in zip(qs, rws):
                full = pre + list(q); tails += [full[len(full) - KW:]] * int(r.numel())
            self.imp._tails = torch.tensor(tails, device=self.dev, dtype=torch.long).view(-1, KW)

        def _mut(self):
            m = super()._mut()
            if isinstance(self.imp, _tkd_cls()): m = m + self.imp.state_tensors()
            return m

        def _f_use_tk(self):
            """install the tools-kernel implant for this generate (refresh its symbol tables from the live host E)."""
            tkd = getattr(self, 'f_tk', None)
            if tkd is None: return
            try:
                tkd.refresh()
            except AssertionError as ex:               # symbol table no longer 0/1-compatible: torch implant from now on
                print('fastroll: TK refresh failed -> torch implant:', ex, flush=True); self.f_tk = None
                self.imp = self.f_dual; self.graphs = {}; return
            if self.imp is not tkd: self.imp = tkd; self.graphs = {}

        def _f_admit(self, slots, rows, qids, G, T, max_new, greedy):
            dev = self.dev
            S = torch.tensor(slots, device=dev, dtype=torch.long)
            for _, t, v in self._f_regs0: t[S] = v
            for k in ('out_lp', 'out_dec', 'out_ops', 'out_iwl', 'out_run', 'out_span', 'out_w'):
                getattr(self, k)[S] = 0
            self.out_tok[S] = self.eos
            # prompts in admission order, each with its target slots
            pr, grp = [], {}
            for s, r in zip(slots, rows):
                p = r // G
                if p not in grp: grp[p] = []; pr.append(p)
                grp[p].append(s)
            rws = [torch.tensor(grp[p], device=dev, dtype=torch.long) for p in pr]
            if self.f_exact:
                # prefill exactly as the old loop did: the whole old wave group of each prompt (ppc consecutive prompts,
                # padded to the group's max length) in one call; only the admitted rows' KV / implant states are written.
                ppc = max(1, self.B_max // G); nP = len(qids)
                gl_, gle, gel = {}, {}, {}
                for gi in sorted({p // ppc for p in pr}):
                    ps = list(range(gi * ppc, min(nP, (gi + 1) * ppc)))
                    rr = [torch.tensor(grp.get(p, []), device=dev, dtype=torch.long) for p in ps]
                    self._f_stash([qids[p] for p in ps], rr)
                    lg, ln, el = EM.Engine._prefill(self, [qids[p] for p in ps], rr)
                    for k_, p in enumerate(ps):
                        gl_[p] = lg[k_]; gle[p] = ln[k_]; gel[p] = None if el is None else el[k_]
                logits = torch.stack([gl_[p] for p in pr]); lens = torch.stack([gle[p] for p in pr])
                e_last = None if gel[pr[0]] is None else torch.stack([gel[p] for p in pr])
            else:
                self._f_stash([qids[p] for p in pr], rws)
                logits, lens, e_last = EM.Engine._prefill(self, [qids[p] for p in pr], rws)
            cnt = torch.tensor([len(grp[p]) for p in pr], device=dev)
            srt = torch.cat(rws)
            z = (logits / T).repeat_interleave(cnt, 0)
            first = z.argmax(-1) if greedy else (z - torch.log(-torch.log(torch.rand_like(z).clamp_(1e-20, 1.0)))).argmax(-1)
            self.out_tok[srt, 0] = first
            self.out_lp[srt, 0] = torch.log_softmax(z, -1).gather(-1, first[:, None])[:, 0]
            if e_last is not None: self.out_w[srt, 0] = (e_last.abs().sum(-1) > 0).to(torch.int8).repeat_interleave(cnt)
            del z
            L = lens.repeat_interleave(cnt)
            self.tokb[srt] = first; self.slot[srt] = L; self.glen[srt] = 0; self.rowid[srt] = srt
            # serial all-weights stack: only the implant's eos (or the budget) stops a row (= rel/v5 fix, task 7 audit)
            hs = torch.zeros_like(first, dtype=torch.bool) if hasattr(self, 'r_eos') else (first == self.eos) | self.has_nn[first]
            self.done[srt] = hs | (max_new <= 1)
            self.row_on[srt] = 1.0
            if hasattr(self, 's_qlen'): self.s_qlen[srt] = L
            return int(lens.max())

        def _f_rowregs(self):
            """every tensor indexed by batch slot (first dim B_max or B_max + 1) that moves with a row on compaction;
            out_* buffers are indexed by rowid (the output row) and stay; KV / side caches are moved up to the length."""
            seen, regs = set(), []
            for k, t in vars(self).items():
                if not isinstance(t, torch.Tensor) or t.dim() < 1 or t.shape[0] not in (self.B_max, self.B_max + 1): continue
                if k.startswith('out_') or k in ('w_Kx', 'w_ar') or id(t) in seen: continue
                seen.add(id(t)); regs.append((k, t))
            return regs

        def _f_move(self, src, dst, nb):
            """compaction: rows in slots src -> free slots dst (all dst < nb <= every src)."""
            dev = self.dev
            S = torch.tensor(src, device=dev, dtype=torch.long); D = torch.tensor(dst, device=dev, dtype=torch.long)
            L = int(self.slot[S].max()) + 1
            for i in range(Q.NL):
                self.Kc[i][D, :, :L] = self.Kc[i][S, :, :L]
                self.Vc[i][D, :, :L] = self.Vc[i][S, :, :L]
            if hasattr(self, 'w_Kx'): self.w_Kx[D, :L] = self.w_Kx[S, :L]
            for _, t in self.f_rowregs: t[D] = t[S]
            if self.imp is not None:
                idx = torch.arange(nb, device=dev); idx[D] = S
                self.imp.gather(idx, nb)
            for s, d in zip(src, dst):
                self.f_occ[d] = self.f_occ[s]; self.f_occ[s] = -1
            self.done[nb:self.B_max] = True; self.slot[nb:self.B_max] = 0

        def _f_harvest(self, slots, res, regrec):
            S = torch.tensor(slots, device=self.dev, dtype=torch.long)
            O = self.rowid[S]                      # output rows (== slots unless rows were compacted)
            gl = self.glen[S].cpu(); ot = self.out_tok[O].cpu(); ol = self.out_lp[O].cpu(); ow = self.out_w[O].cpu()
            if regrec is not None:
                R = dict(sym=self.R_sym[S], ok=self.R_ok[S], bad=self.R_bad[S], lg=self.R_lg[S], natt=self.s_a[S] + 1,
                         nseg=self.s_nf[S] + 1, win=self.s_win[S], nrep=self.s_nrep[S])
                R = {k: v.cpu() for k, v in R.items()}
            for j, s in enumerate(slots):
                r = self.f_occ[s]; Lr = int(gl[j]) + 1
                res[r] = (ot[j, :Lr].tolist(), ol[j, :Lr].tolist(), ow[j, :Lr].tolist())
                if regrec is not None: regrec[r] = {k: v[j] for k, v in R.items()}
                self.f_occ[s] = -1
            self.slot[S] = 0                       # park: a finished slot attends over 1 key until it is refilled

        # ------------------------------------------------------------------ generate
        @torch.no_grad()
        def generate(self, qids, G=1, T=1.0, max_new=320, greedy=False, **kw):
            assert getattr(self, 'c2_mode', 'off') == 'off', 'fast loop: calc v2 python hook must be off'
            assert max(map(len, qids)) + max_new <= self.S_max
            self._alloc_out(max_new)
            self._f_use_tk()
            self._rec = []; self.c2_rows = []; self.c2_prime = None; self._c2_p0 = 0
            self._f_capture = True                     # run the class's own wave-start resets (no decode)
            try: self._wave([], G, T, max_new, greedy)
            finally: self._f_capture = False
            self._rec = []
            self.temp.fill_(T); self.greedy.fill_(greedy); self.maxnew.fill_(max_new)
            self._f_regs0 = self._f_regs()
            dev = self.dev; Bb = self.B_max; DUMMY = self.B_max
            n = len(qids) * G
            self.tokb[:Bb] = self.eos; self.slot[:Bb] = 0; self.glen[:Bb] = 0; self.done[:Bb] = True
            self.rowid[:Bb] = torch.arange(Bb, device=dev); self.row_on.fill_(1.0)
            self.f_occ = [-1] * Bb
            regrec = [None] * n if hasattr(self, 'R_sym') else None
            res = [None] * n
            torch.cuda.synchronize(); t0 = time.time()
            nxt = 0; steps = 0; live = 0; K = 1 if self.f_wave else self.f_K
            n_admit = 0; busy_steps = 0; t_adm = 0.0
            Bc = Bb; bsteps = 0; n_cmp = 0; t_cmp = 0.0
            bhist = {}; t_b = {}                      # decode steps / wall seconds per bucket (FAST_TIME only syncs)
            compact = self.f_compact and not self.f_wave
            if compact:
                self.f_rowregs = self._f_rowregs()
                if not getattr(self, '_f_rr_printed', False):
                    self._f_rr_printed = True; print('fastroll compaction moves:', [k for k, _ in self.f_rowregs], flush=True)
            while True:
                free = [s for s in range(Bc) if self.f_occ[s] < 0]
                if nxt < n and free:
                    if self.f_wave:
                        k = min(len(free), n - nxt) if len(free) == Bb else 0
                    else:
                        k = min(len(free), n - nxt) if (len(free) >= self.f_admit or n - nxt <= len(free)) else 0
                    if k:
                        sl = free[:k]; rows = list(range(nxt, nxt + k))
                        ta = time.time()
                        self._f_admit(sl, rows, qids, G, T, max_new, greedy)
                        if self.f_time: torch.cuda.synchronize(); t_adm += time.time() - ta
                        for s, r in zip(sl, rows): self.f_occ[s] = r
                        nxt += k; n_admit += 1
                        # rows done at their first token: harvest immediately (as the old loop's done mask would)
                live = sum(1 for s in range(Bc) if self.f_occ[s] >= 0)
                if live == 0 and nxt >= n: break
                if compact and nxt >= n:                   # queue drained: live rows to the front, smaller bucket graph
                    nb = next(b for b in B_BUCKETS if b >= live)
                    if nb < Bc:
                        tc = time.time()
                        src = [s for s in range(nb, Bc) if self.f_occ[s] >= 0]
                        dst = [s for s in range(nb) if self.f_occ[s] < 0][:len(src)]
                        if src: self._f_move(src, dst, nb)
                        Bc = nb; n_cmp += 1
                        if self.f_time: torch.cuda.synchronize(); t_cmp += time.time() - tc
                g = self._graph(Bc, self.S_max)
                tb_ = time.time()
                for _ in range(K): g.replay()
                bhist[Bc] = bhist.get(Bc, 0) + K
                steps += K; busy_steps += K * live; bsteps += K * Bc
                dl = self.done[:Bc].tolist()
                fin = [s for s in range(Bc) if self.f_occ[s] >= 0 and dl[s]]
                if fin: self._f_harvest(fin, res, regrec)
                t_b[Bc] = t_b.get(Bc, 0.0) + time.time() - tb_
            torch.cuda.synchronize(); dt = time.time() - t0
            if getattr(self, 'f_tk', None) is not None:
                fl = int(self.f_tk.TK.flag.item())
                if fl: print('fastroll: TK exactness guard flag', fl, flush=True)
                self.f_tkflag = fl
            tok_ = [r[0] for r in res]; lp_ = [r[1] for r in res]; w_ = [r[2] for r in res]
            ntok = sum(map(len, tok_))
            self.stats = dict(seconds=dt, tokens=ntok, tok_s=ntok / dt, decode_steps=steps, rows=n, admits=n_admit,
                              occupancy=busy_steps / max(1, steps * Bb), admit_s=t_adm if self.f_time else None,
                              bucket_occ=busy_steps / max(1, bsteps), compactions=n_cmp, compact_s=t_cmp if self.f_time else None,
                              bucket_steps=bhist, bucket_s={k: round(v, 3) for k, v in t_b.items()})
            out = dict(tokens=tok_, logp=lp_, wflag=w_, dec=[[] for _ in tok_], ops=[[] for _ in tok_], iw=[],
                       on=[1.0] * n, stats=self.stats, c2=[], rollback_tok=0)
            if regrec is not None:
                out['reg'] = {k: torch.stack([regrec[r][k] for r in range(n)]) for k in regrec[0]}
            return out

    Fast.__name__ = cls.__name__ + 'Fast'
    return Fast


# ---------------------------------------------------------------------------------------------------- forget-skip attention
import triton, triton.language as tl
import fastqwen as Q
import torch.nn.functional as F


@triton.jit
def attn_suffix_bias_skip_kernel(q, Kc, Vc, slot, Qx, Kx, sKx_b, qlen, fr, m_in, l_in, acc_in, out, sK_b, sK_h, sK_s, scale,
                                 H: tl.constexpr, HD: tl.constexpr, BN: tl.constexpr, IEEE: tl.constexpr, E: tl.constexpr):
    """forget_ws.attn_suffix_bias_kernel, but key blocks that lie entirely inside a forgotten span [qlen, fr) are not read.
    With M = 1e4 every key in such a block has p = exp(s - m) == 0.0 exactly (s <= q.k*scale - 1e4, m >= a prefix score),
    and the block's online-softmax update is the identity (al = exp(0) = 1, sum += 0, acc += 0): skipping it is bitwise
    exact. Block boundaries stay the old multiples of BN, so every computed block is the old computation."""
    r = tl.program_id(0); g = tl.program_id(1)
    mi = tl.arange(0, 16)
    qm = mi < 2
    head = 2 * g + mi
    d = tl.arange(0, HD)
    Qv = tl.load(q + (r * H + head[:, None]) * HD + d[None, :], mask=qm[:, None], other=0.0)
    o = r * H + head
    m = tl.load(m_in + o, mask=qm, other=0.0); l = tl.load(l_in + o, mask=qm, other=1.0)
    acc = tl.load(acc_in + o[:, None] * HD + d[None, :], mask=qm[:, None], other=0.0)
    n = tl.arange(0, BN)
    e = tl.arange(0, E)
    qx = tl.load(Qx + r * E + e)
    L = tl.load(slot + r) + 1
    lo = tl.load(qlen + r); hi = tl.load(fr + r)
    for j0 in range(0, L, BN):
        if (j0 < lo) | (j0 + BN > hi):
            jj = j0 + n
            km = jj < L
            K = tl.load(Kc + r * sK_b + g * sK_h + jj[:, None] * sK_s + d[None, :], mask=km[:, None], other=0.0)
            if IEEE:
                s = tl.dot(Qv, tl.trans(K), input_precision='ieee') * scale
            else:
                s = tl.dot(Qv, tl.trans(K)) * scale
            kx = tl.load(Kx + r * sKx_b + jj[:, None] * E + e[None, :], mask=km[:, None], other=0.0)
            bias = tl.sum(kx * qx[None, :], axis=1)
            s = s + bias[None, :]
            s = tl.where(km[None, :], s, float('-inf'))
            mn = tl.maximum(m, tl.max(s, axis=1))
            pr = tl.exp(s - mn[:, None])
            al = tl.exp(m - mn)
            l = l * al + tl.sum(pr, axis=1)
            V = tl.load(Vc + r * sK_b + g * sK_h + jj[:, None] * sK_s + d[None, :], mask=km[:, None], other=0.0)
            if IEEE:
                acc = acc * al[:, None] + tl.dot(pr.to(V.dtype), V, input_precision='ieee')
            else:
                acc = acc * al[:, None] + tl.dot(pr.to(V.dtype), V)
            m = mn
    res = acc / l[:, None]
    tl.store(out + o[:, None] * HD + d[None, :], res.to(out.dtype.element_ty), mask=qm[:, None])


SUF_BN = int(os.environ.get('FAST_SUFBN', 64))   # 64 = exact; 32 only for the numerical noise-floor check


def _attn_suffix_skip(self, i, B, q, Kc, Vc, slot, att, sc, ieee):
    if not self.c_forget or self.w_dbg:
        return type(self).__mro__[1]._attn_suffix(self, i, B, q, Kc, Vc, slot, att, sc, ieee)
    if i == 0:                                         # side coordinates of this step (as forget_ws)
        a = self.s_nf[:B].clamp(max=self.w_E - 1)
        lab = (a - (slot < self.s_fr[:B]).long()).clamp(min=0)
        self.w_Kx[self.w_ar[:B], slot] = F.one_hot(lab, self.w_E).float() @ self.w_WK
        self.w_Qx[:B] = (F.one_hot(a, self.w_E).float() @ self.w_WQ) * self.w_M
    attn_suffix_bias_skip_kernel[(B, Q.KVH)](q, Kc, Vc, slot, self.w_Qx, self.w_Kx, self.w_Kx.stride(0), self.s_qlen, self.s_fr,
                                             self.mp, self.lpf, self.accp, att, Kc.stride(0), Kc.stride(1), Kc.stride(2), sc,
                                             H=Q.H, HD=Q.HD, BN=(SUF_BN if not ieee else 32), IEEE=ieee, E=self.w_E,
                                             num_warps=4, num_stages=1 if ieee else 2)


# C2Engine._wave is called explicitly by the classes' own _wave after their resets; in capture mode it does nothing.
import c2_engine as _c2m
if not getattr(_c2m.C2Engine._wave, '_fast_patched', False):
    _orig = _c2m.C2Engine._wave

    def _wave(self, qs, G, T, max_new, greedy):
        if getattr(self, '_f_capture', False): return 0
        return _orig(self, qs, G, T, max_new, greedy)
    _wave._fast_patched = True
    _c2m.C2Engine._wave = _wave


def enable(eng, K=None, admit=None, wave=None):
    K = int(os.environ.get('FAST_K', 8)) if K is None else K
    admit = int(os.environ.get('FAST_ADMIT', 0)) if admit is None else admit
    wave = os.environ.get('FAST_WAVE', '0') == '1' if wave is None else wave
    eng.__class__ = make_fast(eng.__class__)
    # exact forget-skip attention: only for the serial stack (side labels from s_nf / s_fr, prompt length s_qlen) and only
    # when the forget bias underflows exactly (M >= 1e3); arm B (StackH, M = 50) keeps the full kernel.
    if (os.environ.get('FAST_FSKIP', '1') == '1' and hasattr(eng, 'R_sym') and hasattr(eng, 'w_M') and eng.w_M >= 1e3):
        eng.__class__._attn_suffix = _attn_suffix_skip; eng.f_fskip = True
        eng.__class__.f_att3 = True                    # comp_serial FAST_ATT3: split-KV attention on the fast class only
    else: eng.f_fskip = False
    eng.fast_config(K=K, admit=admit or None, wave=wave, exact=os.environ.get('FAST_EXACT_PREFILL', '1') == '1')
    eng.f_dual = eng.imp; eng.f_tk = None
    if os.environ.get('FAST_TK', '1') == '1' and type(eng.imp).__name__ == 'DualImplant':
        import tkdual
        eng.f_tk = tkdual.TKDual(eng.imp, eng.W['E'], eng.B_max)
    if os.environ.get('FAST_C2ALT', '0') != '1': eng.c2_on = False   # calc v2 hook is off: its ALT argmax is dead work
    eng.graphs = {}; eng.pool = None; eng._outshape = None
    return eng
