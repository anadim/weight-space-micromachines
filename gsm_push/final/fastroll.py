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
register op is row-local), so greedy outputs match the old loop up to bf16 effects of a different prefill grouping;
sampling uses the same kernel and the same distribution (another Philox stream assignment).

FAST_WAVE=1 reproduces the old wave schedule exactly (admit only into an empty batch, old wave order, K=1): with the
same seed it must give token-identical outputs to the old loop, sampled or greedy (machinery check).
"""
import os, time, torch
import engine as EM
from engine import B_BUCKETS


def make_fast(cls):
    class Fast(cls):
        def fast_config(self, K=8, admit=None, wave=False):
            self.f_K = int(K); self.f_wave = bool(wave)
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
            qs = [qids[p] for p in pr]
            rws = [torch.tensor(grp[p], device=dev, dtype=torch.long) for p in pr]
            logits, lens, e_last = EM.Engine._prefill(self, qs, rws)
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
            self.done[srt] = (first == self.eos) | self.has_nn[first] | (max_new <= 1)
            self.row_on[srt] = 1.0
            if hasattr(self, 's_qlen'): self.s_qlen[srt] = L
            return int(lens.max())

        def _f_harvest(self, slots, res, regrec):
            S = torch.tensor(slots, device=self.dev, dtype=torch.long)
            gl = self.glen[S].cpu(); ot = self.out_tok[S].cpu(); ol = self.out_lp[S].cpu(); ow = self.out_w[S].cpu()
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
            n_admit = 0; busy_steps = 0
            while True:
                free = [s for s in range(Bb) if self.f_occ[s] < 0]
                if nxt < n and free:
                    if self.f_wave:
                        k = min(len(free), n - nxt) if len(free) == Bb else 0
                    else:
                        k = min(len(free), n - nxt) if (len(free) >= self.f_admit or n - nxt <= len(free)) else 0
                    if k:
                        sl = free[:k]; rows = list(range(nxt, nxt + k))
                        self._f_admit(sl, rows, qids, G, T, max_new, greedy)
                        for s, r in zip(sl, rows): self.f_occ[s] = r
                        nxt += k; n_admit += 1
                        # rows done at their first token: harvest immediately (as the old loop's done mask would)
                live = sum(1 for s in range(Bb) if self.f_occ[s] >= 0)
                if live == 0 and nxt >= n: break
                g = self._graph(Bb, self.S_max)
                for _ in range(K): g.replay()
                steps += K; busy_steps += K * live
                dl = self.done[:Bb].tolist()
                fin = [s for s in range(Bb) if self.f_occ[s] >= 0 and dl[s]]
                if fin: self._f_harvest(fin, res, regrec)
            torch.cuda.synchronize(); dt = time.time() - t0
            tok_ = [r[0] for r in res]; lp_ = [r[1] for r in res]; w_ = [r[2] for r in res]
            ntok = sum(map(len, tok_))
            self.stats = dict(seconds=dt, tokens=ntok, tok_s=ntok / dt, decode_steps=steps, rows=n, admits=n_admit,
                              occupancy=busy_steps / max(1, steps * Bb))
            out = dict(tokens=tok_, logp=lp_, wflag=w_, dec=[[] for _ in tok_], ops=[[] for _ in tok_], iw=[],
                       on=[1.0] * n, stats=self.stats, c2=[], rollback_tok=0)
            if regrec is not None:
                out['reg'] = {k: torch.stack([regrec[r][k] for r in range(n)]) for k in regrec[0]}
            return out

    Fast.__name__ = cls.__name__ + 'Fast'
    return Fast


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
    eng.fast_config(K=K, admit=admit or None, wave=wave)
    if os.environ.get('FAST_C2ALT', '0') != '1': eng.c2_on = False   # calc v2 hook is off: its ALT argmax is dead work
    eng.graphs = {}; eng.pool = None; eng._outshape = None
    return eng
