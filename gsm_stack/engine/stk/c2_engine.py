"""Calculator v2 as an exact decode-time token hook in the RL engine (engine.py), next to the frozen calc_s4 implant.

Inside the decode CUDA graph (one extra op per step, does not touch the host's token): the sampler is wrapped so that
it also stores ALT[b] = argmax of the step's logits over tokens that are not 'vetoed' (calc2.is_vetoed: a digit start,
or an operator / '=' / '('). Outside the graph, between replays (sync every step), each row runs calc2.Row on the token
the host just booked. When it acts, it overwrites that booked token in place (tokb = the token fed at the next step,
out_tok = the output stream) with the exact value's tokens, one per step, and then the ALT token if the host tries to
extend the number or continue the arithmetic. The overwritten token is what the host and calc_s4 read at the next step,
exactly as if the host had sampled it. When a row's tracker never acts, nothing of that row is changed.

out_w bits: 1 calc_s4 write (unchanged), 2 v2-written value token, 4 v2 veto (ALT) token.
"""
import os, sys, time, torch
import engine as EM
from engine import Engine, B_BUCKETS, S_STEP
import fastqwen as Q
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calc2
NOCOMPACT = os.environ.get('C2_NOCOMPACT', '0') == '1'   # keep every row in its slot (row results independent of others)


class _SampProxy:
    def __init__(self, real, eng): self.real, self.eng = real, eng

    def __getitem__(self, grid):
        f = self.real[grid]
        eng = self.eng

        def call(logits, *args, **kw):
            f(logits, *args, **kw)
            B = logits.shape[0]
            if eng.c2_on:
                eng.c2_alt[:B] = logits.float().masked_fill(eng.c2_mask[None], float('-inf')).argmax(-1)
        return call


class _HKProxy:
    def __init__(self, hk, eng): self._hk = hk; self.sample_kernel = _SampProxy(hk.sample_kernel, eng)

    def __getattr__(self, k): return getattr(self._hk, k)


class C2Engine(Engine):
    def c2_setup(self, mode='ext', exact=True):
        tok = self.tok
        V = self.W['E'].shape[0]
        self.c2_txt = [tok.decode([i]) for i in range(len(tok))] + [''] * max(0, V - len(tok))
        self.c2_mask = torch.tensor([calc2.is_vetoed(t) for t in self.c2_txt], device=self.dev)
        self.c2_alt = torch.zeros(self.B_max + 1, dtype=torch.long, device=self.dev)
        self.c2_on = True
        if not isinstance(EM.HK, _HKProxy): EM.HK = _HKProxy(EM.HK, self)
        else: EM.HK.sample_kernel.eng = self
        self.c2_mode, self.c2_exact, self.c2_policy = mode, exact, 'look'
        self.c2_rollback_tok = 0
        self._tcache = {}
        self.graphs = {}; self.pool = None; self._outshape = None
        self.c2_rows = []
        self.c2_prime = None          # optional per-question text the trackers start from (engine exactness test)
        return self

    def set_c2(self, mode, exact=True, policy='look'):
        self.c2_mode, self.c2_exact, self.c2_policy = mode, exact, policy

    def _tokz(self, s):
        if s not in self._tcache: self._tcache[s] = self.tok(s, add_special_tokens=False)['input_ids']
        return self._tcache[s]

    def generate(self, qids, G=1, T=1.0, max_new=320, greedy=False, prime=None):
        self.c2_prime = prime; self.c2_rows = []; self._c2_p0 = 0
        out = super().generate(qids, G, T, max_new, greedy)
        out['c2'] = [dict(events=r.events, vetoes=r.vetoes) for r in self.c2_rows]
        out['rollback_tok'] = self.c2_rollback_tok; self.c2_rollback_tok = 0
        return out

    def _wave(self, qs, G, T, max_new, greedy):
        if self.c2_mode == 'base':                  # the unmodified engine (sync every 16, no tracker)
            self.c2_rows += [calc2.Row('off') for _ in range(len(qs) * G)]
            return Engine._wave(self, qs, G, T, max_new, greedy)
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
        # serial all-weights stack: the generator stops only when the implant wrote eos (or the budget); a host stop token
        # as the first token is fed to the circuit like any other (its END trigger reads it) [task 7]
        hs = torch.zeros_like(first, dtype=torch.bool) if hasattr(self, 'r_eos') else (first == self.eos) | self.has_nn[first]
        self.done[:n] = hs | (max_new <= 1)
        self.tokb[n:Bb] = self.eos; self.slot[n:Bb] = 0; self.glen[n:Bb] = 0; self.rowid[n:Bb] = DUMMY; self.done[n:Bb] = True
        self._wave_on = torch.ones(n, device=dev)
        self.row_on.fill_(1.0); self.row_on[:n] = self._wave_on
        # ---- v2 trackers (one per row), fed the first token
        on = self.c2_mode not in ('off', 'base')
        R = [calc2.Row(self.c2_mode, self.c2_exact, self.c2_policy) for _ in range(n)]
        self.c2_snap = {}
        if self.c2_policy == 'look': assert self.imp.fk is None, 'look policy needs the torch implant step (--fused 0)'
        p0 = self._c2_p0; self._c2_p0 += len(qs)
        if self.c2_prime is not None:
            for j in range(n):
                R[j].text = self.c2_prime[p0 + j // G]
                R[j]._idle_check(R[j].text)
        self.c2_rows += R
        fdone = set()
        ft = first.tolist(); fw = self.out_w[:n, 0].tolist(); fd = self.done[:n].tolist()
        pend_first = []
        for j in range(n):
            if on: self._c2_feed(R[j], j, j, ft[j], fw[j], 0, max_new, None, pend_first, init=True)
            if fd[j]: fdone.add(j)
        self._c2_apply(pend_first, max_new)
        maxlen = int(lens.max()); live_n = n; step = 0
        glen_final = torch.zeros(n + 1, dtype=torch.long, device=dev)
        while True:
            Bb = next(b for b in B_BUCKETS if b >= live_n)
            Sb = self.S_max
            g = self._graph(Bb, Sb)
            g.replay()
            step += 1
            if on:
                gl = self.glen[:live_n]; rid = self.rowid[:live_n]
                info = torch.stack([self.tokb[:live_n], rid, self.done[:live_n].long(), gl, self.c2_alt[:live_n],
                                    self.out_w[rid.clamp(max=n), gl].long(), self.slot[:live_n]]).cpu().tolist()
                self._slots = info[6]
                acts = []
                for s in range(live_n):
                    r = info[1][s]
                    if r >= n or r in fdone: continue
                    self._c2_feed(R[r], s, r, info[0][s], info[5][s], info[3][s], max_new, info[4][s], acts)
                    if info[2][s] and not any(a[0] == s for a in acts): fdone.add(r)
                self._c2_apply(acts, max_new)
                for a in acts:
                    if a[5]: fdone.add(a[1])
            d = self.done[:live_n]
            n_live = int((~d).sum())
            if n_live == 0 or step >= max_new:
                break
            nb = next(b for b in B_BUCKETS if b >= n_live)
            if nb < Bb and not NOCOMPACT:
                idx = torch.nonzero(~d)[:, 0]; fin = torch.nonzero(d)[:, 0]
                glen_final[torch.clamp(self.rowid[fin], max=n)] = self.glen[fin]
                su = min(self.S_max, maxlen + step + 1)
                for i in range(Q.NL):
                    self.Kc[i][:n_live, :, :su] = self.Kc[i][idx, :, :su]
                    self.Vc[i][:n_live, :, :su] = self.Vc[i][idx, :, :su]
                for t in (self.tokb, self.slot, self.glen, self.rowid, self.done, self.row_on):
                    t[:n_live] = t[idx].clone()
                if self.imp is not None and self.imp_on: self.imp.gather(idx, n_live)
                self.done[n_live:nb] = True; self.rowid[n_live:nb] = DUMMY
                live_n = n_live
        glen_final[torch.clamp(self.rowid[:live_n], max=n)] = self.glen[:live_n]
        self._glen_final = glen_final[:n]
        return step

    # one booked token of one row -> maybe an action (slot, row, new_tok, bit, prev_tok_text, stops)
    def _c2_feed(self, row, s, r, t_id, wf, glen, max_new, alt, acts, init=False):
        t = self.c2_txt[t_id]
        if row.st == calc2.FORCING:
            nt = row.q.pop(0); last = not row.q
            row.forced(self.c2_txt[nt], last)
            acts.append((s, r, nt, 2, glen, self._stops(row, nt, glen, max_new)))
            return
        a = row.feed(t, wf, self._tokz)
        if a is None: return
        if a[0] == 'snap':                      # host starts its number: remember where (KV slot, output position, calc_s4 state)
            if init: row.st = calc2.IDLE; return  # (never at the first answer token)
            imp = self.imp
            self.c2_snap[r] = (self._slots[s], glen, imp.sym_h[s].clone(), imp.S_h[s].clone(), imp.key_h[s].clone(), imp.fst[s].clone())
            return
        if a[0] == 'rollback':                  # rewind this row to the snapshot, then write the exact value
            slot0, g0, sh, Sh, kh, fs = self.c2_snap.pop(r)
            imp = self.imp
            imp.sym_h[s] = sh; imp.S_h[s] = Sh; imp.key_h[s] = kh; imp.fst[s] = fs
            self.slot[s] = slot0; self.glen[s] = g0
            self.c2_rollback_tok += glen - g0 + 1
            nt = row.q.pop(0); last = not row.q
            row.forced(self.c2_txt[nt], last)
            acts.append((s, r, nt, 2, g0, self._stops(row, nt, g0, max_new)))
            return
        if a[0] == 'replace':
            ids = row.q; nt = ids.pop(0); last = not ids
            row.forced(self.c2_txt[nt], last)
            acts.append((s, r, nt, 2, glen, self._stops(row, nt, glen, max_new)))
        elif a[0] == 'veto':
            if init or alt is None:          # first answer token cannot be a veto (no number before it)
                return
            row.vetoed(self.c2_txt[alt])
            acts.append((s, r, alt, 4, glen, self._stops(row, alt, glen, max_new)))

    def _stops(self, row, nt, glen, max_new):
        tx = self.c2_txt[nt]
        prev = row.text[:-len(tx)] if tx and row.text.endswith(tx) else row.text
        return (nt == self.eos) or ('\n\n' in tx) or (prev.endswith('\n') and tx.startswith('\n')) or (glen + 2 >= max_new)

    def _c2_apply(self, acts, max_new):
        if not acts: return
        dev = self.dev
        S = torch.tensor([a[0] for a in acts], device=dev); Rr = torch.tensor([a[1] for a in acts], device=dev)
        Tn = torch.tensor([a[2] for a in acts], device=dev); Bt = torch.tensor([a[3] for a in acts], device=dev, dtype=torch.int8)
        Gl = torch.tensor([a[4] for a in acts], device=dev); Dn = torch.tensor([a[5] for a in acts], device=dev)
        self.tokb[S] = Tn
        self.out_tok[Rr, Gl] = Tn
        self.out_w[Rr, Gl] = self.out_w[Rr, Gl] | Bt
        self.done[S] = Dn
