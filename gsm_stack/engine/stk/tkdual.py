"""calc_s4 + chain2 (implant2.DualImplant) decode steps in engine2's exact tools kernel (tk3, NVRTC; integer, differential,
event-driven; bit-exact to the torch CSR stacks, verified on calc_s4 in engine2 and checked here end to end).

TKDual wraps a DualImplant: step() runs both machines in ONE kernel launch over all rows and applies the same fixed gate
(chain2 has priority) as DualImplant.step; prime_rows() primes admitted rows by scanning their last K+W prompt tokens
through the kernel in scratch rows, then copies the ring state to the target slots (= the torch prime: the state after
the tail). Everything else (D, M.emit for the first-token e, n_out, ...) is delegated to the DualImplant.
The reader reads the host embedding E: refresh() rebuilds the symbol tables after a weight update (asserted 0/1)."""
import os
import torch
import tk
from implant2 import _gate

PPRIME = os.environ.get('TK_PPRIME', '1') == '1'      # parallel prime (prime_ids_par); 0 = sequential scan
PEMIT = os.environ.get('TK_PEMIT', '1') == '1'        # prefill first-token e from the kernel (prime_emit) instead of M.emit
NPROBE = 32                                           # probe rows at the end of the scratch block (prime_emit)


class TKDual:
    def __init__(self, dual, E, B_max):
        self.dual = dual
        calc, chain = dual.parts
        self.E = E
        self.TK = tk.ToolsKernel3([('calc', calc.M, E), ('chain', chain.M, E)], B_max, scratch=B_max)
        (self.o1, self.n1), (self.o2, self.n2) = self.TK.outoff
        self.KW = dual.K + dual.W
        self.fst = torch.zeros(B_max, dtype=torch.long, device=E.device)
        self._tails = None

    def __getattr__(self, k):
        return getattr(self.__dict__['dual'], k)

    def state_tensors(self):
        return list(self.TK._state_ids)

    @torch.no_grad()
    def refresh(self):
        ch = [self.TK.update_symtab(i, self.E) for i in range(2)]
        return any(ch)

    def step(self, emb, B, tok=None):
        e = self.TK.step(tok, B)
        return _gate(e[:B, self.o1:self.o1 + self.n1], e[:B, self.o2:self.o2 + self.n2])

    @torch.no_grad()
    def prime_ids(self, tail_ids, rows):
        if PPRIME: return self.prime_ids_par(tail_ids, rows)
        return self.prime_ids_seq(tail_ids, rows)

    @torch.no_grad()
    def prime_ids_par(self, tail_ids, rows, n_scr=None):
        """Same final state as prime_ids_seq (bitwise), without the Lt sequential kernel steps.
        A step recomputes stack 1 from its K-token window alone; the only state is the rings. After the scan over
        the tail (from reset: hs = K-1, hw = W-1) tool t holds: tokring[p % K] = tail[p] for the last K positions,
        keyring[p % W] = keytab[tail[p]] and Sring[p % W] = stack1(window ending at p) for the last W positions,
        hs = (Lt-1) % K, hw = (Lt-1) % W.  The W stack-1 outputs per unique tail are computed by ONE kernel step over
        U*Wmax scratch rows whose token rings are pre-filled with the window (keyring 0, so stack 2 reads the row's own
        S: realistic values for the exactness guard)."""
        TK = self.TK; s0 = TK.scr0; dev = rows.device; Bp = TK.Bp; n_scr = n_scr or TK.n_scr
        if rows.numel() == 0: return
        uq, inv = torch.unique(tail_ids, dim=0, return_inverse=True)
        U, Lt = uq.shape
        geo = self._geo()
        Wm = max(W for _, W, *_ in geo)
        J = U * Wm
        tb = self.__dict__.setdefault('_ptb', torch.zeros(Bp, dtype=torch.long, device=dev))
        for c0 in range(0, J, n_scr):
            n = min(n_scr, J - c0)
            jj = torch.arange(c0, c0 + n, device=dev)
            uj = jj // Wm; dj = jj % Wm; pj = Lt - 1 - dj                  # scratch row s0+i: window ending at pj of tail uj
            r = s0 + torch.arange(n, device=dev)
            for K, W, tr, Sr, kr, hs, hw, Vv, dS in geo:
                o = torch.arange(1, K, device=dev)
                jpos = pj[:, None] - o[None, :]                              # (n, K-1) earlier window positions
                tokv = torch.where(jpos >= 0, uq[uj[:, None], jpos.clamp(min=0)], torch.full_like(jpos, Vv))
                tr[jpos % K, r[:, None]] = tokv.to(tr.dtype)
                kr[:, r] = 0
                hs[r] = ((pj - 1) % K).to(hs.dtype); hw[r] = ((pj - 1) % W).to(hw.dtype)
            tb[r] = uq[uj, pj]
            TK.step(tb, n, r0=s0)
            for K, W, tr, Sr, kr, hs, hw, Vv, dS in geo:
                sel = dj < W
                if not bool(sel.any()): continue
                us, ps, rs = uj[sel], pj[sel], r[sel]
                m = inv[None, :] == us[:, None]                              # (n_sel, N) target rows of each window
                a, b = m.nonzero(as_tuple=True)
                Sr[(ps[a] % W), :, rows[b]] = Sr[(ps[a] % W), :, rs[a]]
        for K, W, tr, Sr, kr, hs, hw, Vv, dS, kt in self._geo(keytab=True):
            pk = torch.arange(Lt - K, Lt, device=dev)
            tr[(pk % K)[None, :], rows[:, None]] = uq[inv][:, pk].to(tr.dtype)
            pw = torch.arange(Lt - W, Lt, device=dev)
            kr[(pw % W)[None, :], rows[:, None]] = kt[uq[inv][:, pw]]
            hs[rows] = (Lt - 1) % K; hw[rows] = (Lt - 1) % W

    def can_emit(self):
        return PEMIT and PPRIME and self.TK.n_scr >= 2 * NPROBE

    @torch.no_grad()
    def prime_emit(self, tail_ids, src, rows):
        """Engine._prefill fast path. tail_ids (P, Lt) per prompt, rows (N,) target slots, src (N,) their prompt.
        Returns the gated implant output at the last tail token per prompt (= M.emit(E[tail])[0][:, -1]) and leaves the
        target rows in the state after the tail (= prime_ids(tail_ids[src], rows)): each unique tail is primed with its
        first Lt-1 tokens (prime_ids_par) into a probe row at the end of the scratch block, then ONE real kernel step on
        the last token gives the output and the full-tail state, which is copied to the target rows."""
        TK = self.TK; dev = tail_ids.device
        p0 = TK.scr0 + TK.n_scr - NPROBE
        uq, inv = torch.unique(tail_ids, dim=0, return_inverse=True)
        U = uq.shape[0]
        e_u = torch.empty(U, TK.nout, device=dev)
        tb = self.__dict__.setdefault('_etb', torch.zeros(TK.Bp, dtype=torch.long, device=dev))
        ri = inv[src] if rows.numel() else inv[:0]
        for c0 in range(0, U, NPROBE):
            n = min(NPROBE, U - c0); pr = p0 + torch.arange(n, device=dev)
            TK.reset_rows(pr)
            self.prime_ids_par(uq[c0:c0 + n, :-1], pr, n_scr=TK.n_scr - NPROBE)
            tb[pr] = uq[c0:c0 + n, -1]
            e_u[c0:c0 + n] = TK.step(tb, n, r0=p0)[p0:p0 + n]
            sel = (ri >= c0) & (ri < c0 + n)
            TK.copy_rows(p0 + ri[sel] - c0, rows[sel])
        e = _gate(e_u[:, self.o1:self.o1 + self.n1], e_u[:, self.o2:self.o2 + self.n2])
        return e[inv]

    def _geo(self, keytab=False):
        """per tool: K, W, tokring (K, Bp), Sring (W, dS, Bp), keyring (W, Bp), hs, hw, empty-slot id, dS[, keytab]."""
        TK = self.TK; Bp = TK.Bp; out = []
        for i, (hs, K, hw, W) in enumerate(TK._heads):
            tr, Sr, kr = TK._state[3 * i:3 * i + 3]
            dS = Sr.numel() // (W * Bp)
            g = [K, W, tr.view(K, Bp), Sr.view(W, dS, Bp), kr.view(W, Bp), hs, hw, TK._tokrings[i][1], dS]
            if keytab: g.append(TK._symbuf[i][1])
            out.append(g)
        return out

    @torch.no_grad()
    def prime_ids_seq(self, tail_ids, rows):
        """tail_ids (N, Lt) the last Lt = K+W tokens of prefix+question per target row; rows (N,) target slots.
        Identical tails (the G rows of one prompt) are primed once; the scan over the tail runs as one CUDA graph per
        32-row scratch block (Lt kernel steps), then the ring state is copied to the target slots."""
        TK = self.TK; s0 = TK.scr0; dev = rows.device
        if rows.numel() == 0: return
        uq, inv = torch.unique(tail_ids, dim=0, return_inverse=True)
        U = uq.shape[0]
        for c0 in range(0, U, TK.n_scr):
            n = min(TK.n_scr, U - c0); nb = (n + 31) // 32 * 32
            g, buf = self._prime_graph(nb, tail_ids.shape[1])
            buf[:n] = uq[c0:c0 + n]
            TK.reset_rows(torch.arange(s0, s0 + nb, device=dev))
            g.replay()
            sel = (inv >= c0) & (inv < c0 + n)
            TK.copy_rows(s0 + inv[sel] - c0, rows[sel])

    def _prime_graph(self, nb, Lt):
        key = (nb, Lt); pg = self.__dict__.setdefault('_pg', {})
        if key not in pg:
            TK = self.TK; s0 = TK.scr0; dev = self.E.device
            buf = torch.zeros(nb, Lt, dtype=torch.long, device=dev)
            tb = torch.zeros(TK.Bp, dtype=torch.long, device=dev)
            g = torch.cuda.CUDAGraph()
            st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                tb[s0:s0 + nb] = buf[:, 0]                  # warm the copy kernel outside capture
            torch.cuda.current_stream().wait_stream(st)
            with torch.cuda.graph(g):
                for t in range(Lt):
                    tb[s0:s0 + nb] = buf[:, t]
                    TK.step(tb, nb, r0=s0)
            pg[key] = (g, buf, tb)      # tb MUST stay alive: the graph writes the tail tokens into it and the kernel reads them
        return pg[key][:2]

    def prime_rows(self, emb_tail, rows):
        """called by Engine._prefill with embeddings; the matching token ids were stashed by the caller (set_tails)."""
        assert self._tails is not None, 'TKDual.prime_rows needs set_tails() first'
        ids = self._tails; self._tails = None
        assert ids.shape[0] == rows.shape[0]
        self.prime_ids(ids, rows)
        return None

    def gather(self, idx, n):
        self.TK.gather(idx, n)
