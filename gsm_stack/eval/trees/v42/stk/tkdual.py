"""calc_s4 + chain2 (implant2.DualImplant) decode steps in engine2's exact tools kernel (tk3, NVRTC; integer, differential,
event-driven; bit-exact to the torch CSR stacks, verified on calc_s4 in engine2 and checked here end to end).

TKDual wraps a DualImplant: step() runs both machines in ONE kernel launch over all rows and applies the same fixed gate
(chain2 has priority) as DualImplant.step; prime_rows() primes admitted rows by scanning their last K+W prompt tokens
through the kernel in scratch rows, then copies the ring state to the target slots (= the torch prime: the state after
the tail). Everything else (D, M.emit for the first-token e, n_out, ...) is delegated to the DualImplant.
The reader reads the host embedding E: refresh() rebuilds the symbol tables after a weight update (asserted 0/1)."""
import torch
import tk
from implant2 import _gate


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
