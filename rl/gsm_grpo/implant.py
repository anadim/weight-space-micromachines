"""The GSM calculator implant (calc_s4 + solved veto) driven with static buffers so it can live inside the decode CUDA graph.

Same math as micromachines.machine.Micromachine.step_sym / prime_sym / emit (reader -> conv window -> stack1 -> hard
attention hop -> stack2 -> one-hot e -> write = e @ D * gain). e (B, 14) is also the normalized "implant feature" the
arm-3 links read: a one-hot over the calculator's output symbols ('0'-'9', '.', '-', OS, OE), all zero when silent."""
import os, sys, torch, torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('WSM_ROOT', os.path.dirname(os.path.dirname(HERE)))   # repo root
sys.path.insert(0, REPO)
from micromachines import attach                      # noqa: E402
from micromachines.machine import BIG, GAMMA          # noqa: E402

MACHINE = os.environ.get('CALC_MACHINE', 'calc_s4')
# veto vector for Qwen3-0.6B-Base: built by `python scripts/solve_veto.py` (writes runs/veto_qwen3.pt, bit-identical rebuild, see VERIFY.md)
VETO = os.environ.get('WSM_VETO', os.path.join(REPO, 'runs', 'veto_qwen3.pt'))


class CalcImplant:
    def __init__(self, hf_model, tok, B_max, machine=MACHINE, fused=False, random=None, rseed=12345):
        zero_os = machine != 'calc_s3'                  # calc_s3 (GSM harness) writes the space itself; calc_s4: the model does
        A = attach(hf_model, tok, machines=(machine,), veto=VETO, veto_machines=(machine,), zero_os=zero_os)
        A.detach()                                      # we drive it ourselves; no hook on the HF model
        self.A, self.M = A, A.nets[machine]
        self.margins = A.margins
        M = self.M; dev = M.reader_w.device
        self.K, self.W, self.NS, self.key = M.K, M.W, M.NS, M.key
        self.dS = M.stack1(torch.zeros(1, M.K * M.NS, device=dev)).shape[-1]
        self.n_out = M.out_dirs.shape[0]
        self.D = (M.out_dirs * M.gain).float()          # (n_out, d) write directions incl. gain (veto row replaced, OS row 0)
        self.sym_h = torch.zeros(B_max, M.K, M.NS, device=dev)
        self.S_h = torch.zeros(B_max, M.W, self.dS, device=dev)
        self.key_h = torch.zeros(B_max, M.W, device=dev)
        self.dist = torch.arange(M.W - 1, -1, -1, device=dev).float()
        self.out_names = None
        self.fk = None; self.last_w = None
        self.random = random
        if random:                                       # "just more weights" control: same structure, values shuffled
            self.randomize(random, rseed)
        self.ring = False
        if fused == 'persist' or fused is True:          # one persistent kernel (default fused path)
            from implant_persist import PersistCalc
            self.fk = PersistCalc(self, B_max); self.ring = True
        elif fused == 'perrow':                          # one kernel, one program per 4 rows (slower)
            from implant_triton import FusedCalc
            self.fk = FusedCalc(self, B_max)

    @torch.no_grad()
    def randomize(self, which, seed):
        """which: 'digits' (writer rows for '0'-'9' permuted by a fixed derangement; fires exactly where calc_s4 fires)
        | 'stack1' | 'stack2' | 'all'. Within each compiled layer, permute the nonzero weight values among that
        layer's own nonzero positions (sparsity pattern, layer sizes and value multiset unchanged), and permute the
        layer's nonzero biases among its nonzero-bias units. Reader, window, hop and write directions untouched."""
        g = torch.Generator().manual_seed(seed)
        if which == 'digits':                           # scrambled writer: digit write directions permuted (derangement)
            while True:
                p = torch.randperm(10, generator=g)
                if not (p == torch.arange(10)).any(): break
            od = self.M.out_dirs
            od[:10] = od[p.to(od.device)].clone()
            self.D = (od * self.M.gain).float()
            self.digit_perm = p.tolist()
            return
        nets = {'stack1': [self.M.stack1], 'stack2': [self.M.stack2], 'all': [self.M.stack1, self.M.stack2]}[which]
        for net in nets:
            for l in range(net.L + 1):
                wv = getattr(net, f'wv{l}')
                p = torch.randperm(wv.numel(), generator=g).to(wv.device)
                wv.copy_(wv[p])
                b = getattr(net, f'b{l}'); nz = (b != 0).nonzero()[:, 0]
                if len(nz) > 1:
                    q = nz[torch.randperm(len(nz), generator=g).to(nz.device)]
                    b[nz] = b[q].clone()
            net._csr = {}

    def symbols(self, emb):
        return self.M.symbols(emb)

    @torch.no_grad()
    def prime_rows(self, emb_tail, rows):
        """emb_tail (N, Lt, d) embeddings of the last tokens of prefix+question; rows: LongTensor (N,) target slots.
        Returns e at the last position (N, n_out)."""
        M = self.M
        e = M.prime_sym(M.symbols(emb_tail))
        sym, S, key = M.state
        self.sym_h[rows] = sym
        if self.ring:                                    # logical oldest..newest -> ring slots
            idx = self.fk.ring_index()
            self.S_h[rows[:, None], idx[None, :]] = S; self.key_h[rows[:, None], idx[None, :]] = key
        else:
            self.S_h[rows] = S; self.key_h[rows] = key
        return e[:, -1]

    def step(self, emb, B):
        """emb (B, d) of the newest token; updates the first B static state rows in place; returns e (B, n_out)."""
        if self.fk is not None:                         # one fused Triton kernel (bit-identical)
            e, self.last_w = self.fk.step(emb, B)
            return e
        M = self.M
        s = M.symbols(emb)
        sym_h = torch.cat([self.sym_h[:B, 1:], s[:, None]], 1)
        win = sym_h.flip(1).reshape(B, 1, M.K * M.NS)
        S_h = torch.cat([self.S_h[:B, 1:], M.stack1(win)], 1)
        key_h = torch.cat([self.key_h[:B, 1:], s[:, M.key:M.key + 1]], 1)
        self.sym_h[:B].copy_(sym_h); self.S_h[:B].copy_(S_h); self.key_h[:B].copy_(key_h)
        C = torch.softmax(BIG * key_h - GAMMA * self.dist, -1)[:, None, :] @ S_h
        return M.stack2(torch.cat([win, C], -1))[:, 0]

    def gather(self, idx, n):
        for t in (self.sym_h, self.S_h, self.key_h):
            t[:n] = t[idx]

    @torch.no_grad()
    def full(self, emb):
        """Teacher-forced: e at every position of emb (N, T, d) (exactly what decode would produce there)."""
        e, _ = self.M.emit(emb)
        return e
