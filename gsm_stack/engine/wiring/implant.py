"""The GSM calculator implant (calc_s4 + solved veto) driven with static buffers so it can live inside the decode CUDA graph.

Same math as micromachines.machine.Micromachine.step_sym / prime_sym / emit (reader -> conv window -> stack1 -> hard
attention hop -> stack2 -> one-hot e -> write = e @ D * gain). e (B, 14) is also the normalized "implant feature" the
arm-3 links read: a one-hot over the calculator's output symbols ('0'-'9', '.', '-', OS, OE), all zero when silent."""
import os, sys, torch, torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('WSM_ROOT') or os.path.abspath(os.path.join(HERE, '..', '..', '..'))   # repo root (micromachines/, data/); launchers set WSM_ROOT
sys.path.insert(0, REPO)
from micromachines import attach                      # noqa: E402
from micromachines.machine import BIG, GAMMA          # noqa: E402

MACHINE = os.environ.get('CALC_MACHINE', 'calc_s4')

# ---- call-form calculator: '<calc>48 * 2</calc> 96' (2026-11-19) -------------------------------------------------------
# Same exact compiled circuit as calc_s (spaced build); only the start/end symbols are re-keyed. '<calc>' and '</calc>' are
# 3 ordinary tokens each in Qwen3 (['<'|' <', 'calc', '>'], ['</', 'calc', '>']), so a token filter sits in front of the
# reader: '<', ' <', '</' and a 'calc' right after them are SKIPPED (the machine state does not shift, e = 0 there); the
# closing '>' of '<calc>' is fed as CS (id of '<') and the closing '>' of '</calc>' as CE (id of '</'). Raw '<', ' <', '</'
# never reach the reader, so the two feed ids are unambiguous. After CE the machine writes ' ' (OS) + the exact digits,
# then the solved end-of-number veto (the host writes its own next token).
from micromachines.specs import SPECS as _SPECS                       # noqa: E402
from micromachines.circuits import calc as _Ccirc                    # noqa: E402
_SPECS['calc_tag'] = dict(_SPECS['calc_s2'], build=lambda: _Ccirc.build(spaced=True, dotveto=True),
                          text={**_SPECS['calc_s2']['text'], 'CS': '<', 'CSN': '¶', 'CE': '</'})   # CSN (second start class, OR-ed with CS) on a token that never occurs
TAG_IDS = dict(OPEN=(27, 366), CLOSE=522, CALC=26586, GT=29, FEED_CS=27, FEED_CE=522)


def tag_filter(ids, T=TAG_IDS):
    """python reference: ids (list) -> feed (list, None = skipped). State: 1 '<' seen, 2 '</', 3 '<calc', 4 '</calc'."""
    st, out = 0, []
    for t in ids:
        if t in T['OPEN']: out.append(None); st = 1
        elif t == T['CLOSE']: out.append(None); st = 2
        elif t == T['CALC'] and st in (1, 2): out.append(None); st += 2
        elif t == T['GT'] and st == 3: out.append(T['FEED_CS']); st = 0
        elif t == T['GT'] and st == 4: out.append(T['FEED_CE']); st = 0
        else: out.append(t); st = 0
    return out
VETO = os.environ.get('WSM_VETO', os.path.join(REPO, 'runs', 'veto_qwen3.pt'))   # python scripts/solve_veto.py


class CalcImplant:
    def __init__(self, hf_model, tok, B_max, machine=MACHINE, fused=False, random=None, rseed=12345):
        self.tagf = machine == 'calc_tag'
        if self.tagf: assert not fused, 'calc_tag: token filter needs the torch step (fused=0)'
        zero_os = machine not in ('calc_s3', 'calc_tag')  # calc_s3 / calc_tag write the space themselves; calc_s4: the model does
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
        self.fst = torch.zeros(B_max, dtype=torch.long, device=dev)      # calc_tag filter state per row
        self.Eref = None                                                  # host input embedding (set by the engine)
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

    def step(self, emb, B, tok=None):
        """emb (B, d) of the newest token; updates the first B static state rows in place; returns e (B, n_out)."""
        if self.tagf: return self._step_tag(emb, B, tok)
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

    def _step_tag(self, emb, B, tok):
        T = TAG_IDS; M = self.M; st = self.fst[:B]
        is_open = (tok == T['OPEN'][0]) | (tok == T['OPEN'][1]); is_close = tok == T['CLOSE']
        is_calc = (tok == T['CALC']) & ((st == 1) | (st == 2)); is_gt = tok == T['GT']
        ev_cs = is_gt & (st == 3); ev_ce = is_gt & (st == 4)
        skip = is_open | is_close | is_calc
        self.fst[:B] = torch.where(is_open, 1, torch.where(is_close, 2, torch.where(is_calc, st + 2, torch.zeros_like(st))))
        feed = torch.where(ev_cs, T['FEED_CS'], torch.where(ev_ce, T['FEED_CE'], tok))
        emb = torch.where((ev_cs | ev_ce)[:, None], self.Eref[feed].to(emb.dtype), emb)
        s = M.symbols(emb)
        sym_h = torch.cat([self.sym_h[:B, 1:], s[:, None]], 1)
        win = sym_h.flip(1).reshape(B, 1, M.K * M.NS)
        S_h = torch.cat([self.S_h[:B, 1:], M.stack1(win)], 1)
        key_h = torch.cat([self.key_h[:B, 1:], s[:, M.key:M.key + 1]], 1)
        C = torch.softmax(BIG * key_h - GAMMA * self.dist, -1)[:, None, :] @ S_h
        e = M.stack2(torch.cat([win, C], -1))[:, 0]
        k3 = skip[:, None, None]
        self.sym_h[:B].copy_(torch.where(k3, self.sym_h[:B], sym_h)); self.S_h[:B].copy_(torch.where(k3, self.S_h[:B], S_h))
        self.key_h[:B].copy_(torch.where(skip[:, None], self.key_h[:B], key_h))
        return torch.where(skip[:, None], torch.zeros_like(e), e)

    # ---- sequence-level (prompt prime / teacher forcing) with the filter: compact, emit, scatter back ----
    def _compact(self, ids):
        rows = ids.tolist(); feeds, pos = [], []
        for r in rows:
            f = tag_filter(r); keep = [i for i, x in enumerate(f) if x is not None]
            feeds.append([f[i] for i in keep]); pos.append(keep)
        Tk = max(1, max(map(len, feeds)))
        F_ = torch.tensor([x + [0] * (Tk - len(x)) for x in feeds], device=ids.device)
        ri = torch.tensor([j for j, p in enumerate(pos) for _ in p], device=ids.device, dtype=torch.long)
        ki = torch.tensor([k for p in pos for k in range(len(p))], device=ids.device, dtype=torch.long)
        pi = torch.tensor([i for p in pos for i in p], device=ids.device, dtype=torch.long)
        return F_, ri, ki, pi, [len(f) for f in feeds]

    def emit_ids(self, ids, E=None):
        """teacher-forced e at every position of ids (N, T); differentiable w.r.t. the reader / stacks when grad is on."""
        E = self.Eref if E is None else E
        F_, ri, ki, pi, _ = self._compact(ids)
        ec = self.M.emit(E[F_])[0]
        e = ec.new_zeros(ids.shape[0], ids.shape[1], ec.shape[-1])
        return e.index_put((ri, pi), ec[ri, ki])

    @torch.no_grad()
    def prime_rows_ids(self, tail_ids, rows):
        """tail_ids (N, Lt): last raw tokens of prefix+question (long enough to hold K+W fed tokens). -> e at the last position."""
        F_, ri, ki, pi, nk = self._compact(tail_ids)
        assert min(nk) >= self.K + self.W, 'tail too short after the filter'
        N = F_.shape[0]
        last = torch.tensor(nk, device=F_.device) - 1
        take = (last[:, None] - torch.arange(self.K + self.W - 1, -1, -1, device=F_.device)[None])     # last K+W fed tokens
        Fl = F_.gather(1, take)
        e = self.prime_rows(self.Eref[Fl], rows)
        self.fst[rows] = 0
        raw_last_fed = tail_ids[:, -1] == F_.gather(1, last[:, None])[:, 0]
        return torch.where(raw_last_fed[:, None], e, torch.zeros_like(e))

    def gather(self, idx, n):
        for t in (self.sym_h, self.S_h, self.key_h, self.fst):
            t[:n] = t[idx]

    @torch.no_grad()
    def full(self, emb):
        """Teacher-forced: e at every position of emb (N, T, d) (exactly what decode would produce there)."""
        e, _ = self.M.emit(emb)
        return e
