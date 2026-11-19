"""Rethink implant v2: a compiled MISTAKE DETECTOR + rethink writer (fixed tensors, no Python state machine).

The detector reads the host's states at the END OF THE REASONING, before the final answer is committed: the position
whose next-token prediction (host LM head) is '####' -- "read point" -- the first one in the answer. It is a small
fitted readout (logistic or 1-hidden-layer ReLU MLP) of the host's residual states at a few layers at that position.

Two intervention points (a fixed configuration choice):
  intervene='hash': if the detector fires at the read point, the implant writes "Hmm, wait. Let me rethink this.\\n"
                    (with a '\\n' in front if the current token does not end in '\\n') INSTEAD of '####'. Optional
                    budget cap: if the rethink has not written '####' after `cap` tokens, the implant writes ('\\n')'####'
                    so the host commits to an answer.
  intervene='end':  the detector's verdict is stored at the read point; the phrase is written where the host would END
                    its answer (exactly the v1 end detector), i.e. after the answer was committed (old behaviour).
mode='always' = the same circuit with the detector output replaced by 1.

Per position t (prompt positions in a prefill, one per decode step); everything below is a fixed tensor op:
  token tables (one-hot @ table): colon_t, answer_t, endsnl_t, hash_t ('####' in token)
  key_t   = AND(colon_t, answer_{t-1}, endsnl_{t-2})                "\\nAnswer:" (3-token conv window)
  x_t     = host final norm(h_t)  (LM-head input; includes any earlier implant's write, e.g. the calculator's)
  z_t     = x_t @ W_S^T  (fixed copy of the host's own LM-head rows: end rows E, hash rows H, all other rows C)
  hn_t    = step(max_H z - max_{not H} z)                           "the host is about to write '####'"
  end_t   = v1 end detector (max-pool + ReLU threshold)              (intervene='end' only)
  det_t   = readout(host states at the detector layers at t)        linear, or W2 relu(W1 (s - mu)/sd + b1) + b2
  pass_t  = step(det_t - thr)                                        (1 in 'always' mode)
  head A  first_t = hn_t AND NOT[most recent of {key, hn} strictly before t is an hn]     (first read point)
  'hash': fire_t = first_t AND pass_t          (fires at most once per answer: only the first read point can fire)
          rows: phrase / '\\n'+phrase by endsnl_t
          cap:  head D  pf_t = (pos_f + len(forced_f)) * [most recent of {key, fire, hash} at or before t is a fire]
                capfire_t = AND(fired_t, [pos_t - pf_t == cap])  -> rows '####' / '\\n####' by endsnl_t
  'end':  head B  gate_t = (first*pass) at the most recent of {key, first} at or before t
          cand_t = end_t AND gate_t;  head C  fire_t = cand_t AND NOT[most recent of {key, cand} before t is a cand]
  write_t = sum_j frow_{t-j} @ WT[j]   (conv window over the implant's own cached one-hot rows; v1 writer)
  h_t <- h_t + write_t  (pre-hook on the host's final norm; where write_t == 0, h_t is returned untouched)

Fitted pieces (disclosed): the detector (trained on GSM8K TRAIN generations + a scaled-number TRAIN analogue, see
scripts/rethink_v2/), the write directions (solved through the host's own norm + head, machine.make_out_dirs).
Hooks: one forward pre-hook on the host model (cache bookkeeping), forward hooks on the detector layers (read only,
return None), one pre-hook on the final norm (the circuit). The only Python `if` on data: "is the host's KV cache empty".
"""
import math, torch, torch.nn.functional as Fn
from .rethink_circuit import step_ge, step_gt, AND, hard_attend, BIG, K_END, PHRASE, RethinkCircuit

K_DET = 1e9


class RethinkCircuitV2:
    def __init__(self, model, tok, mode='gated', detector=None, intervene='hash', cap=0, gain=1e4, phrase=PHRASE,
                 dirs=None, trace=False):
        assert mode in ('always', 'gated') and intervene in ('hash', 'end', 'hybrid')
        assert intervene != 'hybrid' or mode == 'gated'
        self.model, self.mode, self.intervene, self.cap, self.gain = model, mode, intervene, int(cap), gain
        dev = model.device; self.dev = dev
        V = model.lm_head.weight.shape[0]
        T = lambda s: tok(s, add_special_tokens=False)['input_ids']; self.T = T
        eos = tok.convert_tokens_to_ids('<|endoftext|>')
        txt = [tok.decode([i]) for i in range(len(tok))] + [''] * max(0, V - len(tok))
        colon, answer = T(':'), T('Answer')
        tbl = torch.zeros(V, 4)
        tbl[colon[0], 0] = 1; tbl[answer[0], 1] = 1
        tbl[:, 2] = torch.tensor([t.endswith('\n') for t in txt], dtype=torch.float)
        tbl[:, 3] = torch.tensor(['####' in t for t in txt], dtype=torch.float)
        self.tbl = tbl.to(dev)
        H = [i for i, t in enumerate(txt) if '####' in t]; Hs = set(H)
        seqs = []
        def add(s):
            if s not in seqs: seqs.append(s)
            return seqs.index(s)
        if intervene in ('hash', 'hybrid'):
            self.r_ph_nl, self.r_ph = add(T(phrase)), add(T('\n' + phrase))       # current token ends in '\n' / not
            self.len_ph_nl, self.len_ph = len(T(phrase)), len(T('\n' + phrase))
            if self.cap:
                self.r_cap_nl, self.r_cap = add(T('####')), add(T('\n####'))
            E_any, E_nl = [], []
        if intervene in ('end', 'hybrid'):   # v1 end rows (identical construction)
            row_any, row_nl = {}, {}
            for i, t in enumerate(txt):
                if i == eos: stem = ''
                elif '\n\n' in t: stem = t.split('\n\n')[0]
                else: stem = None
                if stem is not None: row_any[i] = add(T(stem.rstrip() + '\n' + phrase))
                if t.startswith('\n') or (stem is not None and stem.strip() == ''):
                    row_nl[i] = add(T(phrase)) if (stem is None or stem.strip() == '') else add(T(stem.rstrip() + '\n' + phrase))
                elif stem is not None:
                    row_nl[i] = row_any[i]
            E_any = [i for i in sorted(row_any) if i not in Hs]; E_nl = [i for i in sorted(set(row_nl) - set(row_any)) if i not in Hs]
            E = E_any + E_nl
            R0 = len(seqs)
            M_any = torch.zeros(len(E), R0); M_nl = torch.zeros(len(E), R0)
            for k, i in enumerate(E):
                if i in row_any: M_any[k, row_any[i]] = 1
                if i in row_nl: M_nl[k, row_nl[i]] = 1
            self.M_any, self.M_nl = M_any.to(dev), M_nl.to(dev)
            self.R0 = R0
        self.nA, self.nN, self.nH = len(E_any), len(E_nl), len(H)
        Es = set(E_any) | set(E_nl) | Hs
        C = [c for c in range(V) if c not in Es]
        self.W_S = model.lm_head.weight.detach()[torch.tensor(E_any + E_nl + H + C, device=dev)].clone()
        self.idx_A, self.idx_N, self.idx_H = E_any, E_nl, H          # (rl_implants_v2: kept for the fused decode step)
        self.nS = self.nA + self.nN + self.nH
        self.R = len(seqs); self.seqs = seqs; self.L = max(len(s) for s in seqs)
        out_ids = sorted({t for s in seqs for t in s}); self.out_ids = out_ids
        D = None
        if isinstance(dirs, str):
            import os
            if os.path.exists(dirs):
                d = torch.load(dirs)
                if set(out_ids) <= set(d['ids']):
                    D = d['D'][torch.tensor([d['ids'].index(t) for t in out_ids])].to(dev)
        if D is None:
            with torch.enable_grad():
                D = RethinkCircuit._solve(model, out_ids)
            import os
            if isinstance(dirs, str) and not os.path.exists(dirs): torch.save({'ids': out_ids, 'D': D.cpu()}, dirs)
        self.D = D.float().to(dev)
        col = {t: k for k, t in enumerate(out_ids)}
        WT = torch.zeros(self.L, self.R, self.D.shape[1])
        for r, s in enumerate(seqs):
            for j, t in enumerate(s): WT[j, r] = self.D[col[t]].cpu() * gain
        self.WT = WT.to(dev)
        # ---- detector (fitted readout of host states) ----
        self.det_layers = []
        if mode == 'gated':
            P = torch.load(detector) if isinstance(detector, str) else detector
            self.det_layers = list(P['layers'])
            self.mu = P['mu'].float().to(dev); self.sd = P['sd'].float().to(dev)
            self.W1 = P['W1'].float().to(dev) if P.get('W1') is not None else None
            self.b1 = P['b1'].float().to(dev) if P.get('W1') is not None else None
            if intervene == 'hybrid':      # two readouts of the shared ReLU layer: sB (rethink at '####'), sA (after the answer)
                self.WB = P['WB'].double().to(dev).reshape(-1); self.bB = float(P['bB']); self.thB = float(P['thB'])
                self.WA = P['WA'].double().to(dev).reshape(-1); self.bA = float(P['bA']); self.thA = float(P['thA'])
                assert int(P['cap']) == self.cap
            else:
                self.W2 = P['W2'].double().to(dev).reshape(-1); self.b2 = float(P['b2'])
                self.thr = float(P['thr_logit'])
            nl = model.config.num_hidden_layers
            assert all(1 <= l <= nl for l in self.det_layers)
        self.trace = trace; self.cache = None; self.active = True; self._feat = {}
        self.margins = self.check_head()
        self.hooks = [model.model.register_forward_pre_hook(self._pre_model, with_kwargs=True),
                      model.model.norm.register_forward_pre_hook(self._pre_norm)]
        for l in self.det_layers:
            if l < model.config.num_hidden_layers:          # hidden_states[l] = output of decoder layer l-1
                self.hooks.append(model.model.layers[l - 1].register_forward_hook(self._grab(l)))

    def _grab(self, l):
        def f(module, args, out):
            if self.active: self._feat[l] = (out[0] if isinstance(out, tuple) else out).detach()
            return None
        return f

    def detach(self):
        for h in self.hooks: h.remove()
        self.cache = None

    def _pre_model(self, module, args, kwargs):
        if not self.active: return None
        ids = kwargs.get('input_ids', args[0] if args else None)
        past = kwargs.get('past_key_values')
        n_past = 0 if past is None else past.get_seq_length()
        if n_past == 0: self.cache = None
        else: assert self.cache is not None and self.cache['T'] == n_past, 'implant cache out of sync with the KV cache'
        self._ids = ids
        return None

    @torch.no_grad()
    def _pre_norm(self, module, args):
        h = args[0]
        if not self.active: return None
        w = self.forward(h, self._ids, module)
        mask = (w != 0).any(-1, keepdim=True)
        h2 = torch.where(mask, (h.float() + w).to(h.dtype), h)
        return (h2,) + tuple(args[1:])

    def detector(self, x):
        nl = self.model.config.num_hidden_layers
        s = torch.cat([(x if l == nl else self._feat[l]).float() for l in self.det_layers], -1)
        z = (s - self.mu) / self.sd
        if self.W1 is not None: z = torch.relu(z @ self.W1.T + self.b1)
        if self.intervene == 'hybrid':
            return z.double() @ self.WB + self.bB, z.double() @ self.WA + self.bA
        return z.double() @ self.W2 + self.b2

    def forward(self, h, ids, norm):
        B, n, _ = h.shape
        c = self.cache; t0 = 0 if c is None else c['T']
        fl = Fn.embedding(ids, self.tbl)
        x = norm.forward(h)
        prev = c['fl_tail'] if c is not None else torch.zeros(B, 2, 4, device=h.device)
        flw = torch.cat([prev, fl], 1)
        key = AND(flw[:, 2:, 0], flw[:, 1:-1, 1], flw[:, :-2, 2])
        hsh = fl[..., 3]; endsnl = fl[..., 2]
        # LM-head readout: E_any | E_nl | H | C, max-pools
        zS = Fn.linear(x, self.W_S[:self.nS]).float()
        zA, zN, zH = zS[..., :self.nA], zS[..., self.nA:self.nA + self.nN], zS[..., self.nA + self.nN:]
        mC = torch.cat([Fn.linear(x[:, i:i + 64], self.W_S[self.nS:]).amax(-1).float() for i in range(0, n, 64)], 1)
        neg = torch.full_like(mC, -BIG)
        mA = zA.amax(-1) if self.nA else neg
        mN = zN.amax(-1) if self.nN else neg
        mH = zH.amax(-1)
        hn = step_gt(mH - torch.maximum(torch.maximum(mC, mA), mN), K_END)
        if self.intervene == 'hybrid':
            sB, sA = self.detector(x)
            pasB = step_gt(sB - self.thB, K_DET).float(); pasA = step_gt(sA - self.thA, K_DET).float()
            pas = pasB
        elif self.mode == 'gated':
            pas = step_ge(self.detector(x) - self.thr, K_DET).float()
        else:
            pas = torch.ones_like(key)
        cat = lambda k, v: v if c is None else torch.cat([c[k], v], 1)
        KEY, HSH, HN = cat('key', key), cat('hash', hsh), cat('hn', hn)
        first = AND(hn, 1 - hard_attend(torch.relu(KEY + HN).clamp(max=1), HN, n, strict=True))
        FIRST = cat('first', first)
        new = {}
        frow = torch.zeros(B, n, self.R, device=h.device); fire = torch.zeros_like(key)
        if self.intervene in ('hash', 'hybrid'):
            fire = AND(first, pas)
            frow[..., self.r_ph_nl] = fire * endsnl; frow[..., self.r_ph] = fire * (1 - endsnl)
            if self.cap:
                pos = (t0 + torch.arange(n, device=h.device, dtype=torch.float32))[None].expand(B, n)
                FIRE = cat('fire', fire)
                pf_new = fire * (pos + self.len_ph_nl * endsnl + self.len_ph * (1 - endsnl))
                PF = cat('pf', pf_new)
                kk = torch.relu(KEY + FIRE + HSH).clamp(max=1)
                fired = hard_attend(kk, FIRE, n, strict=False)          # most recent of {key, fire, hash} is a fire
                pf = hard_attend(kk, PF, n, strict=False)
                d = pos - pf - self.cap
                eq = step_ge(d, 1e3) - step_ge(d - 1, 1e3)              # exact on integers: d == 0
                capf = AND(fired, eq)
                frow[..., self.r_cap_nl] = capf * endsnl; frow[..., self.r_cap] = capf * (1 - endsnl)
                new.update(fire=FIRE, pf=PF)
        if self.intervene in ('end', 'hybrid'):
            if self.intervene == 'hybrid': pas = pasA * (1 - pasB)      # A only where B does not fire
            e = torch.maximum(mA, mN - BIG * (1 - endsnl))
            cmax = torch.maximum(torch.maximum(mC, mH), mN - BIG * endsnl)
            end = step_gt(e - cmax, K_END)
            sE = torch.cat([zA, zN - BIG * (1 - endsnl)[..., None]], -1)
            oh = step_ge(sE - e[..., None], K_END)
            row = torch.relu(oh @ self.M_any - endsnl[..., None]) + torch.relu(oh @ self.M_nl - (1 - endsnl)[..., None])
            row = Fn.pad(row, (0, self.R - row.shape[-1]))
            PAS = cat('pass', pas)
            gate = hard_attend(torch.relu(KEY + FIRST).clamp(max=1), FIRST * PAS, n, strict=False)
            cand = AND(end, gate)
            CAND = cat('cand', cand)
            fireA = AND(cand, 1 - hard_attend(torch.relu(KEY + CAND).clamp(max=1), CAND, n, strict=True))
            frow = frow + row * fireA[..., None]
            fire = fire + fireA
            new.update(**{'pass': PAS, 'cand': CAND})
        tail = c['frow_tail'] if c is not None else torch.zeros(B, self.L - 1, self.R, device=h.device)
        fw = torch.cat([tail, frow], 1)
        w = torch.zeros(B, n, self.WT.shape[-1], device=h.device)
        for j in range(self.L):
            w = w + fw[:, self.L - 1 - j:self.L - 1 - j + n] @ self.WT[j]
        self.cache = dict(T=t0 + n, key=KEY, hash=HSH, hn=HN, first=FIRST, fl_tail=flw[:, -2:],
                          frow_tail=fw[:, -(self.L - 1):], **new)
        if self.trace:
            self.last_trace = dict(hn=hn.detach(), first=first.detach(), fire=fire.detach(), wrote=(w != 0).any(-1),
                                   pas=pas.detach(), endsnl=endsnl.detach(), frow=frow.detach())
        self._feat = {}
        return w

    @torch.no_grad()
    def check_head(self):
        m = self.model; dt = m.lm_head.weight.dtype
        h = torch.randn(64, m.config.hidden_size, device=m.device, dtype=dt) * 10
        worst = float('inf')
        for j, t in enumerate(self.out_ids):
            lg = m.lm_head(m.model.norm(h + (self.D[j] * self.gain).to(dt))).float()
            own = lg[:, t].clone(); lg[:, t] = float('-inf')
            worst = min(worst, (own - lg.max(-1).values).min().item())
        assert worst > 0, 'a rethink write does not win the argmax'
        return worst

    def n_params(self):
        ts = [self.tbl, self.W_S, self.WT] + ([self.M_any, self.M_nl] if self.intervene in ('end', 'hybrid') else [])
        if self.mode == 'gated':
            ts += [self.mu, self.sd] + ([self.WB, self.WA] if self.intervene == 'hybrid' else [self.W2]) + ([self.W1, self.b1] if self.W1 is not None else [])
        return sum(t.numel() for t in ts)
