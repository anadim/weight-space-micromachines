"""The `rethink` implant compiled into fixed tensors (no Python state machine).

Behaviour (same as micromachines/rethink.py): when the frozen host is about to END its answer (its next token would make
the first blank line, or is <|endoftext|>), the implant writes "\\nHmm, wait. Let me rethink this.\\n" (exact tokens of
rethink.py: stem.rstrip() + '\\n' + PHRASE) instead, once per answer, then is silent. Gated variant: only if the
layer-28 probe, read at the first '####' token of the answer, is >= its threshold.

Everything it computes at inference is a fixed tensor op evaluated inside two generic forward hooks, plus its own
per-position tensor cache that lives and is reset exactly like the host's KV cache (new cache <=> the host forward
starts with an empty past_key_values). Per position t (all prompt positions in a prefill, one position per decode step):

  token tables (lookup = one-hot @ table, exact):  colon_t, answer_t, endsnl_t (token ends with '\\n'), hash_t ('####')
  key_t      = AND(colon_t, answer_{t-1}, endsnl_{t-2})              a 3-token conv window: the "\\nAnswer:" of the prompt
  x_t        = host final norm(h_t)                                   (= hidden_states[28], the LM head input)
  z_t        = x_t @ W_S^T                                            the host's own LM-head rows (fixed copy, bf16):
                                                                      E_any (565 '\\n\\n' tokens + <|endoftext|>),
                                                                      E_nl (18 tokens starting with '\\n'), C (all other
                                                                      rows by default; optionally a shortlist)
  end_t      = step( max(E_any, E_nl if endsnl_t) - max(C, E_nl if not endsnl_t) )   max-pool + ReLU threshold
  row_t      = one-hot of the winning end token (ReLU threshold vs the E max) @ lookup table -> one-hot over the 407
               forced sequences (which depends on endsnl_t exactly as in rethink.py)
  pass_t     = step(probe . x_t + b - logit(thr))   (float64; = 1 everywhere in 'always' mode)
  head A     firsthash_t = hash_t AND NOT [most recent of {key, hash} strictly before t is a hash]
  head B     gate_t      = value (firsthash*pass) at the most recent of {key, firsthash} at or before t
               NOTE: with arm_at_hash=True (the default, and what the reported full runs used) this applies in 'always'
               mode too (pass = 1), so 'always' arms only after the answer's first '####'; an answer that ends without
               any '####' gets no phrase (rethink.py's 'always' arms at "Answer:"). arm_at_hash=False: gate = 1.
  cand_t     = AND(end_t, gate_t)
  head C     fire_t      = cand_t AND NOT [most recent of {key, cand} strictly before t is a cand]   (once per answer)
  frow_t     = fire_t * row_t
  write_t    = sum_{j<L} frow_{t-j} @ WT[j]        an L-wide conv window over the implant's own cached frow;
                                                   WT[j][r] = gain * solved direction of token j of sequence r (0 past its end)
  h_t       <- h_t + write_t   (pre-hook on the host's final norm; where write_t == 0, h_t is returned untouched)

Heads A-C are hard attention (softmax(BIG*key - GAMMA*dist), as Micromachine.attend) over cached per-position scalars.
No data-dependent Python branching: the only Python `if` is "is the host's KV cache empty" (start a new implant cache).

Fitted / solved pieces: the write directions (Adam through the host's own norm + head, machine.make_out_dirs), the probe
(sklearn logistic regression, results/rethink/rethink_probe_L28.pt). With C = all other head rows the end detector is the
exact "argmax is an end token" rule up to ties/rounding (z is recomputed from a copy of the head rows, so it can differ from
the host's own logits in the last bf16 bit); agreement with the host's true argmax is measured, not assumed.
"""
import torch, torch.nn.functional as Fn
from .machine import make_out_dirs

PHRASE = "Hmm, wait. Let me rethink this.\n"
BIG, GAMMA = 1e6, 300.0
K_END = 1e6          # ReLU-threshold slope on float32 logit gaps
K_PROBE = 1e9        # on the float64 probe logit


def step_ge(x, k):   # 1 if x >= 0, 0 if x <= -1/k  (= relu(1+kx) - relu(kx))
    return torch.relu(1 + k * x) - torch.relu(k * x)


def step_gt(x, k):   # 1 if x >= 1/k, 0 if x <= 0
    return torch.relu(k * x) - torch.relu(k * x - 1)


def AND(*xs):        # exact on {0,1}
    return torch.relu(sum(xs) - (len(xs) - 1))


def hard_attend(key_all, val_all, n_new, strict):
    """For each of the last n_new positions t: the value at the most recent key position s (s < t if strict else s <= t).
    key_all, val_all: (B, T). Returns (B, n_new). Callers use values that are 0 at non-key positions, so with no key in
    range the result is 0; a zero-valued sentinel column keeps a row with no admissible position (t = 0, strict) finite."""
    B, T = key_all.shape
    t = torch.arange(T - n_new, T, device=key_all.device)
    s = torch.arange(T, device=key_all.device)
    dist = (t[:, None] - s[None, :]).float()                                  # (n_new, T)
    ok = (dist > 0) if strict else (dist >= 0)
    score = BIG * key_all[:, None, :].float() - GAMMA * dist                  # (B, n_new, T)
    score = score.masked_fill(~ok, float('-inf'))
    score = torch.cat([score, torch.zeros_like(score[..., :1]) + (-BIG)], -1)  # sentinel: wins only if no key at all
    v = torch.cat([val_all.float(), torch.zeros_like(val_all[:, :1]).float()], -1)
    return (torch.softmax(score, -1) * v[:, None, :]).sum(-1)


class RethinkCircuit:
    def __init__(self, model, tok, mode='always', probe=None, shortlist=None, gain=1e4, phrase=PHRASE, dirs=None,
                 trace=False, arm_at_hash=True):
        assert mode in ('always', 'gated')
        self.model, self.mode, self.gain = model, mode, gain
        dev = model.device; self.dev = dev
        V = model.lm_head.weight.shape[0]
        T = lambda s: tok(s, add_special_tokens=False)['input_ids']
        eos = tok.convert_tokens_to_ids('<|endoftext|>')
        txt = [tok.decode([i]) for i in range(len(tok))] + [''] * max(0, V - len(tok))
        # ---- token tables (V, 4): colon, answer, endsnl, hash ----
        colon, answer = T(':'), T('Answer')
        assert len(colon) == 1 and len(answer) == 1
        tbl = torch.zeros(V, 4)
        tbl[colon[0], 0] = 1; tbl[answer[0], 1] = 1
        tbl[:, 2] = torch.tensor([t.endswith('\n') for t in txt], dtype=torch.float)
        tbl[:, 3] = torch.tensor(['####' in t for t in txt], dtype=torch.float)
        self.tbl = tbl.to(dev)
        # ---- end tokens and forced sequences (identical construction to rethink.py, budget rows dropped) ----
        seqs = []
        def add(text):
            s = T(text)
            if s not in seqs: seqs.append(s)
            return seqs.index(s)
        row_any, row_nl = {}, {}
        for i, t in enumerate(txt):
            if i == eos: stem = ''
            elif '\n\n' in t: stem = t.split('\n\n')[0]
            else: stem = None
            if stem is not None: row_any[i] = add(stem.rstrip() + '\n' + phrase)
            if t.startswith('\n') or (stem is not None and stem.strip() == ''):
                row_nl[i] = add(phrase) if (stem is None or stem.strip() == '') else add(stem.rstrip() + '\n' + phrase)
            elif stem is not None:
                row_nl[i] = row_any[i]
        E_any = sorted(row_any); E_nl = sorted(set(row_nl) - set(row_any))
        self.E_any, self.E_nl = E_any, E_nl
        E = E_any + E_nl; nE = len(E); R = len(seqs); self.R = R; self.nE_any = len(E_any)
        M_any = torch.zeros(nE, R); M_nl = torch.zeros(nE, R)
        for k, i in enumerate(E):
            if i in row_any: M_any[k, row_any[i]] = 1
            if i in row_nl: M_nl[k, row_nl[i]] = 1
        self.M_any, self.M_nl = M_any.to(dev), M_nl.to(dev)
        # ---- detector rows: the host's own LM-head rows. E first; competitors C = a shortlist, or (default) ALL other rows
        Eset = set(E)
        if shortlist is None:
            C = [c for c in range(V) if c not in Eset]
        else:
            C = [c for c in shortlist if c not in Eset]
        self.C = C
        # fixed copy of the head rows in the host's dtype (so the logits are the host's own bf16 logits)
        self.W_S = model.lm_head.weight.detach()[torch.tensor(E + C, device=dev)].clone()      # (nE + nC, d)
        self.nE = nE
        # ---- writer: WT[j, r] = gain * dir(token j of sequence r) ----
        self.L = max(len(s) for s in seqs)
        out_ids = sorted({t for s in seqs for t in s}); self.out_ids = out_ids
        if dirs is not None and isinstance(dirs, str):
            import os
            if os.path.exists(dirs):
                d = torch.load(dirs)
                assert d['ids'] == out_ids
                D = d['D'].to(dev)
            else:
                D = self._solve(model, out_ids); torch.save({'ids': out_ids, 'D': D.cpu()}, dirs)
        else:
            D = self._solve(model, out_ids)
        self.D = D.float().to(dev)
        col = {t: k for k, t in enumerate(out_ids)}
        WT = torch.zeros(self.L, R, D.shape[1])
        for r, s in enumerate(seqs):
            for j, t in enumerate(s): WT[j, r] = self.D[col[t]].cpu() * gain
        self.WT = WT.to(dev)
        self.seqs = seqs
        # ---- probe (fitted) ----
        if mode == 'gated':
            P = torch.load(probe) if isinstance(probe, str) else probe
            self.pw = (P['w'] / P['sd']).double().to(dev)
            self.pb = float(P['b'] - (P['w'] * P['mu'] / P['sd']).sum())
            thr = float(P['thr']); self.thr = thr
            import math
            self.zthr = math.log(thr / (1 - thr))
            assert int(P.get('layer', 28)) == model.config.num_hidden_layers
        self.arm_at_hash = arm_at_hash
        self.trace = trace; self.cache = None; self.active = True
        self.margins = self.check_head()                          # before the hooks exist
        self.h_model = model.model.register_forward_pre_hook(self._pre_model, with_kwargs=True)
        self.h_norm = model.model.norm.register_forward_pre_hook(self._pre_norm)

    @staticmethod
    def _solve(model, out_ids):
        with torch.enable_grad():
            return torch.cat([make_out_dirs(model, out_ids[c:c + 64]) for c in range(0, len(out_ids), 64)])

    def detach(self):
        self.h_model.remove(); self.h_norm.remove(); self.cache = None

    # ---------------- hooks ----------------
    def _pre_model(self, module, args, kwargs):
        if not self.active: return None                         # (silent check only: the implant is bypassed)
        ids = kwargs.get('input_ids', args[0] if args else None)
        past = kwargs.get('past_key_values')
        n_past = 0 if past is None else past.get_seq_length()
        if n_past == 0: self.cache = None                      # the host starts a new KV cache -> new implant cache
        else: assert self.cache is not None and self.cache['T'] == n_past, 'implant cache out of sync with the KV cache'
        self._ids = ids
        return None

    @torch.no_grad()
    def _pre_norm(self, module, args):
        h = args[0]
        if not self.active: return None                         # (silent check only: the hook is bypassed)
        w = self.forward(h, self._ids, module)
        mask = (w != 0).any(-1, keepdim=True)
        h2 = torch.where(mask, (h.float() + w).to(h.dtype), h)
        return (h2,) + tuple(args[1:])

    # ---------------- the circuit ----------------
    def forward(self, h, ids, norm):
        B, n, _ = h.shape
        c = self.cache
        fl = Fn.embedding(ids, self.tbl)                         # (B, n, 4)
        x = norm.forward(h)                                      # host final norm, own dtype (hidden_states[28])
        # -- key: "\nAnswer:" (3-token window over flags, cached tail of 2) --
        prev = c['fl_tail'] if c is not None else torch.zeros(B, 2, 4, device=h.device)
        flw = torch.cat([prev, fl], 1)                           # (B, n+2, 4)
        key = AND(flw[:, 2:, 0], flw[:, 1:-1, 1], flw[:, :-2, 2])
        hsh = fl[..., 3]; endsnl = fl[..., 2]
        # -- end detector: host LM-head rows, max-pool, ReLU threshold --
        zE = Fn.linear(x, self.W_S[:self.nE]).float()            # (B, n, nE)
        zA, zN = zE[..., :self.nE_any], zE[..., self.nE_any:]
        mC = torch.cat([Fn.linear(x[:, i:i + 64], self.W_S[self.nE:]).amax(-1).float()       # max-pool over C
                        for i in range(0, n, 64)], 1)                                         # (chunked over positions)
        mN = zN.amax(-1)
        e = torch.maximum(zA.amax(-1), mN - BIG * (1 - endsnl))
        cmax = torch.maximum(mC, mN - BIG * endsnl)
        end = step_gt(e - cmax, K_END)
        sE = torch.cat([zA, zN - BIG * (1 - endsnl)[..., None]], -1)
        oh = step_ge(sE - e[..., None], K_END)                   # one-hot of the winning end token
        row = torch.relu(oh @ self.M_any - endsnl[..., None]) + torch.relu(oh @ self.M_nl - (1 - endsnl)[..., None])
        # -- probe --
        if self.mode == 'gated':
            p = x.double() @ self.pw + self.pb
            pas = step_ge(p - self.zthr, K_PROBE).float()
        else:
            pas = torch.ones_like(key)
        # -- cache append (per-position scalars) --
        cat = lambda k, v: v if c is None else torch.cat([c[k], v], 1)
        KEY, HSH, PAS = cat('key', key), cat('hash', hsh), cat('pass', pas)
        # head A: first '####' since the last key
        fh_new = AND(hsh, 1 - hard_attend(torch.relu(KEY + HSH).clamp(max=1), HSH, n, strict=True))
        FH = cat('fh', fh_new)
        # head B: gate = probe pass at the first '####' of this answer
        gate = hard_attend(torch.relu(KEY + FH).clamp(max=1), FH * PAS, n, strict=False)
        if self.mode == 'always' and not self.arm_at_hash:       # (a fixed configuration choice, not data-dependent)
            gate = torch.ones_like(gate)
        cand = AND(end, gate)
        CAND = cat('cand', cand)
        # head C: once per answer
        fire = AND(cand, 1 - hard_attend(torch.relu(KEY + CAND).clamp(max=1), CAND, n, strict=True))
        frow = row * fire[..., None]                             # (B, n, R)
        # writer: L-wide conv window over cached frow
        tail = c['frow_tail'] if c is not None else torch.zeros(B, self.L - 1, self.R, device=h.device)
        fw = torch.cat([tail, frow], 1)                          # (B, n+L-1, R)
        w = torch.zeros(B, n, self.WT.shape[-1], device=h.device)
        for j in range(self.L):                                  # offset j: frow at t-j
            w = w + fw[:, self.L - 1 - j:self.L - 1 - j + n] @ self.WT[j]
        self.cache = {'T': (0 if c is None else c['T']) + n, 'key': KEY, 'hash': HSH, 'pass': PAS, 'fh': FH, 'cand': CAND,
                      'fl_tail': flw[:, -2:], 'frow_tail': fw[:, -(self.L - 1):]}
        if self.trace:
            tr = dict(end=end, fire=fire, gate=gate, wrote=(w != 0).any(-1), probe=(p if self.mode == 'gated' else None),
                      fh=fh_new, endsnl=endsnl)
            self.last_trace = {k: (v.detach() if v is not None else None) for k, v in tr.items()}
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
        return sum(t.numel() for t in (self.tbl, self.M_any, self.M_nl, self.W_S, self.WT)) + (self.pw.numel() + 2 if self.mode == 'gated' else 0)
