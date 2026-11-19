"""Topic blocker compiled fully into fixed tensors (no Python state machine). Two modes:

  mode='text_stop'  (= the emulated 'probe_text_stop'): a last-prompt-token logistic probe GATE; if it fires, write the
                    decline phrase + <|im_end|> starting at the last prompt position, then stop.
  mode='monitor'    (= the emulated 'monitor'): a per-generated-token probe; after k CONSECUTIVE hits, write
                    '\n\n' + phrase + <|im_end|>.

What used to be Python (fc.py) and how it is now a fixed tensor op, per position t (prompt positions in a prefill,
one per decode step), evaluated inside forward hooks + a per-position cache that lives/reset like the host KV cache:
  * token table (one-hot @ tbl):  is_nl2 ('\n\n'), is_endthink ('</think>')
  * key_t = AND(is_nl2_t, is_endthink_{t-1})      <- the last prompt position (right before the assistant turn),
            found by a 2-token conv window, NOT by a Python step==0 test.
  * probe score z_t = w . hidden[L]_t + b  (hidden[L] stashed by a read-only layer hook; L = gate/monitor layer).
  * TEXT mode:
      graw_t = step(z_t - logit(thr));  G_t = graw at the most recent key <= t      (HARD ATTENTION latch, not a Python dict)
      j_t    = pos_t - (position of the most recent key)                            (position counter = hard attention)
      frow_t[s] = AND(G_t, haskey_t, [j_t == s])   for phrase slot s                (which phrase token to write)
  * MONITOR mode:
      hit_t  = step(z_t - logit(thr));  ehit_t = AND(hit_t, haskey_t)               (count only from the last prompt token)
      krun_t = AND(ehit_t, ehit_{t-1}, ..., ehit_{t-k+1})                           (k consecutive = width-k conv window)
      fire_t = AND(krun_t, NOT[a krun strictly before t])                           (fire once)
      j_t    = pos_t - (position of the fire);  frow_t[s] = AND(hasfire_t, [j_t == s])
  * write_t = frow_t @ WT,   WT[s] = gain * solved_direction(seq[s])                (added to the host residual before the final norm)
Where frow_t == 0 the host residual is returned untouched -> logits bit-identical to the plain host. Fitted/solved
pieces (disclosed, same as the emulated fence): the logistic probe (w,b,thr); the write directions (make_out_dirs,
Adam through the host's own norm+head). Hooks: _pre_model (cache bookkeeping), a read-only hook on the probe layer,
and the final-norm pre-hook (the circuit). The only non-weight pieces are the plain greedy decode loop and the hook
on/off switch."""
import torch, torch.nn.functional as Fn
from fc import make_out_dirs
from micromachines.rethink_circuit import hard_attend, AND, step_ge, step_gt, BIG

# Saturating ReLU-threshold: 1 for x >= 1/KZ, 0 for x <= 0, exact in float32 over the probe's dynamic range
# (a slope as steep as 1e9 loses the '+1' in float32 when k*x >> 2^24, which silently zeroes the unit).
KZ = 1e4


def thresh(x):              # 1 iff x >= ~0  (step_gt saturates at 1 for large x)
    return step_gt(x, KZ)


def eq0(d):                 # 1 iff integer d == 0
    return step_ge(d, 1e3) - step_ge(d - 1, 1e3)


class BlockerCircuit:
    def __init__(self, model, tok, mode, probe, phrase, k=3, gain=1e4, trace=False):
        assert mode in ('text_stop', 'monitor')
        self.model, self.mode, self.gain, self.k, self.trace = model, mode, gain, int(k), trace
        dev = model.device
        V = model.lm_head.weight.shape[0]
        self.L = int(probe['layer'])
        self.pw = torch.tensor(probe['w'], dtype=torch.float32, device=dev)
        self.pb = float(probe['b']); thr = float(probe['thr'])
        self.logit_thr = float(torch.logit(torch.tensor(thr)))
        im_end = tok.convert_tokens_to_ids('<|im_end|>')
        T = lambda s: tok(s, add_special_tokens=False)['input_ids']
        if mode == 'text_stop':
            self.seq = T(phrase) + [im_end]
        else:
            self.seq = T('\n\n') + T(phrase) + [im_end]
        self.Lseq = len(self.seq)
        # token table: [is_nl2, is_endthink]
        nl2 = T('\n\n'); assert len(nl2) == 1
        endthink = tok.convert_tokens_to_ids('</think>')
        tbl = torch.zeros(V, 2, device=dev)
        tbl[nl2[0], 0] = 1; tbl[endthink, 1] = 1
        self.tbl = tbl
        # write directions for the phrase tokens
        out_ids = sorted(set(self.seq))
        with torch.enable_grad():
            Dm = make_out_dirs(model, out_ids)
        Dd = {t: Dm[j] for j, t in enumerate(out_ids)}
        WT = torch.zeros(self.Lseq, model.config.hidden_size, device=dev)
        for s, t in enumerate(self.seq):
            WT[s] = Dd[t] * gain
        self.WT = WT
        self.cache = None; self.active = True; self._feat = None; self._ids = None
        self.margins = self.check_head(out_ids, Dd)
        self.hooks = [model.model.register_forward_pre_hook(self._pre_model, with_kwargs=True),
                      model.model.layers[self.L - 1].register_forward_hook(self._grab),
                      model.model.norm.register_forward_pre_hook(self._pre_norm)]

    @torch.no_grad()
    def check_head(self, out_ids, Dd):
        m = self.model; dt = m.lm_head.weight.dtype
        h = torch.randn(64, m.config.hidden_size, device=m.device, dtype=dt) * 10
        worst = float('inf')
        for t in out_ids:
            lg = m.lm_head(m.model.norm(h + (Dd[t] * self.gain).to(dt))).float()
            own = lg[:, t].clone(); lg[:, t] = float('-inf')
            worst = min(worst, (own - lg.max(-1).values).min().item())
        assert worst > 0, 'a phrase write does not win the argmax'
        return worst

    def _grab(self, module, args, out):
        if self.active: self._feat = (out[0] if isinstance(out, tuple) else out).detach()
        return None

    def detach(self):
        for h in self.hooks: h.remove()
        self.cache = None

    def _pre_model(self, module, args, kwargs):
        if not self.active: return None
        ids = kwargs.get('input_ids', args[0] if args else None)
        past = kwargs.get('past_key_values')
        n_past = 0 if past is None else past.get_seq_length()
        if n_past == 0:
            self.cache = None
            B = ids.shape[0]
            self.fired = torch.zeros(B, dtype=torch.bool, device=ids.device)
            self.firestep = torch.full((B,), -1, dtype=torch.long, device=ids.device)
            self._step = 0
        else:
            assert self.cache is not None and self.cache['T'] == n_past, 'cache out of sync'
        self._ids = ids
        return None

    @torch.no_grad()
    def _pre_norm(self, module, args):
        h = args[0]
        if not self.active: return None
        w = self.forward(h, self._ids)
        wrote = (w != 0).any(-1)                 # (B,n)
        if self.fired is not None:
            newf = wrote.any(1) & (~self.fired)
            self.firestep = torch.where(newf, torch.full_like(self.firestep, self._step), self.firestep)
            self.fired = self.fired | wrote.any(1)
            self._step += 1
        mask = wrote.unsqueeze(-1)
        h2 = torch.where(mask, (h.float() + w).to(h.dtype), h)
        return (h2,) + tuple(args[1:])

    def forward(self, h, ids):
        B, n, _ = h.shape
        dev = h.device
        c = self.cache; t0 = 0 if c is None else c['T']
        pos = (t0 + torch.arange(n, device=dev, dtype=torch.float32))[None].expand(B, n)
        fl = Fn.embedding(ids, self.tbl)                       # (B,n,2)
        isnl2 = fl[..., 0]; isend = fl[..., 1]
        tail_end = c['tail_end'] if c is not None else torch.zeros(B, 1, device=dev)
        prev_end = torch.cat([tail_end, isend], 1)[:, :-1]      # is_endthink of t-1
        key = AND(isnl2, prev_end)                             # (B,n) last prompt position
        z = (self._feat[:, :n].float() @ self.pw + self.pb)    # probe score at hidden[L]
        gh = thresh(z - self.logit_thr)                        # graw / hit
        cat = lambda k, v: v if c is None else torch.cat([c[k], v], 1)
        KEY = cat('key', key); KPOS = cat('kpos', key * pos)
        haskey = hard_attend(KEY, KEY, n, strict=False)
        keypos = hard_attend(KEY, KPOS, n, strict=False)
        frow = torch.zeros(B, n, self.Lseq, device=dev)
        trace = {}
        if self.mode == 'text_stop':
            gval = key * gh
            GVAL = cat('gval', gval)
            G = hard_attend(KEY, GVAL, n, strict=False)        # latched gate
            j = pos - keypos
            for s in range(self.Lseq):
                frow[..., s] = AND(G, haskey, eq0(j - s))
            new = dict(gval=GVAL)
            if self.trace: trace = dict(G=G, haskey=haskey, j=j)
        else:
            ehit = AND(gh, haskey)
            EH = cat('ehit', ehit)
            # width-k conv window: krun_t = AND(ehit_t, ehit_{t-1}, ..., ehit_{t-k+1})
            pad = c['ehit_tail'] if c is not None else torch.zeros(B, self.k - 1, device=dev)
            EHp = torch.cat([pad, ehit], 1)                    # (B, k-1+n)
            krun = AND(*[EHp[:, i:i + n] for i in range(self.k)])
            KRUN = cat('krun', krun)
            prevk = hard_attend(KRUN, KRUN, n, strict=True)
            fire = AND(krun, 1 - prevk)
            FIRE = cat('fire', fire); FPOS = cat('fpos', fire * pos)
            hasfire = hard_attend(FIRE, FIRE, n, strict=False)
            firepos = hard_attend(FIRE, FPOS, n, strict=False)
            j = pos - firepos
            for s in range(self.Lseq):
                frow[..., s] = AND(hasfire, eq0(j - s))
            new = dict(ehit=EH, krun=KRUN, fire=FIRE, fpos=FPOS, ehit_tail=EHp[:, -(self.k - 1):])
            if self.trace: trace = dict(fire=fire, hasfire=hasfire, j=j, krun=krun)
        w = frow @ self.WT
        self.cache = dict(T=t0 + n, key=KEY, kpos=KPOS, tail_end=isend[:, -1:], **new)
        if self.trace:
            trace['wrote'] = (w != 0).any(-1)
            self.last_trace = trace
        self._feat = None
        return w
