"""Call gate compiled into the model: a fixed logit write on the LM head plus a 1-bit latch register (latgate, [12] item 5).

The published gate (cg_gate_gen.py, mode 'gate') does, in the Python decode loop, per row:
    s = w.h + b ; blk |= (argmax(lg) == <tool_call>) & (s < 0) ; lg[blk, <tool_call>] = -1e9
Here the same computation is a module attached to the host's LM head (forward hook), built from fixed tensors only
(matmul, ReLU, argmax, as in the compiled IFEval enforcer), and the decode loop is the UNCHANGED baseline loop (mode
'none': no gate code in it). probe_v3 (w, b) is used unchanged.

  s      = h @ w + b                                   h = the LM head's input (post-norm last hidden = the probe's input)
  neg    = min(1, relu(-K s))                          ReLU step, K = 2^24: 1 iff s <= -2^-24 (margin logged, CHECK ONLY)
  top    = onehot(argmax(lg + m_eos))[TC]              argmax with the eos column pushed to -1e9 (fixed mask; see note)
  ev     = relu(top + neg - 1)                         AND
  latch  = 1 - relu(1 - latch - ev)                    OR into the latch register (B,), reset per batch like a KV cache
  lg_TC' = lg_TC - (relu(lg_TC - M(1-latch)) - relu(-lg_TC - M(1-latch))) - 1e9 latch       big-M ReLU pair, M = 1e6:
           = lg_TC when latch = 0, exactly -1e9 when latch = 1 (|lg| < M, checked)

Note on the eos mask: the Python gate takes argmax AFTER the loop's min_tokens rule (eos at -1e9 for the first 4 steps).
Excluding eos from the gate's argmax gives the same decision in every step: before min_tokens the loop does the same;
after it, when eos is the argmax the row emits eos and ends, so a latch set in that step is never used.

The latch is a register carried across decode steps (one float per row), the same status as the KV cache and as the
registers of the compiled IFEval implant; it is not a Python flag in the loop. reset(B) at the start of each batch.
"""
import torch, torch.nn.functional as Fn
import cg_gate_gen as CG

TC = CG.TC
K_STEP = float(2 ** 24); BIGM = 1.0e6; NEG = -1.0e9


class CompiledCallGate:
    def __init__(self, probe, eos, vocab, dev):
        self.w = probe['w'].to(dev).float(); self.b = torch.tensor(float(probe['b']), device=dev)
        self.m_eos = torch.zeros(vocab, device=dev); self.m_eos[eos] = NEG
        self.e_tc = torch.zeros(vocab, device=dev); self.e_tc[TC] = 1.0
        self.latch = None; self.handle = None
        self.check = None            # CHECK ONLY: dict of stashed raw logits / s, used by the shadow comparison
        self.min_abs_s = float('inf'); self.max_abs_lg = 0.0

    def reset(self, B, dev):
        self.latch = torch.zeros(B, device=dev)

    def attach(self, model):
        self.handle = model.lm_head.register_forward_hook(self._hook); return self

    def detach_(self):
        if self.handle is not None: self.handle.remove(); self.handle = None

    @torch.no_grad()
    def _hook(self, mod, args, out):
        h = args[0]
        assert h.dim() == 2, 'the gate is attached to the per-step LM head call (B, D)'
        lg = out.float()
        s = h.float() @ self.w + self.b
        neg = torch.clamp(Fn.relu(-K_STEP * s), max=1.0)
        top = Fn.one_hot((lg + self.m_eos).argmax(-1), lg.shape[-1]).float() @ self.e_tc
        ev = Fn.relu(top + neg - 1.0)
        self.latch = 1.0 - Fn.relu(1.0 - self.latch - ev)
        x = lg[:, TC]; off = BIGM * (1.0 - self.latch)
        lg_tc = x - (Fn.relu(x - off) - Fn.relu(-x - off)) + NEG * self.latch
        new = lg * (1.0 - self.e_tc) + lg_tc[:, None] * self.e_tc       # only column TC changes (exact)
        # CHECK ONLY (not used in the math): margins and the raw logits for the shadow comparison
        dec = (top > 0.5)
        if dec.any(): self.min_abs_s = min(self.min_abs_s, float(s[dec].abs().min()))
        self.max_abs_lg = max(self.max_abs_lg, float(x.abs().max()))
        if self.check is not None: self.check['raw'] = lg.clone(); self.check['s'] = s.clone()
        return new.to(out.dtype)


SHADOW = {'steps': 0, 'latch_mismatch': 0, 'token_mismatch': 0, 'rows': 0, 'rows_latched': 0}
GATE = {}


@torch.no_grad()
def generate(model, tok, net, prompts, max_new, dev, mode='none', probe=None, min_tokens=4, shadow=True):
    """modes none / gate / breaker: the original cg_gate_gen.generate. mode 'compiled': the original loop with mode
    'none' (no gate code in the loop) and the CompiledCallGate on the LM head; shadow=True also runs the Python gate's
    rule on the stashed raw logits each step (CHECK ONLY) and counts disagreements."""
    if mode != 'compiled':
        return CG.generate(model, tok, net, prompts, max_new, dev, mode=mode, probe=probe, min_tokens=min_tokens)
    eos = tok.convert_tokens_to_ids('<|endoftext|>')
    if 'g' not in GATE: GATE['g'] = CompiledCallGate(probe, eos, model.lm_head.weight.shape[0], dev)
    g = GATE['g']
    seqs = [tok(p, add_special_tokens=False)['input_ids'] for p in prompts]; L = max(map(len, seqs)); B = len(seqs)
    g.reset(B, dev); g.check = {} if shadow else None; g.attach(model)
    w = probe['w'].to(dev).float(); bvec = float(probe['b']); blk_py = torch.zeros(B, dtype=torch.bool, device=dev)
    try:
        # ---- below: cg_gate_gen.generate with mode 'none', line for line (+ the CHECK ONLY shadow block) ----
        x = torch.tensor([[eos] * (L - len(s)) + s for s in seqs], device=dev)
        att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=dev)
        gs = None
        if net is not None:
            import integ; net.prime(x); gs = integ.GraphStep(net, B)
        o = model.model(input_ids=x, attention_mask=att, use_cache=True); past = o.past_key_values
        h = o.last_hidden_state[:, -1]; lg = model.lm_head(h).float()
        gen = [[] for _ in range(B)]; done = [False] * B
        ncall = torch.zeros(B, dtype=torch.long, device=dev)
        for step in range(max_new):
            if step < min_tokens: lg[:, eos] = -1e9
            if shadow:                                    # CHECK ONLY: the published Python gate on the raw logits
                raw = g.check['raw']; raw = raw.clone()
                if step < min_tokens: raw[:, eos] = -1e9
                s = h.float() @ w + bvec
                dec_py = (raw.argmax(-1) == TC) & (s < 0)
                blk_py = blk_py | dec_py
                raw[blk_py, TC] = -1e9
                live = torch.tensor([not d for d in done], device=dev)
                SHADOW['steps'] += int(live.sum())
                SHADOW['latch_mismatch'] += int(((g.latch > 0.5) != blk_py)[live].sum())
                SHADOW['token_mismatch'] += int((raw.argmax(-1) != lg.argmax(-1))[live].sum())
            nxt = lg.argmax(-1)
            ncall += (nxt == TC).long()
            nl = nxt.tolist()
            for i in range(B):
                if done[i]: continue
                gen[i].append(nl[i])
                if nl[i] == eos or (len(gen[i]) % 6 == 0 and '\n\nQ' in tok.decode(gen[i][-12:])): done[i] = True
            if all(done): break
            nxt = torch.where(torch.tensor(done, device=dev), torch.full_like(nxt, eos), nxt)
            att = torch.cat([att, att.new_ones(B, 1)], 1)
            if gs is not None: gs.step(nxt)
            o = model.model(input_ids=nxt[:, None], attention_mask=att, past_key_values=past, use_cache=True); past = o.past_key_values
            h = o.last_hidden_state[:, -1]; lg = model.lm_head(h).float()
    finally:
        g.detach_(); g.check = None
    SHADOW['rows'] += B; SHADOW['rows_latched'] += int((g.latch > 0.5).sum())
    SHADOW['min_abs_s_at_tc_top'] = g.min_abs_s; SHADOW['max_abs_lg_tc'] = g.max_abs_lg
    stats = [{'ncall': int(ncall[i]), 'blocked': bool(g.latch[i] > 0.5)} for i in range(B)]
    return gen, stats
