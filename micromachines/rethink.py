"""The `rethink` implant: when the frozen host is about to end its answer, it writes "\\nHmm, wait. Let me rethink this.\\n"
token by token instead, then goes silent so the host writes a second solution. Optionally gated by a linear probe on
the host's last hidden state at its own '####' token.

It uses the same attach point as the other implants: a vector added to the host's last hidden state, right before the
final norm (a forward pre-hook on `model.model.norm`; it is registered after any other implant's hook, so it sees the
hidden state with their writes added). Where it does not write, its write is exactly zero (the hook passes the hidden
state through untouched), so the host's logits there are bit-identical to the unmodified model.

    reads   : the host's token ids (0/1 lookup tables over the vocabulary: "ends with a newline", "contains ####")
              the host's last hidden state h at the current position, two ways:
                * end detector: the host's own next-token choice, argmax(lm_head(norm(h))), computed with the host's frozen
                  norm and head. It fires when that choice would end the answer the way the evaluation script's stop
                  rule sees it: a token that contains a blank line ('\\n\\n', '.\\n\\n', ...), a token that starts with a
                  newline right after a token that ended with one, or <|endoftext|>. It also fires when the answer
                  reaches `budget` tokens (320 = the script's first-pass cap) so that it matches the script exactly.
                * gate (optional): a logistic probe on norm(h) (= Hugging Face `hidden_states[28]`, the input of the LM
                  head) at the first token containing '####' after "Answer:". Fires iff score >= thr.
    writes  : at the end position and the following ones, one solved direction per token of the forced text (the same
              `make_out_dirs` as the other implants: Adam through the host's own final norm + LM head). The forced text
              is the one the script builds: stem.rstrip() + "\\n" + PHRASE, where stem is the part of the end token before
              its blank line (for '\\n\\n' it is empty; for '.\\n\\n' it is '.'); if the previous token already ended with
              a newline and the stem is blank, just PHRASE.
    once    : it fires at most once per answer (state is reset by `prime`, i.e. at "Answer:" of a new prompt).

What is written and what is fitted: the trigger, the token tables and the writer's sequence are written (code below); the
write directions are solved against the frozen head; the probe weights (w, b, standardization, threshold) are FITTED:
scikit-learn logistic regression on the host's states at '####' for 1,500 GSM8K training problems, threshold chosen on
those training problems (scripts/gate/fit_detector.py -> scripts/gate/export_probe.py -> results/rethink/rethink_probe_L28.pt).
"""
import torch, torch.nn.functional as Fn
from .machine import make_out_dirs

PHRASE = "Hmm, wait. Let me rethink this.\n"


class RethinkImplant:
    def __init__(self, model, tok, mode='always', probe=None, budget=320, gain=1e4, phrase=PHRASE):
        assert mode in ('always', 'gated')
        self.model, self.tok, self.mode, self.budget, self.gain, self.phrase = model, tok, mode, budget, gain, phrase
        dev = model.device
        V = model.lm_head.weight.shape[0]
        T = lambda s: tok(s, add_special_tokens=False)['input_ids']
        self.eos = tok.convert_tokens_to_ids('<|endoftext|>')
        txt = [tok.decode([i]) for i in range(len(tok))] + [''] * max(0, V - len(tok))
        # ---- token lookup tables (0/1 over the vocabulary) ----
        self.ends_nl = torch.tensor([t.endswith('\n') for t in txt], device=dev)
        self.hashes = torch.tensor(['####' in t for t in txt], device=dev)
        # ---- end tokens and the text forced in their place ----
        seqs, row_any, row_nl = [], torch.full((V,), -1, dtype=torch.long), torch.full((V,), -1, dtype=torch.long)
        def add(text):
            s = T(text)
            if s not in seqs: seqs.append(s)
            return seqs.index(s)
        for i, t in enumerate(txt):
            if i == self.eos: stem = ''
            elif '\n\n' in t: stem = t.split('\n\n')[0]
            else: stem = None
            if stem is not None: row_any[i] = add(stem.rstrip() + '\n' + phrase)          # previous token did not end with '\n'
            if t.startswith('\n') or (stem is not None and stem.strip() == ''):           # previous token ended with '\n'
                row_nl[i] = add(phrase) if (stem is None or stem.strip() == '') else add(stem.rstrip() + '\n' + phrase)
            elif stem is not None:
                row_nl[i] = row_any[i]
        self.row_budget = (add('\n' + phrase), add(phrase))                                  # (prev not '\n', prev '\n')
        self.L = max(len(s) for s in seqs)
        out_ids = sorted({t for s in seqs for t in s})
        self.out_ids = out_ids
        with torch.enable_grad():                                                             # (n_out, d) rms 1; in chunks (GPU memory)
            D = torch.cat([make_out_dirs(model, out_ids[c:c + 64]) for c in range(0, len(out_ids), 64)])
        self.D = D.to(dev)
        col = {t: k for k, t in enumerate(out_ids)}
        F = torch.full((len(seqs), self.L), -1, dtype=torch.long)
        for r, s in enumerate(seqs): F[r, :len(s)] = torch.tensor([col[t] for t in s])
        self.F, self.Ftok = F.to(dev), torch.tensor([[s[k] if k < len(s) else -1 for k in range(self.L)] for s in seqs], device=dev)
        self.row_any, self.row_nl = row_any.to(dev), row_nl.to(dev)
        self.n_seqs = len(seqs)
        # ---- probe (fitted) ----
        if mode == 'gated':
            P = torch.load(probe) if isinstance(probe, str) else probe
            self.pw = (P['w'] / P['sd']).double().to(dev)                                      # score = sigmoid(pw . x + pb), float64
            self.pb = float(P['b'] - (P['w'] * P['mu'] / P['sd']).sum())
            self.thr = float(P['thr'])
            self.probe_layer = int(P.get('layer', 28))
            assert self.probe_layer == model.config.num_hidden_layers, 'the probe reads the final normed state'
        self.hook = model.model.norm.register_forward_pre_hook(self._hook)
        self.state = None; self.active = True                  # active=False: hook bypassed (used by the silent check)
        self.probed = None
        self.margins = self.check_head()

    # ---- state: per row ----
    def prime(self, input_ids):
        """New prompt (ending with 'Answer:'): arm the implant for every row."""
        B = input_ids.shape[0]; dev = input_ids.device
        z = lambda dt=torch.long: torch.zeros(B, dtype=dt, device=dev)
        self.wrote = torch.zeros(B, dtype=torch.bool, device=dev)
        self.state = dict(last=input_ids[:, -1].clone(), n=z(), phase=z(), row=z(), seen=z(torch.bool),
                          gate=torch.full((B,), self.mode == 'always', device=dev),
                          score=torch.full((B,), float('nan'), dtype=torch.float64, device=dev))

    def step(self, next_ids):
        """The host's newest token (the one just emitted / fed back)."""
        s = self.state
        s['last'] = next_ids.clone(); s['n'] = s['n'] + 1
        writing = (s['phase'] > 0) & (s['phase'] <= self.L)
        s['phase'] = torch.where(writing, s['phase'] + 1, s['phase'])                       # move to the next forced token

    # ---- the write, computed inside the host's forward (pre-hook on the final norm) ----
    @torch.no_grad()
    def _hook(self, module, args):
        if self.state is None or not self.active: return None
        h = args[0]                                             # (B, T, d); the decision is for the last position
        s = self.state; m = self.model
        x = module.forward(h[:, -1])                            # the host's own final norm (forward, no hooks) = hidden_states[28]
        prev_nl = self.ends_nl[s['last']]
        armed = s['phase'] == 0
        self.probed = torch.zeros_like(armed)
        if self.mode == 'gated':                                # probe at the first '####' token of the answer
            at = armed & ~s['seen'] & self.hashes[s['last']]
            if at.any():
                sc = torch.sigmoid(x.double() @ self.pw + self.pb)
                s['score'] = torch.where(at, sc, s['score'])
                s['gate'] = torch.where(at, sc >= self.thr, s['gate'])
                s['seen'] = s['seen'] | at; self.probed = at
        choice = m.lm_head(x).float().argmax(-1)                 # the host's own next token
        r_end = torch.where(prev_nl, self.row_nl[choice], self.row_any[choice])
        r_bud = torch.where(prev_nl, torch.full_like(r_end, self.row_budget[1]), torch.full_like(r_end, self.row_budget[0]))
        at_budget = (s['n'] >= self.budget) if self.budget else torch.zeros_like(armed)
        fire = armed & s['gate'] & ((r_end >= 0) | at_budget)
        s['row'] = torch.where(fire, torch.where(r_end >= 0, r_end, r_bud), s['row'])
        s['phase'] = torch.where(fire, torch.ones_like(s['phase']), s['phase'])
        k = (s['phase'] - 1).clamp(0, self.L - 1)
        col = self.F[s['row'], k]
        writing = (s['phase'] >= 1) & (s['phase'] <= self.L) & (col >= 0)
        s['phase'] = torch.where((s['phase'] >= 1) & (col < 0), torch.full_like(s['phase'], self.L + 1), s['phase'])
        self.wrote = writing
        if not writing.any(): return None                       # silent: h passes through untouched
        w = (self.D[col.clamp(min=0)] * self.gain) * writing[:, None].float()
        h2 = h.clone(); h2[:, -1] = (h[:, -1].float() + w).to(h.dtype)
        return (h2,) + tuple(args[1:])

    def forced_token(self):
        """Token each row's write forces at the current position (-1 where silent)."""
        s = self.state; k = (s['phase'] - 1).clamp(0, self.L - 1)
        return torch.where(self.wrote, self.Ftok[s['row'], k], torch.full_like(k, -1))

    def detach(self):
        self.hook.remove(); self.state = None

    @torch.no_grad()
    def check_head(self):
        """Each forced token wins the host's argmax by a margin (random states, host's own norm + head and dtype)."""
        m = self.model; dt = m.lm_head.weight.dtype
        h = torch.randn(64, m.config.hidden_size, device=m.device, dtype=dt) * 10
        worst = float('inf')
        for j, t in enumerate(self.out_ids):
            lg = m.lm_head(m.model.norm(h + (self.D[j] * self.gain).to(dt))).float()
            own = lg[:, t].clone(); lg[:, t] = float('-inf')
            worst = min(worst, (own - lg.max(-1).values).min().item())
        assert worst > 0, 'a rethink write does not win the argmax'
        return worst
