"""Weight-space micromachines: hand-written ReLU side networks that read the host LM's token embeddings and write
into its final residual stream.

    token embeddings (host)  --reader-->  symbol flags per token (exact 0/1, checked on every vocabulary row)
    conv window (K tokens)   --stack 1 (ReLU MLP)-->  a state vector          (meaningful at the machine's key token)
    1 hard attention head: copy the state from the most recent key token (within W tokens)
    window + state           --stack 2 (ReLU MLP)-->  one-hot over the machine's output tokens (all zeros outside its span)
    one-hot @ D * gain       --> added to the host's last hidden state, before its final norm and LM head

The ReLU stacks are written, not trained (circuits/*.py). Two small pieces are *solved* against the frozen host:
the reader (a linear map from the embedding matrix to the symbol flags; closed form, with a max-margin Adam fallback for
symbols that stand for several tokens) and the write directions D (Adam on the host's own norm + LM head, so that each
direction makes the host emit that token). Nothing in the forward pass branches on data, and nothing of the host changes.
Several machines can be attached to one host; their writes are added. When no machine fires, every write is exactly
zero and the host's logits are bit-identical to the unmodified model.
"""
import torch, torch.nn as nn, torch.nn.functional as Fn
from .hdl import ReLUNet
from .specs import SPECS

BIG, GAMMA = 1e6, 300.0


class Micromachine(nn.Module):
    def __init__(self, name, reader_w=None, reader_b=None, out_dirs=None, gain=1e4):
        super().__init__()
        sp = SPECS[name]; self.name = name
        c1, c2 = sp['build']()
        self.stack1, self.stack2 = ReLUNet(c1), ReLUNet(c2)
        self.K, self.W, self.NS, self.key = sp['K'], sp['W'], len(sp['syms']), sp['syms'].index(sp['key'])
        z = torch.zeros(0)
        self.register_buffer('reader_w', reader_w if reader_w is not None else z)      # (n_sym, d_model)
        self.register_buffer('reader_b', reader_b if reader_b is not None else z)      # (n_sym,)
        self.register_buffer('out_dirs', out_dirs if out_dirs is not None else z)      # (n_out, d_model)
        self.gain = gain

    # ---- pieces ----
    def symbols(self, emb, ids=None):                    # (..., d) -> (..., n_sym) exact 0/1
        return torch.clamp(Fn.linear(emb.float(), self.reader_w, self.reader_b), 0, 1)

    def windows(self, sym):                              # (B,T,n_sym) -> (B,T,K*n_sym), offset 0 = current token
        B, T, _ = sym.shape
        pad = Fn.pad(sym, (0, 0, self.K - 1, 0))
        return pad.unfold(1, self.K, 1).permute(0, 1, 3, 2).flip(2).reshape(B, T, self.K * self.NS)

    def attend(self, S, is_key):                         # copy S from the most recent key token within the window
        T = S.shape[1]
        i = torch.arange(T, device=S.device)
        dist = (i[:, None] - i[None, :]).float()
        score = BIG * is_key[:, None, :] - GAMMA * dist
        score = score.masked_fill((dist < 0) | (dist >= self.W), float('-inf'))
        return torch.softmax(score, -1) @ S

    # ---- full-sequence forward (teacher forcing / prompt) ----
    def emit_sym(self, sym):
        win = self.windows(sym)
        S = self.stack1(win)
        C = self.attend(S, sym[..., self.key])
        return self.stack2(torch.cat([win, C], -1)), S

    def emit(self, emb):
        sym = self.symbols(emb)
        e, S = self.emit_sym(sym)
        return e, (sym, S)

    def forward(self, emb):                              # -> residual write (B,T,d)
        e, _ = self.emit(emb)
        return (e @ self.out_dirs) * self.gain

    # ---- incremental (generation) ----
    def prime_sym(self, sym):
        e, S = self.emit_sym(sym)
        def last(x, n):
            x = x[:, -n:]
            return Fn.pad(x, (0, 0, n - x.shape[1], 0)) if x.shape[1] < n else x
        self.state = (last(sym, self.K), last(S, self.W), last(sym[..., self.key:self.key + 1], self.W)[..., 0])
        return e

    def prime(self, emb, full=False, ids=None):
        """Start incremental decoding after a prompt. The receptive field is K+W-1 tokens, so the tail of the prompt
        is enough for the last position (full=True computes every prompt position exactly)."""
        B, T, _ = emb.shape
        tail = emb if full else emb[:, -(self.K + self.W):]
        e = self.prime_sym(self.symbols(tail))
        write = emb.new_zeros(B, T, self.out_dirs.shape[1], dtype=torch.float32)
        write[:, T - tail.shape[1]:] = (e @ self.out_dirs) * self.gain
        return write

    def step_sym(self, s):                               # s (B, n_sym): flags of the newest token
        sym_h, S_h, key_h = self.state
        sym_h = torch.cat([sym_h[:, 1:], s[:, None]], 1)
        win = sym_h.flip(1).reshape(sym_h.shape[0], 1, self.K * self.NS)
        S_h = torch.cat([S_h[:, 1:], self.stack1(win)], 1)
        key_h = torch.cat([key_h[:, 1:], s[:, self.key:self.key + 1]], 1)
        self.state = (sym_h, S_h, key_h)
        dist = torch.arange(self.W - 1, -1, -1, device=s.device).float()
        C = torch.softmax(BIG * key_h - GAMMA * dist, -1)[:, None, :] @ S_h
        return self.stack2(torch.cat([win, C], -1))

    def step(self, emb, ids=None):                       # emb (B,d) of the newest token
        e = self.step_sym(self.symbols(emb))
        return (e @ self.out_dirs) * self.gain, e[:, 0]


# ---------- linking a machine to a host model ----------
first = lambda v: v[0] if isinstance(v, list) else v        # a symbol with several input tokens writes the first one


def symbol_token_ids(tok, name):
    ids = {}
    for s, text in SPECS[name]['text'].items():
        alts = list(text) if isinstance(text, (list, tuple)) else [text]          # several tokens may stand for one symbol
        got = []
        for a in alts:
            t = tok(a, add_special_tokens=False)['input_ids']
            assert len(t) == 1, (s, a, t)
            got.append(t[0])
        ids[s] = got[0] if len(got) == 1 else got
    return ids


@torch.no_grad()
def make_reader(E, ids, syms):
    """Linear read-out of the symbol flags from the host embedding matrix; checked on every vocabulary row."""
    E = E.float()
    W, b, gaps = [], [], {}
    for s in syms:
        t = ids[s] if isinstance(ids[s], list) else [ids[s]]
        w = (E[t] / E[t].norm(dim=-1, keepdim=True)).mean(0); w = w / w.norm()
        sc = E @ w
        own = sc[t].min().item(); sc[t] = float('-inf'); other = sc.max().item()
        gap = own - other
        if gap <= 0:                                     # fall back to a ridge-whitened direction
            A = E.T @ E + 1e-3 * torch.eye(E.shape[1], device=E.device) * E.shape[0]
            w = torch.linalg.solve(A, E[t].mean(0)); sc = E @ w
            own = sc[t].min().item(); sc[t] = float('-inf'); other = sc.max().item(); gap = own - other
        if gap <= 0 and len(t) > 1:                      # a set of tokens: solve a max-margin linear separator over the whole vocabulary
            with torch.enable_grad():
                pos = torch.zeros(E.shape[0], dtype=torch.bool, device=E.device); pos[t] = True
                Ew = E / E.norm(dim=-1, keepdim=True); wv = Ew[t].mean(0).clone().requires_grad_(True); bb = torch.zeros((), requires_grad=True)
                opt = torch.optim.Adam([wv, bb], lr=0.01)
                for _ in range(3000):
                    sc_ = Ew @ wv + bb
                    loss = torch.relu(1 - sc_[pos]).mean() * 50 + torch.relu(1 + sc_[~pos]).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
            w = (wv.detach() / E.norm(dim=-1).mean()); sc = E @ w
            own = sc[t].min().item(); sc[t] = float('-inf'); other = sc.max().item(); gap = own - other
        assert gap > 0, f'symbol {s} is not linearly separable from the rest of the vocabulary'
        a = 8.0 / gap
        W.append(w * a); b.append(-a * (other + gap / 4)); gaps[s] = gap / max(abs(own), 1e-9)
    W, b = torch.stack(W), torch.tensor(b, device=E.device)
    sym = torch.clamp(E @ W.T + b, 0, 1)                 # exhaustive check over the vocabulary
    want = torch.zeros_like(sym)
    for i, s in enumerate(syms): want[ids[s], i] = 1
    assert torch.equal(sym, want), 'reader is not exact on the vocabulary'      # (a list of ids marks every alternative)
    return W, b, gaps


def make_out_dirs(model, out_ids, steps=3000):
    """For each output token: a residual direction that makes the host's own (norm + LM head) say that token."""
    norm, head = model.model.norm, model.lm_head
    g = norm.weight.detach().float(); E = head.weight.detach().float()
    tgt = torch.tensor(out_ids, device=E.device)
    d = (E[tgt] / g.clamp(min=1e-3)).clone().requires_grad_(True)
    opt = torch.optim.Adam([d], lr=0.3)
    for _ in range(steps):
        x = d * torch.rsqrt(d.pow(2).mean(-1, keepdim=True) + 1e-6) * g
        loss = Fn.cross_entropy(x @ E.T, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    d = d.detach()
    return d * torch.rsqrt(d.pow(2).mean(-1, keepdim=True))        # rms 1


class Attached:
    """Host LM + one or more micromachines. Their writes are added to the residual stream right before the host's final norm."""

    def __init__(self, model, tok, gain=1e4, machines=('calc_s4',)):
        self.model, self.tok = model, tok
        E = model.get_input_embeddings().weight.detach()
        all_ids = {n: symbol_token_ids(tok, n) for n in machines}
        uniq = sorted({first(all_ids[n][t]) for n in machines for t in SPECS[n]['out']})
        with torch.enable_grad():
            D = dict(zip(uniq, make_out_dirs(model, uniq)))
        self.nets, self.gaps = {}, {}
        for n in machines:
            W, b, gaps = make_reader(E, all_ids[n], SPECS[n]['syms'])
            self.gaps.update({f'{n}:{k}': v for k, v in gaps.items()})
            self.nets[n] = Micromachine(n, W, b, torch.stack([D[first(all_ids[n][t])] for t in SPECS[n]['out']]), gain).to(E.device)
        self.net = self.nets[machines[0]]; self.ids = all_ids[machines[0]]; self.all_ids = all_ids
        self.pending = None
        self.hook = model.model.norm.register_forward_pre_hook(self._add)
        self.margins = self.check_head()

    def _add(self, module, args):
        if self.pending is None: return None
        h = args[0]
        return (h + self.pending.to(h.dtype),)

    def detach(self):
        """Remove the hook: the host is exactly the unmodified model again."""
        self.hook.remove(); self.pending = None

    @torch.no_grad()
    def check_head(self, strict=True):
        """Through the host's real norm + head, in its own dtype: does each write win, and by how much?"""
        m = self.model; dt = m.lm_head.weight.dtype
        h = torch.randn(64, 1, m.config.hidden_size, device=m.device, dtype=dt) * 10
        out = {}
        for n, net in self.nets.items():
            for i, t in enumerate(SPECS[n]['out']):
                x = h + (net.out_dirs[i] * net.gain).to(dt)
                lg = m.lm_head(m.model.norm(x)).float()[:, 0]
                tid = first(self.all_ids[n][t])
                own = lg[:, tid].clone(); lp = torch.log_softmax(lg, -1)[:, tid]
                lg[:, tid] = float('-inf'); margin = (own - lg.max(-1).values).min().item()
                assert margin > 0 or not strict, f'write for {t} does not win the argmax'
                out[f'{n}:{t}'] = (margin, (1 - lp.exp()).max().item())
        return out

    @torch.no_grad()
    def prime(self, input_ids, full=False):
        emb = self.model.get_input_embeddings()(input_ids)
        self.pending = sum(net.prime(emb, full, input_ids) for net in self.nets.values())

    @torch.no_grad()
    def step(self, next_ids):
        emb = self.model.get_input_embeddings()(next_ids)
        res = [net.step(emb, next_ids) for net in self.nets.values()]
        self.pending = sum(w for w, _ in res)
        return res[0][1]

    def set_veto(self, name, v, zero_os=True):
        """End-of-number 'veto' write for a calculator-style machine: its OE output becomes the solved direction v
        (veto.py: no digit may win, the model's own non-digit choice is kept). zero_os: the model writes the space
        after '=' itself (OS output removed)."""
        M = self.nets[name]; outs = SPECS[name]['out']
        M.out_dirs[outs.index('OE')] = v.to(M.out_dirs) / M.gain
        if zero_os: M.out_dirs[outs.index('OS')] = 0


def attach(model, tok, machines=('calc_s4',), gain=1e4, veto=None, veto_machines=('calc_s4',), zero_os=True):
    """Attach micromachines to a (frozen) HF causal LM. veto: a tensor (d_model,) or a path to a file saved by
    scripts/solve_veto.py; applied to the machines in veto_machines that are attached."""
    A = Attached(model, tok, gain, tuple(machines))
    if veto is not None:
        v = torch.load(veto)['v'] if isinstance(veto, str) else veto
        for m in veto_machines:
            if m in A.nets: A.set_veto(m, v, zero_os=zero_os)
    return A
