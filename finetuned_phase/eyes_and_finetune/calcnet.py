"""Weight-space micromachines: hard-coded side networks that read the host LM's token embeddings and write into
its final residual stream.

    token embeddings (host)  --reader-->  symbol flags per token
    conv window (K tokens)   --stack 1 (ReLU MLP)-->  a state vector          (meaningful at the machine's key token)
    1 attention head: copy the state from the most recent key token
    window + state           --stack 2 (ReLU MLP)-->  one-hot over the machine's output tokens (all zeros outside its span)
    one-hot @ D * gain       --> added to the host's last hidden state, before its final norm and LM head

No weight is trained and nothing in the forward pass branches on data. Several machines can be attached to one host;
their writes are added.
"""
import torch, torch.nn as nn, torch.nn.functional as Fn
import circuits, circuits2, circuits3, circuits4, circuits5, circuits6, circuits7, circuits8, circuits9, circuits10, circuits11
from hdl import ReLUNet
from calcfmt import CS, CE, OS, OE
from swapfmt import CS2, CE2, OS2, OE2
from liefmt import CS3, CE3, OS3, OE3
from boolfmt import CS4, CE4, OS4, OE4

BIG, GAMMA = 1e6, 300.0
import eyes as _eyes
SPECS = {
    'calc': dict(build=circuits.build, syms=circuits.SYMS, out=circuits.OUT, key='CE', K=circuits.K, W=circuits.K,
                 text={**{c: c for c in '0123456789+-*/.'}, 'CS': CS, 'CE': CE, 'OS': OS, 'OE': OE, 'N+': '+-', 'N-': '--', 'N*': '*-', 'N/': '/-'}),
    'swap': dict(build=circuits2.build, syms=circuits2.SYMS, out=circuits2.OUT, key='OE', K=circuits2.K, W=512,
                 text={**{c: c for c in '0123456789'}, 'CS': CS2, 'CE': CE2, 'OS': OS2, 'OE': OE2}),
    'bool': dict(build=circuits4.build, syms=circuits4.SYMS, out=circuits4.OUT, key='CE', K=circuits4.K, W=8,
                 text={'T0': 'True', 'T1': ' True', 'F0': 'False', 'F1': ' False', 'N0': 'not', 'N1': ' not', 'A0': 'and', 'A1': ' and', 'R0': 'or', 'R1': ' or',
                       'B0': '(', 'B1': ' (', 'C0': ')', 'C1': ' )', 'CS': CS4, 'CE': CE4, 'OS': OS4, 'OE': OE4}),
    'logic': dict(build=circuits5.build, syms=circuits5.SYMS, out=circuits5.OUT, key='CE', K=circuits5.K, W=circuits5.LMAX + 3,
                  text={**{c: c for c in '0123456789'}, '=': '=', '=-': '=-', '<': '<', '>': '>', ';': ';', ':': ':', '?': '?', '?-': '?-',
                        'CS': '<|logic|>', 'CE': '<|/logic|>', 'OS': '<|logic_out|>', 'OE': '<|/logic_out|>'}),
    'dyck': dict(build=circuits6.build, syms=circuits6.SYMS, out=circuits6.OUT, key='CE', K=circuits6.K, W=circuits6.LMAX + 3,
                 text={**{c + v: (' ' if v == '1' else '') + c for c in '([{<)]}>' for v in '01'},
                       'CS': '<|dyck|>', 'CE': '<|/dyck|>', 'OS': '<|dyck_out|>', 'OE': '<|/dyck_out|>'}),
    'walk': dict(build=circuits7.build, syms=circuits7.SYMS, out=circuits7.OUT, key='CE', K=circuits7.K, W=circuits7.LMAX + 3,
                 text={**{c: c for c in '0123456789'}, 'F': 'F', 'B': 'B', 'L': 'L', 'R': 'R', 'TL': 'TL', 'TR': 'TR', 'TA': 'TA', ';': ';',
                       'Yes': 'Yes', 'No': 'No', 'CS': '<|walk|>', 'CE': '<|/walk|>', 'OS': '<|walk_out|>', 'OE': '<|/walk_out|>'}),
    'time': dict(build=circuits8.build, syms=circuits8.SYMS, out=circuits8.OUT, key='CE', K=circuits8.K, W=circuits8.LMAX + 3,
                 text={**{c: c for c in '0123456789ABCD'}, '-': '-', ';': ';', '|': '|',
                       'CS': '<|time|>', 'CE': '<|/time|>', 'OS': '<|time_out|>', 'OE': '<|/time_out|>'}),
    'sort': dict(build=circuits9.build, syms=circuits9.FEAT, out=circuits9.OUT, key='CE', K=circuits9.K, W=circuits9.LMAX + 3, table=True,
                 text={**{s: s for s in circuits9.CONTENT}, 'CS': '<|sort|>', 'CE': '<|/sort|>', 'OS': '<|sort_out|>', 'OE': '<|/sort_out|>'}),
    'date': dict(build=circuits10.build, syms=circuits10.SYMS, out=circuits10.OUT, key='CE', K=circuits10.K, W=circuits10.LMAX + 3,
                 text={**{c: c for c in '0123456789'}, '/': '/', '+': '+', '-': '-', 'D': 'D', 'W': 'W', 'M': 'M', 'Y': 'Y',
                       'CS': '<|date|>', 'CE': '<|/date|>', 'OS': '<|date_out|>', 'OE': '<|/date_out|>'}),
    'shape': dict(build=circuits11.build, syms=circuits11.SYMS, out=circuits11.OUT, key='CE', K=circuits11.K, W=circuits11.LMAX + 3,
                  text={**{c: c for c in '0123456789'}, '.': '.', ',': ',', 'S_': ' ', 'M': 'M', 'Mx': ' M', 'L': ' L', 'A': ' A',
                        **{c: c for c in 'KTSE'}, 'CS': '<|shape|>', 'CE': '<|/shape|>', 'OS': '<|shape_out|>', 'OE': '<|/shape_out|>'}),
    'eye': dict(syms=None, out=None, text=None, eye=True),                       # MM12, built in eyes.py (CNN + slot memory)
    'truth': dict(build=circuits3.build, syms=circuits3.SYMS, out=circuits3.OUT, key='OE', K=circuits3.K, W=512,
                  text={**{c: c for c in '0123'}, 'CS': CS3, 'CE': CE3, 'OS': OS3, 'OE': OE3}),
}
SPECS['eye'].update(syms=_eyes.SYMS, out=_eyes.OUT, text=_eyes.TEXT)
# kept for older scripts
SYMS, SI, NS, K, OUT, N_STATE = circuits.SYMS, circuits.SI, circuits.NS, circuits.K, circuits.OUT, circuits.N_STATE


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
        self.register_buffer('table', z)                                                # (V, n_feat) for table-reading machines
        self.uses_ids = bool(sp.get('table'))
        self.gain = gain

    # ---- pieces ----
    def symbols(self, emb, ids=None):                    # (..., d) -> (..., n_sym) exact 0/1  (or table features by token id)
        if self.uses_ids: return self.table[ids]
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
        tid = None if ids is None else (ids if full else ids[:, -(self.K + self.W):])
        e = self.prime_sym(self.symbols(tail, tid))
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
        e = self.step_sym(self.symbols(emb, ids))
        return (e @ self.out_dirs) * self.gain, e[:, 0]


CalcNet = Micromachine


# ---------- linking a machine to a host model ----------
def symbol_token_ids(tok, name='calc'):
    ids = {}
    for s, text in SPECS[name]['text'].items():
        if text.startswith('<|') and text not in tok.get_vocab():
            tok.add_special_tokens({'additional_special_tokens': [text]}, replace_additional_special_tokens=False)
        t = tok(text, add_special_tokens=False)['input_ids']
        assert len(t) == 1, (s, t)
        ids[s] = t[0]
    return ids


@torch.no_grad()
def make_reader(E, ids, syms):
    """Linear read-out of the symbol flags from the host embedding matrix; checked on every vocabulary row."""
    E = E.float()
    W, b, gaps = [], [], {}
    for s in syms:
        t = ids[s]
        w = E[t] / E[t].norm()
        sc = E @ w
        own = sc[t].item(); sc[t] = float('-inf'); other = sc.max().item()
        gap = own - other
        if gap <= 0:                                     # fall back to a ridge-whitened direction
            A = E.T @ E + 1e-3 * torch.eye(E.shape[1], device=E.device) * E.shape[0]
            w = torch.linalg.solve(A, E[t]); sc = E @ w
            own = sc[t].item(); sc[t] = float('-inf'); other = sc.max().item(); gap = own - other
        assert gap > 0, f'symbol {s} is not linearly separable from the rest of the vocabulary'
        a = 8.0 / gap
        W.append(w * a); b.append(-a * (other + gap / 4)); gaps[s] = gap / max(abs(own), 1e-9)
    W, b = torch.stack(W), torch.tensor(b, device=E.device)
    sym = torch.clamp(E @ W.T + b, 0, 1)                 # exhaustive check over the vocabulary
    want = torch.zeros_like(sym)
    for i, s in enumerate(syms): want[ids[s], i] = 1
    assert torch.equal(sym, want), 'reader is not exact on the vocabulary'
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

    def __init__(self, model, tok, gain=1e4, machines=('calc',), link=None):
        """link: another checkpoint to compute the readers and write directions from (to test a stale link)."""
        self.model, self.tok = model, tok
        src = link if link is not None else model
        E = src.get_input_embeddings().weight.detach()
        all_ids = {n: symbol_token_ids(tok, n) for n in machines}
        uniq = sorted({all_ids[n][t] for n in machines for t in SPECS[n]['out']})
        with torch.enable_grad():
            D = dict(zip(uniq, make_out_dirs(src, uniq)))
        self.nets, self.gaps = {}, {}
        for n in machines:
            if SPECS[n].get('eye'):                      # MM12: exact reader + trained CNN + slot memory
                W, b, gaps = make_reader(E, all_ids[n], SPECS[n]['syms']); self.gaps.update({f'{n}:{k}': v for k, v in gaps.items()})
                cnn = _eyes.CNN(); cnn.load_state_dict(torch.load('runs/eye_cnn.pt', map_location='cpu'))
                self.nets[n] = _eyes.EyeMachine(W, b, torch.stack([D[all_ids[n][t]] for t in SPECS[n]['out']]), cnn.eval(), gain).to(E.device)
                continue
            if SPECS[n].get('table'):                    # table-reading machine: features by token id, checked by its own tests
                import circuits9
                net = Micromachine(n, None, None, torch.stack([D[all_ids[n][t]] for t in SPECS[n]['out']]), gain).to(E.device)
                net.table = circuits9.token_table(tok, E.shape[0], {k: all_ids[n][k] for k in ('CS', 'CE', 'OS', 'OE')}).to(E.device)
                self.nets[n] = net; continue
            W, b, gaps = make_reader(E, all_ids[n], SPECS[n]['syms'])
            if link is not None:                         # how does the old reader do on the new embeddings?
                Enew = model.get_input_embeddings().weight.detach().float()
                sym = torch.clamp(Enew @ W.T + b, 0, 1); want = torch.zeros_like(sym)
                for i, s in enumerate(SPECS[n]['syms']): want[all_ids[n][s], i] = 1
                self.stale_reader_exact = getattr(self, 'stale_reader_exact', True) and bool(torch.equal(sym, want))
            self.gaps.update({f'{n}:{k}': v for k, v in gaps.items()})
            self.nets[n] = Micromachine(n, W, b, torch.stack([D[all_ids[n][t]] for t in SPECS[n]['out']]), gain).to(E.device)
        self.net = self.nets[machines[0]]; self.ids = all_ids[machines[0]]; self.all_ids = all_ids
        self.pending = None
        self.hook = model.model.norm.register_forward_pre_hook(self._add)
        self.margins = self.check_head(strict=link is None)

    def _add(self, module, args):
        if self.pending is None: return None
        h = args[0]
        return (h + self.pending.to(h.dtype),)

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
                tid = self.all_ids[n][t]
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


def attach_calcnet(model, tok, gain=1e4, machines=('calc',), link=None):
    return Attached(model, tok, gain, machines, link)
