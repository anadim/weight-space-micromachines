"""Compiled latent-write implant (all inference ops are fixed tensors).

Inference path, per forward call of the frozen host:
  1. reader      : symbol flags = clamp(E_tok @ Wr^T + br, 0, 1) on the host's token embeddings (fixed linear map,
                   checked exact 0/1 on every vocabulary row; same construction as micromachines.machine.make_reader).
  2. window      : flags at offsets -KP..+KF around every position (fixed unfold; a tensor cache keeps the last KP rows
                   across decode steps; offsets beyond the current chunk read as 0).
  3. circuit     : compiled ReLU net (hdl.Net, written, not trained). At position t it outputs, for every layout class
                   c = (la, lb, op) and slot i, the role flag g[c,i] (1 iff t is slot i of a span ` (` a `*` b `)+|)-`
                   with la/lb digits and len(a*b) == n-1, n = la+1+lb), and AND(g[c,i], onehot of digit p of P=a*b).
                   P is computed exactly inside the net (gated digit products, digit extraction by threshold units).
  4. write       : entering block L:  h' = (h - gate(h, G)) + F @ Wbank_L,  G = sum_c,i g[c,i] in {0,1},
                   gate(h, G) = relu(h - M(1-G)) - relu(-h - M(1-G))   (big-M ReLU pair, exact for |h| < M)
                   Wbank_L rows = [mu_ci ; A_ci] per slot = the least-squares writer (the only fitted tensors).
                   => slot positions get mu + A f(P) (replacement), every other position gets h back unchanged.
The host model's weights are untouched; the implant is attached with a forward hook on the embedding and a
forward pre-hook on block L (the attachment points only; no data-dependent Python in either).
"""
import torch, torch.nn as nn, torch.nn.functional as Fn
from micromachines.hdl import Net, Lin, lsum, compile_net, ReLUNet   # the repo's circuit compiler (this experiment used an identical copy)

BIGM = 1.0e6       # big-M of the gated replacement (|h| < BIGM checked at run time)


class Spec:
    """symbols: name -> list of token strings (the reader flags); patterns: list of dicts
    {elems: [('S', name) | ('A', power) | ('B', power)], slots: [pattern indices written], nP: digits of P}.
    A pattern fires at slot k when every element j sits at offset j-k and len(str(a*b)) == nP."""

    def __init__(self, symbols, patterns):
        self.TEXT = {str(d): [str(d)] for d in range(10)}; self.TEXT.update(symbols)
        self.SYMS = list(self.TEXT); self.SI = {s_: i for i, s_ in enumerate(self.SYMS)}; self.NS = len(self.SYMS)
        self.patterns = patterns
        self.K = max(len(p['elems']) for p in patterns)
        self.KP = self.KF = self.K
        self.W = 2 * self.K + 1


def arith_spec(classes):
    """(la, lb, op) classes for ' (' a '*' b ')+|)-' ; slots = a..b (n = la+1+lb), nP = n-1."""
    pats = []
    for la, lb, op in classes:
        el = [('S', 'OP')] + [('A', la - 1 - j) for j in range(la)] + [('S', 'TI')] + [('B', lb - 1 - j) for j in range(lb)] + \
             [('S', 'CP' if op == '+' else 'CM')]
        n = la + 1 + lb
        pats.append(dict(elems=el, slots=list(range(1, n + 1)), nP=n - 1, name=(la, lb, op)))
    return Spec({'OP': [' (', '('], 'TI': ['*'], 'CP': [')+'], 'CM': [')-']}, pats)


def blk(p): return 1 + 10 * p['nP']
def nfeat_p(p): return len(p['slots']) * blk(p)


def build_circuit(spec):
    """Outputs F = per pattern, per slot: [g, g*onehot(P digit p, MSB first)], then debug wires [P, sum g]."""
    net = Net(spec.W * spec.NS); A = net.AND; KP, KF = spec.KP, spec.KF
    s = lambda o, ch: net.inp((o + KP) * spec.NS + spec.SI[ch]) if -KP <= o <= KF else Lin()
    Dg = lambda o: lsum(s(o, str(d)) for d in range(10))
    role = {}
    for pi, p in enumerate(spec.patterns):
        for k in p['slots']:
            role[pi, k] = A(*[(s(j - k, e[1]) if e[0] == 'S' else Dg(j - k)) for j, e in enumerate(p['elems'])])
    agl = []
    bf = {(q, v): Lin() for q in range(2) for v in range(1, 10)}
    for (pi, k), r in role.items():
        el = spec.patterns[pi]['elems']
        a_lin = lsum(10 ** e[1] * d * s(j - k, str(d)) for j, e in enumerate(el) if e[0] == 'A' for d in range(1, 10))
        agl.append(net.relu(a_lin - 100 * (1 - r)))                     # a <= 99 (assert below)
        for j, e in enumerate(el):
            if e[0] == 'B':
                for v in range(1, 10): bf[e[1], v] = bf[e[1], v] + A(r, s(j - k, str(v)))
        assert max(e[1] for e in el if e[0] == 'A') <= 1 and max(e[1] for e in el if e[0] == 'B') <= 1
    a = lsum(agl)
    P = lsum(10 ** q * v * net.relu(a - 100 * (1 - bf[q, v])) for q in range(2) for v in range(1, 10))
    Rd, rem = {}, P
    for k in (3, 2, 1):
        d = lsum(net.ge(rem, j * 10 ** k) for j in range(1, 10)); Rd[k] = d; rem = rem - 10 ** k * d
    Rd[0] = rem
    outs, sumg = [], Lin()
    for pi, p in enumerate(spec.patterns):
        nP = p['nP']
        lo, hi = net.ge(P, 10 ** (nP - 1)), net.ge(P, 10 ** nP)
        for k in p['slots']:
            g = A(role[pi, k], lo, 1 - hi)
            sumg = sumg + g
            outs.append(g)
            for q in range(nP):
                for d in range(10):
                    outs.append(A(g, net.eq(Rd[nP - 1 - q], d)))
    return compile_net(net, outs + [P, sumg])


def match_spans(tok_strs, spec):
    """REFERENCE / CHECK ONLY (Python): all (pattern index, start index, P) matches, sorted by start."""
    out = []
    for pi, p in enumerate(spec.patterns):
        el = p['elems']
        for st in range(len(tok_strs) - len(el) + 1):
            a = b = 0; ok = True
            for j, e in enumerate(el):
                t = tok_strs[st + j]
                if e[0] == 'S':
                    if t not in spec.TEXT[e[1]]: ok = False; break
                else:
                    if not (len(t) == 1 and t.isdigit()): ok = False; break
                    if e[0] == 'A': a += int(t) * 10 ** e[1]
                    else: b += int(t) * 10 ** e[1]
            if ok and len(str(a * b)) == p['nP']: out.append((pi, st, a * b))
    return sorted(out, key=lambda x: x[1])


def make_reader(E, ids, syms):
    """Copied from micromachines.machine.make_reader (closed-form part): linear read-out of symbol flags from the
    host embedding matrix, checked exact on every vocabulary row."""
    E = E.float()
    Wr, b = [], []
    for s_ in syms:
        t = ids[s_] if isinstance(ids[s_], list) else [ids[s_]]
        w = (E[t] / E[t].norm(dim=-1, keepdim=True)).mean(0); w = w / w.norm()
        sc = E @ w
        own = sc[t].min().item(); sc[t] = float('-inf'); other = sc.max().item(); gap = own - other
        if gap <= 0:
            A_ = E.T @ E + 1e-3 * torch.eye(E.shape[1], device=E.device) * E.shape[0]
            w = torch.linalg.solve(A_, E[t].mean(0)); sc = E @ w
            own = sc[t].min().item(); sc[t] = float('-inf'); other = sc.max().item(); gap = own - other
        if gap <= 0 and len(t) > 1:
            with torch.enable_grad():
                pos = torch.zeros(E.shape[0], dtype=torch.bool, device=E.device); pos[t] = True
                Ew = E / E.norm(dim=-1, keepdim=True); wv = Ew[t].mean(0).clone().requires_grad_(True)
                bb = torch.zeros((), device=E.device, requires_grad=True)
                opt = torch.optim.Adam([wv, bb], lr=0.01)
                for _ in range(3000):
                    sc_ = Ew @ wv + bb
                    loss = torch.relu(1 - sc_[pos]).mean() * 50 + torch.relu(1 + sc_[~pos]).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
            w = (wv.detach() / E.norm(dim=-1).mean()); sc = E @ w
            own = sc[t].min().item(); sc[t] = float('-inf'); other = sc.max().item(); gap = own - other
        assert gap > 0, f'symbol {s_} not linearly separable'
        a_ = 8.0 / gap
        Wr.append(w * a_); b.append(-a_ * (other + gap / 4))
    Wr, b = torch.stack(Wr), torch.tensor(b, device=E.device)
    sym = torch.clamp(E @ Wr.T + b, 0, 1)
    want = torch.zeros_like(sym)
    for i, s_ in enumerate(syms): want[ids[s_], i] = 1
    assert torch.equal(sym, want), 'reader not exact on the vocabulary'
    return Wr, b


def reader_ids(tok, spec):
    ids = {}
    for s_ in spec.SYMS:
        got = []
        for a in spec.TEXT[s_]:
            t = tok(a, add_special_tokens=False).input_ids
            if len(t) == 1: got.append(t[0])
        assert got, s_
        ids[s_] = got if len(got) > 1 else got[0]
    return ids


class Implant(nn.Module):
    """Attach with .attach(model, L). Wbank: dict layer -> [NF, D] float32 (rows ordered like the circuit's F)."""

    def __init__(self, model, tok, spec, Wbank):
        super().__init__()
        dev = model.device
        self.spec = spec
        self.NF = sum(nfeat_p(p) for p in spec.patterns)
        Wr, br = make_reader(model.get_input_embeddings().weight.detach(), reader_ids(tok, spec), spec.SYMS)
        self.register_buffer('Wr', Wr.float()); self.register_buffer('br', br.float())
        self.net = ReLUNet(build_circuit(spec)).to(dev)
        self.Wbank = {l: w.to(dev).float() for l, w in Wbank.items()}
        self.L = None; self.handles = []
        self.override = None        # CONTROL ONLY (corrupted / mean arms): fn(F, G, dbg) -> F
        self.capture = None         # CHECK ONLY: list to receive (h_in, h_out) at block L
        self.reset()

    # ---- tensor cache (reset by the host at the start of a new batch of sequences, like a KV cache) ----
    def reset(self):
        self.hist = None; self.pending = None; self.maxabs = 0.0; self.last = None

    def attach(self, model, L):
        self.detach_(); self.L = L
        self.handles = [model.get_input_embeddings().register_forward_hook(self._emb),
                        model.model.layers[L].register_forward_pre_hook(self._write, with_kwargs=True)]
        return self

    def detach_(self):
        for h in self.handles: h.remove()
        self.handles = []

    @torch.no_grad()
    def features(self, emb):
        B, T, _ = emb.shape
        KP, KF, W, NS = self.spec.KP, self.spec.KF, self.spec.W, self.spec.NS
        sym = torch.clamp(Fn.linear(emb.float(), self.Wr, self.br), 0, 1)
        if self.hist is None: self.hist = sym.new_zeros(B, KP, NS)
        allx = torch.cat([self.hist, sym, sym.new_zeros(B, KF, NS)], 1)
        win = allx.unfold(1, W, 1).permute(0, 1, 3, 2).reshape(B, T, W * NS)   # row t = offsets -KP..+KF
        self.hist = torch.cat([self.hist, sym], 1)[:, -KP:]
        o = self.net(win)
        F, dbg = o[..., :self.NF], o[..., self.NF:]
        return F, dbg[..., 1:2], dbg

    @torch.no_grad()
    def _emb(self, mod, args, out):
        F, G, dbg = self.features(out)
        if self.override is not None: F = self.override(F, G, dbg)
        self.last = (F, G, dbg)
        self.pending = (F, G)

    @torch.no_grad()
    def _write(self, mod, args, kwargs):
        h = args[0] if args else kwargs['hidden_states']
        F, G = self.pending
        hf = h.float()
        self.maxabs = max(self.maxabs, hf.abs().max().item())    # CHECK ONLY (big-M margin), not used in the math
        off = BIGM * (1 - G)
        gh = Fn.relu(hf - off) - Fn.relu(-hf - off)
        out = ((hf - gh) + F @ self.Wbank[self.L]).to(h.dtype)
        if self.capture is not None: self.capture.append((h.detach().clone(), out.detach().clone()))
        if args: return (out,) + tuple(args[1:]), kwargs
        kwargs['hidden_states'] = out; return args, kwargs
