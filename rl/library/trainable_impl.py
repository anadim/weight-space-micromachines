"""Arm C: make every BBH micromachine's own weights trainable (readers, stack nonzero values + biases, writer), initialised
EXACT, with the writer softened (W2ws-style log-gain) so the greedy u0 tokens are unchanged but p(written token) is not
saturated. Sparsity patterns stay fixed. The decode path keeps reading the SAME storage (CUDA-graph safe): parameters
are updated in place by the optimizer, and the derived decode copies (CSR values, DenseStack dense matrices, the
effective write D) are refreshed in place after each update (refresh()).

Teacher-forced features with grad: Micromachine.emit through ReLUNet.lin replaced by a differentiable sparse product
when grad is enabled (torch.sparse.mm on a COO built from the live value parameter)."""
import math, copy, types, torch, torch.nn as nn, torch.nn.functional as Fn


def _lin_grad(self, l, x):
    """ReLUNet.lin with gradient w.r.t. the nonzero values and the bias (fixed sparsity)."""
    b = getattr(self, f'b{l}')
    if not self.sparse:
        return Fn.linear(x, getattr(self, f'w{l}'), b)
    if not torch.is_grad_enabled():
        return _lin_orig(self, l, x)
    Wsp = torch.sparse_coo_tensor(getattr(self, f'wi{l}'), getattr(self, f'wv{l}'), getattr(self, f'shape{l}'), is_coalesced=True)
    flat = x.reshape(-1, x.shape[-1])
    return (torch.sparse.mm(Wsp, flat.T).T + b).reshape(*x.shape[:-1], -1)


_lin_orig = None


def _netparams(net):
    """turn wv{l}, b{l} buffers into Parameters sharing storage; returns (values, biases) lists."""
    net = getattr(net, 'net', net)
    vals, bs = [], []
    for l in range(net.L + 1):
        for nm, lst in ((f'wv{l}', vals), (f'b{l}', bs)):
            if not hasattr(net, nm): continue
            t = getattr(net, nm)
            del net._buffers[nm]
            p = nn.Parameter(t, requires_grad=True)
            net.register_parameter(nm, p); lst.append(p)
    global _lin_orig
    if _lin_orig is None: _lin_orig = type(net).lin
    net.lin = types.MethodType(_lin_grad, net)
    return vals, bs


class Snap(nn.Module):
    """stack-2 output snapped to the exact one-hot it is designed to be (argmax if max > 0.5, else all-zero) in the forward
    pass, straight-through gradient in the backward pass. At the exact initial weights the stack output already IS that
    one-hot, so u0 is unchanged; after training it keeps the implant silent/discrete (no tiny nonzero leak everywhere)."""
    def __init__(self, net):
        super().__init__(); self.net = net
    def forward(self, x):
        y = self.net(x)
        m, i = y.max(-1, keepdim=True)
        hard = torch.zeros_like(y).scatter_(-1, i, 1.0) * (m > 0.5).to(y.dtype)
        return hard + (y - y.detach()) if torch.is_grad_enabled() else hard


class TrainableTool:
    def __init__(self, tool, gain_target=None, ste_reader=False, freeze=()):
        self.t = tool; M = tool.M
        nets_ = [getattr(n, 'net', n) for n in (M.stack1, M.stack2)]
        caches = [n._csr for n in nets_]
        for n in nets_: n._csr = {}
        self.frozen = copy.deepcopy(M)                         # exact original, for drift
        for n, c in zip(nets_, caches): n._csr = c
        for p_ in self.frozen.parameters(): p_.requires_grad_(False)
        # reader
        rw, rb = M.reader_w, M.reader_b
        del M._buffers['reader_w']; del M._buffers['reader_b']
        M.register_parameter('reader_w', nn.Parameter(rw)); M.register_parameter('reader_b', nn.Parameter(rb))
        self.reader = [M.reader_w, M.reader_b]
        if ste_reader:                       # straight-through reader: forward = exact clamp(.,0,1); backward = identity
            def _sym(self_, emb, ids=None):
                pre = Fn.linear(emb.float(), self_.reader_w, self_.reader_b)
                hard = torch.clamp(pre, 0, 1)
                return hard if not torch.is_grad_enabled() else pre + (hard - pre).detach()
            M.symbols = types.MethodType(_sym, M)
        v1, b1 = _netparams(M.stack1); v2, b2 = _netparams(M.stack2)
        M.stack2 = Snap(M.stack2)                               # discrete forward, straight-through backward
        self.values, self.biases = v1 + v2, b1 + b2
        self.freeze = set(freeze)
        if 'core' in self.freeze:
            for p_ in self.values + self.biases: p_.requires_grad_(False)
        if 'reader' in self.freeze:
            for p_ in self.reader: p_.requires_grad_(False)
        # writer: unit directions (fixed rows of out_dirs, trainable) x soft gain (trainable log-gain)
        od = M.out_dirs.detach().clone()
        self.dirs = nn.Parameter(od.float())                   # (n_out, d), rms-1 rows (or 0 rows)
        if tool.perm is not None: self.dirs.data = self.dirs.data[tool.perm]
        self.log_gain = nn.Parameter(torch.tensor(math.log(gain_target or float(M.gain)), device=od.device))
        self.refresh()

    def D_eff(self):
        return self.dirs * torch.exp(self.log_gain)

    @torch.no_grad()
    def refresh(self):
        """in-place copies into the decode path's derived tensors (same storage the CUDA graphs read)."""
        t, M = self.t, self.t.M
        t.D.copy_(self.D_eff().float())
        for net in (M.stack1, getattr(M.stack2, 'net', M.stack2)):
            n = getattr(net, 'net', net)
            for key, csr in n._csr.items():                     # CSR values: same (row-major) order as the coalesced COO values
                l = key[0]; csr.values().copy_(getattr(n, f'wv{l}').detach().to(csr.dtype))
        if t.fast1 is not None:                                # dense decode copy of the deep stack
            n = getattr(M.stack1, 'net', M.stack1)
            for l, (idx, Wd, b) in enumerate(t.fast1.layers):
                wi = getattr(n, f'wi{l}'); Wd[wi[0], wi[1]] = getattr(n, f'wv{l}').detach().to(Wd.dtype)
                b.copy_(getattr(n, f'b{l}').detach().to(b.dtype)[:, None])

    def groups(self, lr_reader, lr_values, lr_writer, eps):
        """Adam groups; lrs relative to each tensor's scale (weights are small integers, biases too)."""
        sc = lambda ps: 1.0                                    # absolute lrs (weights span 1 .. 1e14 in expr_gen: no relative scaling)
        gs = [dict(params=self.reader, lr=lr_reader * sc(self.reader), eps=eps, name=f'{self.t.name}:reader'),
              dict(params=self.values + self.biases, lr=lr_values * sc(self.values), eps=eps, name=f'{self.t.name}:core'),
              dict(params=[self.dirs, self.log_gain], lr=lr_writer, eps=eps, name=f'{self.t.name}:writer')]
        return [g for g in gs if g['name'].split(':')[1] not in self.freeze]

    @torch.no_grad()
    def drift(self, emb_streams, head_fn, out_tok_ids):
        """exactness vs the frozen original on fixed token streams: fraction of positions where the original fires and
        (i) the trained machine's one-hot picks the same output, (ii) its written direction still decodes to that token."""
        e0, _ = self.frozen.emit(emb_streams); e1, _ = self.t.M.emit(emb_streams)
        fire = e0.abs().sum(-1) > 0
        if not fire.any(): return dict(n=0)
        a0, a1 = e0.argmax(-1)[fire], e1.argmax(-1)[fire]
        same = (a0 == a1).float().mean().item()
        D = self.D_eff().float()
        pred = head_fn(D)                                      # (n_out,) argmax token of each written direction
        ok_rows = (pred == out_tok_ids).float()
        wr = ok_rows[a1].mean().item()
        return dict(n=int(fire.sum()), same_output=same, write_decodes=wr, exact=float(((a0 == a1).float() * ok_rows[a1]).mean()))
