"""[38] arm I: the exact calculator implant on the <calc>EXPR</calc> call path (no Python at decode time).

Firing = arm T's: a row pauses on a sampled '>' within 4 tokens of a '/', and at the sync the engine checks that its text
now ends a '</calc>'; EXPR = the text after the last '<calc>' before it (the same events as the text tool). The frozen
exact CORE (core.py, int64 tensor program) evaluates EXPR -> result symbols (0-9 . - ERR) + registers.
WIRING (trainable in arm I; I-tok: writer only; I0: frozen writer, no in-wire, no adapters):
  writer   at 'final': gate * W e   (e = one-hot of the symbol being written; W0 = hand directions, gate 1). The host
           SAMPLES the next token with the write added (soft write); the token counts as implant-written (not in the
           model-token cap) but is the policy's sample (in the policy loss).
  adapters CWOut at host layers 7/14/21/final: g(h) * B relu(A f + c), f = [e, registers] while writing, B = 0 at init
           (exact zero output), mask f != 0 (silent on every other token).
  in-wire  at each call: admit with p = eps + (1-2eps) sigmoid(k/2 + b + v.LN(h_L)), h_L = host residual at layer L
           input at the step that sampled the '>' (init P = 0.95 before the floor); veto = no write (the model goes on).
Feature per write step: feat = [e (13), registers (45)] (zeros elsewhere). Registers are stored as a 45-bit mask.
"""
import math, os, torch, torch.nn as nn, torch.nn.functional as F
from core import ExactCore, N_SYM, R_DIM, SYM_TXT, FMAX

NF = N_SYM + R_DIM


def _ln(h):
    hf = h.float()
    return (hf - hf.mean(-1, keepdim=True)) * torch.rsqrt(hf.var(-1, unbiased=False, keepdim=True) + 1e-5)


class Writer(nn.Module):
    """token write: gate * (W e) at the final residual (before the final norm)."""
    def __init__(self, W0):
        super().__init__()
        self.W = nn.Parameter(W0.float().clone())                     # (D, N_SYM)
        self.gate = nn.Parameter(torch.ones(W0.shape[0], device=W0.device))
        self.register_buffer('W0', W0.float().clone())
        self.w_scale = max(1.0, float(self.W.detach().pow(2).mean().sqrt()))

    def forward(self, feat, h=None):
        return self.gate * F.linear(feat[..., :N_SYM].float(), self.W)


class CWOut(nn.Module):
    """CW-full out-wire (lr27 CWOut): g(h) * B relu(A f + c); B = 0 (or N(0, std)), w_g = b_g = 0; masked to f != 0."""
    def __init__(self, nf, d, hidden=64, c=0.1, std=0.0, dev='cuda'):
        super().__init__()
        self.A = nn.Parameter(torch.randn(hidden, nf, device=dev))
        self.c = nn.Parameter(torch.full((hidden,), float(c), device=dev))
        self.B = nn.Parameter(torch.randn(d, hidden, device=dev) * std if std > 0 else torch.zeros(d, hidden, device=dev))
        self.wg = nn.Parameter(torch.zeros(d, device=dev)); self.bg = nn.Parameter(torch.zeros((), device=dev))
        self.last = None

    def forward(self, feat, h=None):
        f = feat.float()
        z = torch.relu(F.linear(f, self.A) + self.c)
        g = torch.sigmoid(_ln(h) @ self.wg + self.bg)
        out = g[..., None] * F.linear(z, self.B)
        m = (f.abs().sum(-1) > 0)
        out = out * m[..., None].to(out.dtype)
        if torch.is_grad_enabled(): self.last = out.detach().norm(dim=-1)[m].mean() if bool(m.any()) else None
        return out


class LinkSum(nn.Module):
    def __init__(self, ls):
        super().__init__(); self.ls = nn.ModuleList(ls)

    def forward(self, feat, h=None):
        return sum(l(feat, h) for l in self.ls)


class InWire(nn.Module):
    """lr27 InWire; trigger = 1 (a closed call always has a result). P(admit) at init = eps + (1-2eps) 0.95."""
    def __init__(self, L, d, eps, dev='cuda'):
        super().__init__()
        lo, hi = math.log(0.95 / 0.05), math.log(0.02 / 0.98)
        self.k = lo - hi; self.b0 = (lo + hi) / 2; self.L = L; self.eps = float(eps)
        self.b = nn.Parameter(torch.full((), self.b0, device=dev)); self.v = nn.Parameter(torch.zeros(d, device=dev))

    def logit(self, h):
        return self.k * 0.5 + self.b + _ln(h) @ self.v

    def logp(self, logit, admit):
        """log P(decision); admit bool tensor."""
        if self.eps == 0.0:
            return torch.where(admit, F.logsigmoid(logit), F.logsigmoid(-logit))
        p = self.eps + (1 - 2 * self.eps) * torch.sigmoid(logit)
        return torch.where(admit, torch.log(p), torch.log1p(-p))

    def decide(self, logit, greedy):
        if greedy: return logit > 0
        p = self.eps + (1 - 2 * self.eps) * torch.sigmoid(logit)
        return torch.rand_like(p) < p


def sym_token_ids(tok):
    ids = []
    for s in SYM_TXT:
        t = tok(s, add_special_tokens=False)['input_ids']
        assert len(t) == 1, (s, t)
        ids.append(t[0])
    return ids


def bits_of(regs):
    """(N, R_DIM) 0/1 float -> (N,) int64 bitmask."""
    ar = torch.arange(R_DIM, device=regs.device)
    return ((regs > 0.5).long() << ar).sum(-1)


def unbits(rb):
    ar = torch.arange(R_DIM, device=rb.device)
    return ((rb[..., None] >> ar) & 1).float()


def feat_of(sym, rb):
    """sym (...,) long (-1 = no write), rb (...,) int64 -> (..., NF) float, zero where sym < 0."""
    w = (sym >= 0)
    e = (sym[..., None] == torch.arange(N_SYM, device=sym.device)).float()
    return torch.cat([e, unbits(rb)], -1) * w[..., None].float()


def hand_dirs(x, lnf, E, sym_ids, steps=300, m_target=9.0, log=print):
    """Hand write directions d_s (D, N_SYM): adding d_s to the final residual makes token s the argmax with a large
    margin at every calibration position. x: (P, D) fp32 final residuals (pre-norm) of the host; lnf (D,); E (V, D).
    Optimised in units of the residual rms (L2 penalty keeps them small); returns W0 (D, N_SYM) and a report."""
    torch.manual_seed(1234)
    P, D = x.shape
    perm = torch.randperm(P, device=x.device); tr, te = x[perm[:P // 2]], x[perm[P // 2:]]
    xr = float(x.pow(2).mean().sqrt())
    Eb = E.float()
    out = torch.zeros(D, N_SYM, device=x.device); rep = {}

    def margins(xx, d, t):
        z = xx + d[None]
        h = z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-6) * lnf.float()
        lg = h @ Eb.T
        lt = lg[:, t].clone(); lg[:, t] = -1e9
        return lt - torch.logsumexp(lg, -1)
    for s, t in enumerate(sym_ids):
        u = Eb[t] / lnf.float().clamp(min=1e-3)
        u = u / u.pow(2).mean().sqrt()
        delta = (3.0 * u).clone().requires_grad_(True)
        opt = torch.optim.Adam([delta], lr=0.05)
        for it in range(steps):
            mg = margins(tr, delta * xr, t)
            loss = F.softplus(m_target - mg).mean() + 1e-3 * delta.pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            d = delta.detach() * xr
            mte = margins(te, d, t); mtr = margins(tr, d, t)
            pmin = float(torch.sigmoid(mte).min())                 # P(token) >= sigmoid(margin)
            rep[SYM_TXT[s]] = dict(rms_rel=float(delta.detach().pow(2).mean().sqrt()), min_margin_tr=float(mtr.min()),
                                   min_margin_te=float(mte.min()), mean_margin_te=float(mte.mean()), p_min_te=pmin)
            out[:, s] = d
    log('hand write directions:', {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in rep.items()})
    return out, rep


def calib_W0(Q, Wc, pre, sym_ids, dev, log=print):
    """hand directions from the host's final residuals (pre-norm) on the calc prefix (Wc: fused weights)."""
    cap = {}
    def f(feat, x):
        cap['x'] = x.detach().float(); return torch.zeros_like(x)
    cos_t, sin_t = Q.rope_tables(len(pre) + 8, dev)
    with torch.no_grad():
        Q.fwd_chunk(Wc, torch.tensor(pre, device=dev)[None], 0, cos_t, sin_t, links={'final': f})
    return hand_dirs(cap['x'][0], Wc['lnf'], Wc['E'], sym_ids, log=log)


def load_wiring(path, sym_ids, dev='cuda'):
    """ArmI from a saved wiring.pt (train2.save)."""
    ck = torch.load(path, map_location=dev)
    ar, st = ck['args'], ck['state']
    ci = ArmI(ck['arm'], st['writer.W0'].to(dev), sym_ids,
              layers=tuple(int(l) if l != 'final' else 'final' for l in ar['ci_layers'].split(',')),
              iw_eps=ar['iw_eps'], iw_layer=ar['iw_layer'],
              adapters=(bool(ar['ci_adapters']) if ck['arm'] == 'I' else None), dev=dev)
    ci.load_state_dict(st)
    return ci


class ArmI(nn.Module):
    def __init__(self, arm, W0, sym_ids, layers=(7, 14, 21, 'final'), cw_hidden=64, cw_c=0.1, link_std=0.0, iw_eps=0.05,
                 iw_layer=14, adapters=None, inwire=None, dev='cuda'):
        """arm: 'I' (writer + adapters + in-wire, all trainable), 'Itok' (writer trainable only), 'I0' (frozen writer)."""
        super().__init__()
        D = W0.shape[0]
        self.arm = arm
        self.core = ExactCore(os.environ.get('CORE_DEV', 'cpu'))     # launch-bound int64 program: ~3x faster on CPU at engine batch sizes
        self.sym_tok = torch.tensor(sym_ids, device=dev)
        self.writer = Writer(W0)
        adapters = (arm == 'I') if adapters is None else adapters
        inwire = (arm == 'I') if inwire is None else inwire
        self.adapters = nn.ModuleDict()
        if adapters:
            g = torch.Generator(device='cpu').manual_seed(4242)
            for l in layers:
                cw = CWOut(NF, D, cw_hidden, cw_c, 0.0, dev)
                with torch.no_grad():
                    cw.A.copy_(torch.randn(cw.A.shape, generator=g).to(dev))
                    if link_std > 0: cw.B.copy_((torch.randn(cw.B.shape, generator=g) * link_std).to(dev))
                self.adapters[str(l)] = cw
        self.inwire = InWire(iw_layer, D, iw_eps, dev) if inwire else None
        self.links = {}
        for l in layers:
            if str(l) in self.adapters:
                self.links[l if l == 'final' else int(l)] = self.adapters[str(l)]
        self.links['final'] = LinkSum([self.writer, self.adapters['final']]) if 'final' in self.adapters else self.writer
        if arm == 'I0':
            for p in self.parameters(): p.requires_grad_(False)

    def param_groups(self, link_lr, eps=1e-12, iw_lr=None):
        if self.arm == 'I0': return []
        gs = [{'params': [self.writer.gate], 'lr': link_lr, 'eps': eps},
              {'params': [self.writer.W], 'lr': link_lr * self.writer.w_scale, 'eps': eps}]
        if len(self.adapters): gs.append({'params': list(self.adapters.parameters()), 'lr': link_lr, 'eps': eps})
        if self.inwire is not None: gs.append({'params': list(self.inwire.parameters()), 'lr': iw_lr or link_lr, 'eps': eps})
        return gs

    @torch.no_grad()
    def norms(self):
        r = dict(writer_dW=float((self.writer.W - self.writer.W0).norm() / self.writer.W0.norm()),
                 writer_gate=float(self.writer.gate.mean()))
        for k, cw in self.adapters.items():
            r[f'cw{k}_B'] = float(cw.B.norm()); r[f'cw{k}_g0'] = float(torch.sigmoid(cw.bg))
            r[f'cw{k}_wg'] = float(cw.wg.norm())
            if cw.last is not None: r[f'cw{k}_out'] = float(cw.last)
        if self.inwire is not None:
            r['iw_b'] = float(self.inwire.b); r['iw_v'] = float(self.inwire.v.norm())
        return r

    def wiring_state(self):
        return {k: v.detach().cpu() for k, v in self.state_dict().items()}
