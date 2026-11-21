"""lane dola: contrastive layer decoding (DoLa-style) as a fixed linear map on two residuals + a compiled comparator.
DOLA=l:alpha[:beta]
  z   = E . n(h_fin)     host logits (n = the host's final RMSNorm, E = tied lm_head / embedding)
  z_l = E . n(h_l)       h_l = layer-l input residual (same read point as the side-net), same norm weights, same E
  z'  = z + alpha (z - z_l) = E . ((1 + alpha) n(h_fin) - alpha n(h_l))
  APC (beta > 0): z'_i = -inf where z_i < max(z) + log(beta)   (p_i < beta * p_max under the host distribution, T = 1)
Applied only on rows whose step has no write (serial-writer queue token, calc_s4 write): those rows keep z exactly.
The host's sampler then draws from z' / T as before (Gumbel-max, one pass)."""
import math, torch


def parse(s):
    p = s.split(':')
    L = int(p[0]); a = float(p[1]); b = float(p[2]) if len(p) > 2 else 0.0
    assert 0 < L < 28 and a > 0 and 0 <= b < 1, s
    return L, a, b


def dola_logits(z, zl, on, alpha, beta):
    """z, zl: (B, V) logits (bf16 or fp32); on: (B,) bool. Returns fp32 (B, V)."""
    zf = z.float()
    zp = zf + alpha * (zf - zl.float())
    if beta > 0:
        zp = zp.masked_fill(zf < zf.amax(-1, keepdim=True) + math.log(beta), float('-inf'))
    return torch.where(on[:, None], zp, zf)
