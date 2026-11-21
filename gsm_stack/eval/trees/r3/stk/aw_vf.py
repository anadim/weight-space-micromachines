"""Answer-group voter (lane vfeat, ClusterLin trainC fit) as fixed register logic on the all-weights serial stack (AW_VF=1).
Per committed slot i: Fv_i = [max(lg_i, -12), sane_i, nn_i / 10, z_ntok_i, qc_i] where nn = distinct numbers in the attempt body
(numreg automaton), qc = fraction of the question's distinct numbers present in the body (1 if none), z_ntok = within-row z-score of
the attempt's final-segment length over the used slots (0 if one slot). Per answer group (slots with all-symbol-equal answers,
kept slots only): g = [cnt/8, log cnt, sum sigmoid(lg), mean Fv, max Fv, min Fv] (18); score = g . th + c0 (fixed weights,
implants/vfeat/job/stk/vf_w.pt). Pick = the earliest kept slot of the best-scoring group. Pure torch (comp_serial + CPU fuzz)."""
import torch, torch.nn.functional as F

NSYM = 13


def vf_qc(Sq, nq, Sb, nb):
    """question-number coverage of the body set: |{q in Sq[:nq]} & {b in Sb[:nb]}| / nq (1 if nq == 0)"""
    ab = torch.arange(Sb.shape[1], device=Sb.device)[None, None]
    inb = ((Sq[:, :, None] == Sb[:, None, :]) & (ab < nb[:, None, None])).any(-1)
    qv = torch.arange(Sq.shape[1], device=Sq.device)[None] < nq[:, None]
    return torch.where(nq > 0, (inb & qv).sum(1).double() / nq.clamp(min=1).double(), torch.ones_like(Sq[:, 0]))


def vf_vote(R_ok, R_bad, R_lg, R_sym, R_nn, R_qc, R_nt, n, k_san, th, c0):
    """returns has (B,), widx (B,) over slots < n (per row)"""
    dev = R_ok.device; NS, KS = R_sym.shape[1], R_sym.shape[2]
    ar = torch.arange(NS, device=dev)
    used = ar[None] < n[:, None]
    ok = R_ok & used
    sane = ok & ~R_bad
    anys = sane.any(1, keepdim=True) & k_san
    keep = ok & (sane | ~anys)
    Rh = F.one_hot(R_sym, NSYM).float()
    eq = torch.relu(torch.einsum('bikc,bjkc->bij', Rh, Rh) - (KS - 1))       # 1 iff all KS symbols agree
    usd = used.double(); nu = usd.sum(1, keepdim=True)
    mu = (R_nt * usd).sum(1, keepdim=True) / nu.clamp(min=1)
    sd = (((R_nt - mu) ** 2 * usd).sum(1, keepdim=True) / nu.clamp(min=1)).sqrt()
    zt = torch.where(nu > 1, (R_nt - mu) / (sd + 1e-3), torch.zeros_like(R_nt))
    Fv = torch.stack([R_lg.double().clamp(min=-12.0), sane.double(), R_nn / 10, zt, R_qc], -1)       # (B, NS, 5)
    Mk = (eq > 0.5) & keep[:, None, :]; Md = Mk.double()
    cnt = Md.sum(-1, keepdim=True); c1 = cnt.clamp(min=1)
    big = torch.where(Mk[..., None], Fv[:, None], torch.full_like(Fv[:, None], -1e9)).amax(2)
    small = torch.where(Mk[..., None], Fv[:, None], torch.full_like(Fv[:, None], 1e9)).amin(2)
    g = torch.cat([cnt / 8, torch.log(c1), Md @ torch.sigmoid(Fv[..., :1]), (Md @ Fv) / c1, big, small], -1)
    g = torch.where(keep[..., None], g, torch.zeros_like(g))
    sc = g @ th + c0
    sc = torch.where(keep, sc - 1e-9 * ar.double(), torch.full_like(sc, -1e18))
    return keep.any(1), sc.argmax(1)
