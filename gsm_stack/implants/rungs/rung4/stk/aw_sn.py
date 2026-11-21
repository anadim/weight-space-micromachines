"""lane sidenet (AW_SN=path/sn_w.pt, needs AW_VF=1): multi-layer verifier score as per-slot register + BLD answer-group writer.
Side-net = ONE fixed linear map (5-fold-mean of the trainC-trained c4:HM:lin:pair models, folded with the trainC standardization)
over 8 x 1024 layer-input states: H = states at the token completing '####' (layers 14, 18, 22, 27), M = mean of the states at
host-read tokens containing '\\n' between the segment start and the '####' token (same layers).  Because the map is linear, the
stack projects each read state immediately (x_L . [wH_L, wM_L], fixed (1024, 2) per layer) and keeps per row only
  s_snH  = projection of the '####' states (set at the '####' token, like the L18 probe register s_lg),
  s_snM  = running sum of the line-end projections, s_sncnt = line-end counter (reset with the segment),
and commits R_sn[slot] = s_snH + s_snM / max(cnt, 1) + b at the attempt's END (with R_lg).
Writer (vf_vote_sn): ClusterLin over answer groups exactly as aw_vf.vf_vote, with the per-slot lg feature replaced by the blend
b_i = max(sn_i, -12) + max(lg_i, -12) and the group sum sigmoid(lg) replaced by sum sigmoid(b / 2) (offline mode 'BLD',
repick_sn.py), theta fitted on trainC (out-of-fold side-net scores), stored in the vf_vote feature order."""
import torch, torch.nn.functional as F

NSYM = 13
BLD = ['c8', 'lc', 'ssb', 'mean_sane', 'mean_nn', 'mean_zt', 'mean_qc', 'max_sane', 'max_nn', 'max_zt', 'max_qc',
       'min_sane', 'min_nn', 'min_zt', 'min_qc', 'mean_b', 'max_b', 'min_b']
VFORD = ['c8', 'lc', 'ssb'] + [f'{a}_{b}' for a in ('mean', 'max', 'min') for b in ('b', 'sane', 'nn', 'zt', 'qc')]


def to_vf_order(th, names):
    """theta over the offline BLD feature names -> the vf_vote g order"""
    assert list(names) == BLD, names
    return [float(th[names.index(k)]) for k in VFORD]


def load(path, dev):
    P = torch.load(path, map_location='cpu')
    L = [int(x) for x in P['layers']]; D = int(P['D']); w = P['w'].float()
    assert P['kinds'] == 'HM' and w.numel() == 2 * len(L) * D
    wl = {}
    for j, l in enumerate(L):
        wl[l] = torch.stack([w[j * D:(j + 1) * D], w[(len(L) + j) * D:(len(L) + j + 1) * D]], 1).contiguous().to(dev)   # (D, 2) [H, M]
    th = torch.tensor(P['th_vf'], dtype=torch.float64, device=dev); assert th.numel() == 18
    return L, wl, float(P['b']), th, float(P['c0'])


def vf_vote_sn(R_ok, R_bad, R_lg, R_sn, R_sym, R_nn, R_qc, R_nt, n, k_san, th, c0):
    """returns has (B,), widx (B,) over slots < n (per row); = aw_vf.vf_vote with Fv[0] = max(sn,-12) + max(lg,-12), ssig -> sum sigmoid(b/2)"""
    dev = R_ok.device; NS, KS = R_sym.shape[1], R_sym.shape[2]
    ar = torch.arange(NS, device=dev)
    used = ar[None] < n[:, None]
    ok = R_ok & used
    sane = ok & ~R_bad
    anys = sane.any(1, keepdim=True) & k_san
    keep = ok & (sane | ~anys)
    Rh = F.one_hot(R_sym, NSYM).float()
    eq = torch.relu(torch.einsum('bikc,bjkc->bij', Rh, Rh) - (KS - 1))
    usd = used.double(); nu = usd.sum(1, keepdim=True)
    mu = (R_nt * usd).sum(1, keepdim=True) / nu.clamp(min=1)
    sd = (((R_nt - mu) ** 2 * usd).sum(1, keepdim=True) / nu.clamp(min=1)).sqrt()
    zt = torch.where(nu > 1, (R_nt - mu) / (sd + 1e-3), torch.zeros_like(R_nt))
    bl = R_sn.double().clamp(min=-12.0) + R_lg.double().clamp(min=-12.0)
    Fv = torch.stack([bl, sane.double(), R_nn / 10, zt, R_qc], -1)
    Mk = (eq > 0.5) & keep[:, None, :]; Md = Mk.double()
    cnt = Md.sum(-1, keepdim=True); c1 = cnt.clamp(min=1)
    big = torch.where(Mk[..., None], Fv[:, None], torch.full_like(Fv[:, None], -1e9)).amax(2)
    small = torch.where(Mk[..., None], Fv[:, None], torch.full_like(Fv[:, None], 1e9)).amin(2)
    g = torch.cat([cnt / 8, torch.log(c1), Md @ torch.sigmoid(Fv[..., :1] / 2), (Md @ Fv) / c1, big, small], -1)
    g = torch.where(keep[..., None], g, torch.zeros_like(g))
    sc = g @ th + c0
    sc = torch.where(keep, sc - 1e-9 * ar.double(), torch.full_like(sc, -1e18))
    return keep.any(1), sc.argmax(1)
