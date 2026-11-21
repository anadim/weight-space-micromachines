"""Offline steering-implant simulation (step 4).  python3.11 probe/branch/sim.py --model out_models_trainC_pairs.pt --name had_22

Implant: at a sampled token where the host's top-2 are close (p2/p1 >= rho, token not written / calc), compute the probe
score s(top1, top2) from h and either
  'calib': replace the pair log-odds by s  (logit_top1 - logit_top2 := s; the rest of the distribution untouched), or
  'hard d': add +d to the probe-preferred token when |s - dlp| (the probe's correction) >= tau.
Events:
  branch: atlas branch points of wrong attempts (dev/dev2), right token r vs wrong w (if {r,w} = host top-2 and gate on)
          -> gain = P'(r) - P(r) (prob of taking the right branch at the fork that sent this attempt wrong; n_wrong-weighted)
  ctrl:   random close forks (<= 2 per attempt) along RIGHT attempts of mixed items -> loss = P(t) - P'(t), t = token the
          right attempt took; scaled by the number of gated forks per right attempt (from the per-position host stats).
"""
import os, sys, json, argparse, collections
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fit as FT


def pair_probs(lp1, lp2, dl_new):
    """host full-softmax probs p1, p2 of the pair; new probs after setting log(p1'/p2') = dl_new keeping p1+p2 fixed"""
    p1, p2 = np.exp(lp1), np.exp(lp2); m = p1 + p2
    q1 = m / (1 + np.exp(-dl_new)); return p1, p2, q1, m - q1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', default='out_models_trainC_pairs.pt'); ap.add_argument('--name', default='had_22')
    ap.add_argument('--sets', default='dev,dev2')
    a = ap.parse_args()
    st, cfg = torch.load(f'{HERE}/{a.model}', weights_only=False)[a.name]
    kind, layers = cfg[0], cfg[1]; r = cfg[3] if len(cfg) > 3 else 16
    mdl = FT.Model(kind, layers, r); mdl.load_state_dict(st); mdl.cfg = cfg
    res = {}
    for S in a.sets.split(','):
        d = FT.load_set(S)
        # ---- branch events
        P = d['pairs']; F = FT.feats(d, P); s = FT.score(mdl, F)          # s>0 => right token preferred
        M = d['M']
        rows = np.array([p['r'] for p in P])
        lp_top = M['top_lp'][rows]; top = M['top_id'][rows]
        lpa = F['dlp'].numpy() * 0 + np.array([p['lpa'] for p in P]); lpb = np.array([p['lpb'] for p in P])
        nw = np.array([p['n_wrong'] for p in P], float)
        top2 = np.array([p['top2'] for p in P])
        # ---- control events: right attempts' close forks
        CH = np.load(f'{FT.OUT}/{S}_ctrl_h.npy').astype(np.float32); CM = dict(np.load(f'{FT.OUT}/{S}_ctrl_meta.npz'))
        CH /= np.sqrt((CH ** 2).mean(-1, keepdims=True)) + 1e-6
        okA = CM['ok'][CM['att']].astype(bool); bA = CM['b'][CM['att']]
        keep = okA                                                       # right attempts only
        ct = CM['tok'][keep]; t1, t2 = CM['top_id'][keep, 0], CM['top_id'][keep, 1]; l1, l2 = CM['top_lp'][keep, 0], CM['top_lp'][keep, 1]
        in2 = (ct == t1) | (ct == t2)
        alt = np.where(ct == t1, t2, t1)
        U = FT.unemb()
        ok_tok = np.array([int(x) in U['map'] and int(y) in U['map'] for x, y in zip(ct, alt)]) & in2
        Fc = dict(hn=torch.tensor(CH[keep][ok_tok]), du=torch.tensor(((U['U'][[U['map'][int(x)] for x in ct[ok_tok]]] -
                                                                       U['U'][[U['map'][int(x)] for x in alt[ok_tok]]]) * U['g']).astype(np.float32)),
                  dlp=torch.tensor(np.where(ct == t1, l1 - l2, l2 - l1)[ok_tok].astype(np.float32)))
        sc = FT.score(mdl, Fc)                                             # >0 => keep the right attempt's token
        lpt = np.where(ct == t1, l1, l2)[ok_tok]; lpo = np.where(ct == t1, l2, l1)[ok_tok]
        ratio_c = np.exp(np.minimum(l1, l2) - np.maximum(l1, l2))[ok_tok]
        # per right attempt: number of gated positions (host top-2 ratio >= rho, wf == 0)
        PS = np.load(f'{FT.OUT}/{S}_pos.npz'); okP = CM['ok'][PS['att']].astype(bool)
        n_right_att = int(CM['ok'].sum())
        out = {}
        for rho in (0.3, 0.5, 0.7):
            gate_b = top2 & (np.exp(-np.abs(lpa - lpb)) >= rho)
            npos = int(((PS['wf'] == 0) & okP & ((PS['lp2'].astype(np.float32) - PS['lp1'].astype(np.float32)) >= np.log(rho))).sum())
            forks_per_right = npos / max(1, n_right_att)
            gc = ratio_c >= rho
            for mode in ('calib', 'hard1', 'hard2', 'hard3'):
                for tau in ((0.0,) if mode == 'calib' else (0.0, 0.5, 1.0)):
                    corr_b = s - (lpa - lpb); corr_c = sc - Fc['dlp'].numpy()
                    if mode == 'calib':
                        nb = np.where(gate_b, s, lpa - lpb); nc = np.where(gc, sc, Fc['dlp'].numpy())
                    else:
                        dlt = float(mode[-1])
                        nb = np.where(gate_b & (np.abs(corr_b) >= tau), (lpa - lpb) + dlt * np.sign(s), lpa - lpb)
                        nc = np.where(gc & (np.abs(corr_c) >= tau), Fc['dlp'].numpy() + dlt * np.sign(sc), Fc['dlp'].numpy())
                    _, _, qr, _ = pair_probs(lpa, lpb, nb); pr = np.exp(lpa)
                    gain = (qr - pr)                                             # change of P(take right) at the branch
                    _, _, qt, _ = pair_probs(lpt, lpo, nc); pt = np.exp(lpt)
                    loss = (pt - qt)                                             # drop of P(right attempt's token)
                    key = f'rho{rho}|{mode}|tau{tau}'
                    out[key] = dict(gate_branch=float(np.average(gate_b, weights=nw)),
                                    branch_gain_mean=float(np.average(gain, weights=nw)),
                                    branch_flip_toward_r=float(np.average((gain > 0) & gate_b, weights=nw)),
                                    branch_flip_away=float(np.average((gain < 0) & gate_b, weights=nw)),
                                    ctrl_gated=float(gc.mean()), ctrl_loss_mean_gated=float(loss[gc].mean()) if gc.any() else 0.,
                                    ctrl_away=float(((loss > 0) & gc).sum() / max(1, gc.sum())),
                                    forks_per_right_att=forks_per_right,
                                    exp_dev_per_right_att=float(forks_per_right * (loss[gc].mean() if gc.any() else 0.)),
                                    exp_fix_per_wrong_att=float(np.average(gain * gate_b, weights=nw)))
        res[S] = out
        print(f'== {S}  model {a.name}  branch pairs {len(P)}  ctrl (right att) {int(ok_tok.sum())}  right atts {n_right_att}')
        for k, v in out.items():
            print(f'  {k:22s} gateB {v["gate_branch"]:.2f} fixP/wrong {v["exp_fix_per_wrong_att"]:+.3f} (toward r {v["branch_flip_toward_r"]:.2f} '
                  f'away {v["branch_flip_away"]:.2f}) | forks/right {v["forks_per_right_att"]:.1f} lossP/gated fork {v["ctrl_loss_mean_gated"]:+.3f} '
                  f'away-share {v["ctrl_away"]:.2f} -> exp deviations/right att {v["exp_dev_per_right_att"]:+.3f}')
    json.dump(res, open(f'{HERE}/sim_{a.name}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
