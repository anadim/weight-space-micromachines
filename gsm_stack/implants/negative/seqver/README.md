# lane seqver: SEQUENCE side-net verifier (attention pooling over all body-token states) vs rung-4 sn+lg (todo seqver)

Rung 4 (implants/sidenet/aw, rows sidenet/aw/out/sn_{dev,dev2}: 90.46 / 88.33) reads 2 pooled points per attempt (H = token
completing '####', M = mean of body line ends) at layers 14/18/22/27 + the L18 probe lg. Closed verifier lanes (sidenet MLPs,
sndata, attnver, prmstep) all sit at within-item AUROC ~0.75 dev / ~0.67 dev2 (b = max(sn,-12)+max(lg,-12): trainC 0.6975 /
dev 0.7525 / dev2 0.6726). Here: a learned pooling over ALL body tokens (from the first token after the written phrase through
the '####' token) of low-rank projected states x_t = P_L(h_t - mu) (PCA r=256 on standardized states, fit on a trainC subsample),
layers {14,18,22,27}, + per-token channels (sampled logp, host entropy, the sn per-token projections).
Model families: attention pooling (K heads: Σ softmax_t(q_h·x_t)(v_h·x_t)), max-badness (max_t w·x_t), token mean; 'alone' (+ linear
on projected H/M) and 'inc' (s = b + pooled terms, b frozen). Pairwise within-item loss, trainC 5-fold item CV (idx % 5),
dev/dev2 = 5-fold mean; epoch/config selection on trainC OOF only.
PRE-REGISTERED GATE: within-item AUROC increment >= +0.02 on BOTH dev and dev2 over sn+lg AND offline re-pick (rung-4 BLD writer,
theta refit on trainC) >= +0.3 on BOTH vs rung 4 (paired, same streams, 4 seeds). Fail -> NEGATIVE, no in-stack build.
In-stack plan (only if gated): per pooling head an online-softmax register triple (running max m, Σexp(a-m), Σexp(a-m)·v),
gated to body tokens with the same token tables/flags sidenet uses (host-read, after the written phrase, up to '####', reset at
newseg), P_L folded into the q/v vectors (q_h·P_L is one fixed 1024-vector per layer), committed per slot at END.

## log (UTC, newest last)
- 12:40Z job/ = copy of attnver/job (common.py, run.sh, wsmrepo, verifier_probe_L18.pt, tables incl. sc_g4.npz) + sn_w_g4.pt
  (sidenet job/out/w_g4_c4_HM_lin_pair.pt). job/extract_sq.py (see docstring): pass 1 PCA on 3000 trainC attempts' body tokens
  (per-layer standardized, top-256 eigvecs -> out/pca.pt), pass 2 per-token X (Ntok, 4, 256) fp16 + tok.npz + meta.npz under
  $WSM_DATA/implants/seqver/{set}/ (body tokens: dev 2.68M, dev2 1.46M, trainC 6.70M -> ~22 GB).
  Validation: L18 probe at H vs register lg; sn recomputed from per-token sH/sM vs sidenet sc_g4 (dev/dev2); lp_m vs row logp.
  `cd job && <queue-submit> $PWD 2400 job_seqver_xt -- bash run.sh bash xs_job.sh` -> **j23533**.
- 13:02Z j23533 DONE (12.9 min run, 8.7 min queue). PCA (3000 trainC attempts, 422k body tokens): r=256 keeps 80.6 / 80.2 / 78.8 /
  78.5 % of the standardized variance at L14/18/22/27 (out/pca.pt: mu, P = diag(1/sd)V[:, :256], eigvals). Validation:

  | set | attempts | body tokens | lg (L18 probe at H) vs register corr / MAE / max | sn recomputed vs sidenet sc_g4 | lp_m vs row logp (host tokens) | GB |
  |---|---|---|---|---|---|---|
  | dev | 18477 | 2.68M | 0.9982 / 0.083 / 0.44 | corr 0.99999999, max 0.0011 | 0.9951 | 5.5 |
  | dev2 | 9804 | 1.46M | 0.9983 / 0.081 / 0.42 | corr 0.99999999, max 0.0013 | 0.9951 | 3.0 |
  | trainC | 47525 | 6.70M | 0.9983 / 0.082 / 0.50 | (vs OOF scores: 0.950, expected) | 0.9950 | 13.7 |

  -> body window and context = the sidenet/in-stack ones exactly. Features $WSM_DATA/implants/seqver/{dev,dev2,trainC}/ (22 GB).
- 13:03Z job/train_sq.py (docstring) smoke-tested locally on synthetic features (off/fake; b baseline reproduces 0.6975/0.7525/0.6726).
  off/ana.py (within AUROC, trainC-selected blend weights, exports deltas), repick_sq.py (= attnver/repick_av.py, BLD writer on
  rung-4 rows). Grid q1: `cd job && <queue-submit> $PWD 2400 job_seqver_tr1 -- bash run.sh bash tr_job.sh q1
  c4:x:a4:inc,c4:xc:a4mu:inc,c4:x:h:alone,c4:x:m:inc,c4:x:u:inc,-:c:a2mu:inc,c4:x:a4:alone,c4:xc:a4mh:alone,-:c:a2mu:alone,
  c4:x:a1:inc,l1827:x:a4:inc,c4r64:x:a4:inc --epochs 1,2,4,8,12 --tmax 1900` -> **j23537**.
- 13:20Z j23537 DONE (16 min queue, 2 min run). `python3.11 off/ana.py job/out/sc_q1.npz > off/ana_q1.log`. Within-item AUROC on
  committed attempts. 'alone' = the model's score s; blend = b + v·sd_b·z(s) with v chosen on trainC OOF. 'inc' = s is b + learned
  terms (b frozen), delta = the learned terms.

  | q1 cfg | alone trainC / dev / dev2 | b+blend Δ vs b trainC / dev / dev2 |
  |---|---|---|
  | c4:x:a4:inc / c4:x:u:inc / c4:x:m:inc | – | +0.002/+0.006/+0.003 · +0.004/+0.007/+0.003 · +0.002/+0.005/+0.003 (inc overfits after ep1–2) |
  | c4:x:h:alone (projected H/M linear ≈ sn) | 0.688 / 0.718 / 0.676 | +0.003 / −0.001 / +0.009 |
  | -:c:a2mu:alone (channels only: logp, entropies, sH/sM, ...) | 0.641 / 0.636 / 0.618 | +0.010 / −0.006 / +0.007 |
  | **c4:x:a4:alone** (4-head attention pooling) | 0.689 / 0.770 / 0.694 | **+0.019 / +0.031 / +0.025** (v=0.75, wcorr(s,b) dev 0.65) |
  | c4:xc:a4mh:alone | 0.692 / 0.779 / 0.694 | +0.013 / +0.027 / +0.017 |

  -> the AUROC half of the gate is met by attention pooling 'alone' blended with b. Re-picks (`python3.11 repick_sq.py --sc
  off/sc_inc_q1.npz --cfgs CFG --l2 1e-3,1e-2 [--scales 0.5,2]`, logs off/repick_q1_*.log). Δ = accuracy vs rung 4, mean of 4 seeds;
  f/b = fixes/breaks summed over seeds. X0 = rung-4 θ; BLD = θ refit on trainC (trainC CV is vs the in-sample rung-4 θ, so slightly pessimistic):

  | delta (q1) | X0 dev / dev2 | BLD l2=1e-3 trainC-CV / dev / dev2 | BLD l2=1e-2 trainC-CV / dev / dev2 |
  |---|---|---|---|
  | c4:x:a4:alone v=0.75 | −0.04 (9/10) / +0.08 (8/7) | −0.13 (29/37) / −0.12 (3/6) / +0.42 (8/3) | −0.18 / +0.04 (7/6) / +0.50 (7/1) |
  | same ×0.5 | −0.12 (3/6) / +0.25 (7/4) | −0.10 / −0.04 (3/4) / +0.42 (7/2) | −0.10 / +0.00 (6/6) / +0.17 (3/1) |
  | same ×2 | +0.00 (18/18) / +0.33 (13/9) | −0.32 / −0.04 (8/9) / +0.25 (10/7) | −0.30 / +0.04 (9/8) / +0.50 (10/4) |
  | c4:x:u:inc | ≤ −0.12 / ≤ −0.08 (all modes) | | |
  | c4:xc:a4mh:alone | ≤ 0.00 / ≤ +0.08 (all modes) | | |

- 13:22Z train_sq.py `--perfold 1` (also stores the fold-0 single model's dev/dev2 scores as `cfg#f0`; this matches θ, which
  is fit on single-model OOF scores on trainC). Grid q2 (alone variants): `cd job && <queue-submit> $PWD 2400 job_seqver_tr2 --
  bash run.sh bash tr_job.sh q2 c4:x:a4:alone,c4:x:u:alone,c4:x:m:alone,c4:x:a1:alone,c4:x:a2:alone,c4:x:a8:alone,c4:x:a4u:alone,
  c4:x:a4m:alone,c4:x:a4h:alone,c4:xc:a4:alone,l1827:x:a4:alone,l27:x:a4:alone,l18:x:a4:alone,c4r64:x:a4:alone,c4r128:x:a4:alone
  --epochs 2,3,4,6,8,12 --perfold 1 --tmax 1900` -> **j23540** (3 min). `python3.11 off/ana.py job/out/sc_q2.npz > off/ana_q2.log`:

  | q2 cfg | alone trainC / dev / dev2 | blend Δ vs b (5-fold mean) trainC / dev / dev2 | blend Δ, fold-0 single model dev / dev2 |
  |---|---|---|---|
  | c4:x:u (token mean) | 0.632 / 0.689 / 0.665 | +0.002 / +0.006 / +0.008 | +0.006 / +0.008 |
  | c4:x:m (max-badness) | 0.615 / 0.678 / 0.624 | +0.006 / +0.005 / +0.002 | +0.001 / −0.001 |
  | c4:x:a1 | 0.669 / 0.755 / 0.684 | +0.010 / +0.021 / +0.020 | +0.019 / +0.008 |
  | c4:x:a2 | 0.677 / 0.759 / 0.679 | +0.014 / +0.023 / +0.017 | +0.020 / +0.021 |
  | c4:x:a4 | 0.689 / 0.770 / 0.694 | +0.019 / +0.031 / +0.025 | +0.031 / +0.029 |
  | c4:x:a8 | 0.683 / 0.772 / 0.689 | +0.019 / +0.029 / +0.022 | +0.026 / +0.018 |
  | **c4:x:a4u** (attn + mean) | 0.689 / 0.770 / 0.693 | **+0.018 / +0.032 / +0.025** | +0.032 / +0.030 |
  | c4:x:a4m (attn + max) | 0.676 / 0.778 / 0.697 | +0.012 / +0.029 / +0.026 | +0.022 / +0.032 |
  | c4:x:a4h (attn + projected H/M) | **0.700** / 0.775 / 0.703 | +0.016 / +0.027 / +0.025 | +0.024 / +0.030 |
  | c4:xc:a4 (+ channels) | 0.673 / 0.772 / 0.690 | +0.013 / +0.027 / +0.016 | +0.021 / +0.032 |
  | l1827:x:a4 | 0.674 / 0.748 / 0.703 | +0.015 / +0.019 / +0.021 | +0.012 / +0.015 |
  | l27:x:a4 / l18:x:a4 | 0.617 / 0.667 / 0.647 · 0.660 / 0.737 / 0.697 | +0.005/+0.009/+0.009 · +0.008/+0.017/+0.019 | |
  | c4r64:x:a4 / c4r128:x:a4 | 0.671 / 0.736 / 0.675 · 0.672 / 0.764 / 0.690 | +0.010/+0.014/+0.013 · +0.011/+0.026/+0.020 | |

  Attention pooling is the only family that helps: mean and max pooling add ≈0, and logp/entropy channels add nothing. All 4 layers
  and r=256 matter. The gain holds for a single model (not just from the ensemble).
- 13:30Z off/selm.py measures selection on streams that have both a right and a wrong committed attempt (trainC 1827, dev 706, dev2 384).
  top1 = the verifier's argmax attempt is right. grp-mean = right vs wrong answer-group pairs scored by mean member score.
  grp-Σσ = the same pairs scored by Σσ(s/2) (the writer's vote+verifier feature). `python3.11 off/selm.py off/sc_inc_q2.npz ...` -> off/selm_q2.log:

  | score | top1 trainC / dev / dev2 | grp-mean | grp-Σσ |
  |---|---|---|---|
  | b (rung 4) | 0.702 / 0.768 / 0.685 | 0.747 / 0.770 / 0.763 | 0.865 / 0.905 / 0.877 |
  | b + a4 | 0.725 / 0.782 / 0.716 | 0.771 / 0.814 / 0.791 | 0.870 / 0.909 / 0.884 |
  | b + a4u | 0.729 / 0.788 / 0.727 | 0.771 / 0.815 / 0.792 | 0.870 / 0.911 / 0.884 |
  | b + a4h | 0.719 / 0.792 / 0.724 | 0.765 / 0.810 / 0.799 | 0.868 / 0.909 / 0.888 |
  | b + a8 | 0.731 / 0.793 / 0.714 | 0.764 / 0.812 / 0.793 | 0.869 / 0.910 / 0.883 |

  -> verifier-only selection improves by 2–4 pts. Once votes are counted (Σσ) the group-level gain shrinks to ~0.5 pt. The rows the
  writer gets wrong are mostly ones where the wrong answer has more votes, and the improved score is not strong enough to overturn them.
- 13:40Z q2 re-picks (`python3.11 repick_sq.py --sc off/sc_inc_q2.npz --cfgs q2.c4_x_{a4u,a4h,a8}_alone:inc:sel,...#f0... --l2 1e-3,1e-2`,
  off/repick_q2_{a4u,a4h,a8}.log). Δ vs rung 4 is the mean of 4 seeds; (f/b) are summed over seeds:

  | delta (q2) | X0 dev / dev2 | BLD l2=1e-3 trainC-CV / dev / dev2 | BLD l2=1e-2 trainC-CV / dev / dev2 |
  |---|---|---|---|
  | a4u (5-fold mean) | +0.04 (9/8) / +0.08 (8/7) | −0.15 / −0.12 (3/6) / +0.50 (9/3) | −0.17 / +0.13 (8/5) / +0.33 (7/3) |
  | a4u #f0 (single model) | +0.04 (9/8) / +0.25 (10/7) | −0.15 / −0.04 (5/6) / +0.33 (8/4) | −0.17 / −0.04 (5/6) / +0.33 (8/4) |
  | a4h | −0.25 (8/14) / +0.08 (12/11) | −0.22 / −0.12 (4/7) / +0.50 (9/3) | −0.27 / +0.00 (6/6) / +0.50 (7/1) |
  | a4h #f0 | −0.54 (3/16) / −0.08 (15/16) | −0.22 / +0.00 (4/4) / +0.50 (10/4) | −0.27 / +0.04 (7/6) / +0.42 (8/3) |
  | a8 | +0.00 (5/5) / +0.00 (7/7) | −0.03 / +0.00 (3/3) / +0.25 (6/3) | −0.12 / +0.00 (5/5) / +0.33 (6/2) |
  | a8 #f0 | +0.13 (10/7) / +0.25 (8/5) | −0.03 / −0.04 (4/5) / +0.42 (9/4) | −0.12 / +0.00 (6/6) / +0.42 (7/2) |

  The pattern is consistent across all 4 configs, single vs ensemble scores, 3 scales and 2 l2 values. dev is about 0 (best −0.12..+0.13),
  dev2 is +0.25..+0.50, and the trainC CV is negative in every BLD fit. No variant reaches +0.3 on dev. The trainC-selected
  variant (a8, BLD l2=1e-3, best trainC CV −0.03) gives dev +0.00 (3/3) and dev2 +0.25 (6/3).

## VERDICT (13:45Z): NEGATIVE. The pre-registered gate fails on the re-pick half. No in-stack build.
- AUROC half PASSES. The best trainC-selected model is attention pooling (4 heads over all body-token PCA-256 states at L14/18/22/27,
  plus token mean), blended with b using a weight chosen on trainC. Within-item AUROC vs b = sn+lg:

  | | trainC OOF | dev | dev2 |
  |---|---|---|---|
  | b = max(sn,−12)+max(lg,−12) (rung 4) | 0.6975 | 0.7525 | 0.6726 |
  | seq model alone (c4:x:a4u) | 0.689 | 0.770 | 0.693 |
  | b + seq blend (v=0.75) | 0.716 (+0.018) | 0.785 (+0.032) | 0.697 (+0.025) |

  This is the first verifier lane past the ~0.75/0.67 ceiling. The pooled signal is partly new (within-item corr with b is 0.65), and
  verifier-only best-of-N top1 rises by 2–4 pts (off/selm_q2.log).
- RE-PICK half FAILS: Δ vs rung 4 is ≈0 on dev (−0.12..+0.13) and +0.25..+0.50 on dev2 (pooled at best 15/8, p≈0.1), and the
  trainC CV is negative. The cause: once votes are counted (Σσ group score, the writer's feature), right-vs-wrong group ordering
  improves by only ~0.5 pt. The writer's errors are mostly vote-heavy wrong groups, and an AUROC +0.03 verifier does not flip them.
- What it would have been (weights): PCA mu/P per layer (out/pca.pt), 4 heads × (q, v) vectors on the 4×256 projections, the linear head
  and the blend weight. In-stack these fold into fixed 1024-vectors per layer for the online-softmax registers (design above). The
  model was trained on trainC only, and all selection was done on trainC OOF.
- Node data $WSM_DATA/implants/seqver/ (~22 GB of features) can be deleted. Possible resumption, not pre-registered: a writer
  that uses the seq score in the group feature differently would need a new gate (θ form change, trainC-selected).
