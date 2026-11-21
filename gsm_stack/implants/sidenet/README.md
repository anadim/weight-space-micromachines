# lane sidenet: small NON-LINEAR trained side-net verifier for the rung-3 final writer (todo sidenet-verifier)

Rung 3 = avf (implants/ladr3/aw; rows ladr3/aw/out/avf_{dev,dev2}: 90.29 / 87.33). Selection gap is the largest remaining bucket;
linear probes on attempt states plateau at within-item AUROC ~0.75 dev / ~0.70 dev2. Here: an MLP side-net on multi-layer
'####' + line-end-mean states, trained on trainC rung-3 rows only (pointwise / pairwise / listwise), evaluated held out on dev/dev2,
then offline re-pick on the avf rows with ClusterLin refit on trainC. Gate: >= +0.3 on both dev and dev2 offline AND within-item
AUROC gain >= +0.02 on both. Only if it passes: in-stack build on a copy (implants/sidenet/aw).

## log (UTC, newest last)
- 08:13Z rows copied to rows/ (avf_dev, avf_dev2 = j23407; tc_trainC = vfrefit j23422, seeds 23/29/31/37). Attempt tables:
  `python3.11 ../../atlas/parse.py --sets dev dev2 trainC --gen rows --out tables --notes tables/counts.md` -> attempts dev 18477,
  dev2 9804, trainC 47525 (parser vs register: answer 99.98/99.96/99.98%, natt/nseg/end 100%).
- 08:20Z job/extract_sn.py (from probe/extract.py): per attempt, context pre + q_ids + seg_tokens through host ckpt_a2s2 (bf16 sdpa);
  stores H = layer-input states (L10..27, 28 = final-norm input) at the '####' token, M = mean over model-written line ends
  (phrase_len <= off < hash_off); fp16 layer-major (19, n, 1024) under $WSM_DATA/implants/sidenet/{set}/. Validation = L18 . probe vs lg.
  `cd job && <queue-submit> $PWD 3600 job_sn_ext -- bash run.sh python3 extract_sn.py --sets dev,dev2,trainC` -> **j23439**.
- 08:21Z j23439 extraction: dev / dev2 validation L18('####') . probe vs register lg: corr 0.9982 / 0.9983, MAE 0.083 / 0.081,
  p99 0.27, max 0.44 / 0.42 (no |d| > 0.5) -> teacher-forced states = in-stack states up to bf16 noise (same as hs_extract 0.998).
  dev 170 s (23.8M tok), dev2 89 s; no-'####' attempts 125 / 33 (H = last token).
- 08:23Z job/train_sn.py (MLP/linear side-nets; losses bce / pair / list; trainC 5-fold item CV, dev/dev2 = 5-fold-model mean) and
  repick_sn.py (local: ClusterLin with side-net features, refit on trainC OOF scores; checked: B18 control reproduces vfrefit exactly,
  trainC CV 88.86, dev -0.08 (5/7), dev2 +0.67 (9/1)). Aux tables: `python3.11 mk_aux.py` -> job/tables/aux_*.npz.
  Submitted `cd job && <queue-submit> $PWD 3600 job_sn_tr1 -- bash run.sh bash train_job.sh g1 c4:HM:mlp:bce,c4:HM:mlp:pair,
  c4:HM:mlp:list,c4:HM:lin:bce,c4:HM:lin:pair,l18:H:lin:bce,c4:H:mlp:pair,c8:HM:mlp:pair,all:HM:mlp:pair,all:HM:lin:pair` -> **j23441**
  (waits for the trainC features).
- 08:29Z j23439 trainC done: corr 0.9983, MAE 0.082, max 0.50 (n 47365). Features under $WSM_DATA/implants/sidenet/{dev,dev2,trainC}/
  (H.npy, M.npy, meta.npz; 5.9 GB). j23441 (g1) DONE in 7 min -> job/out/train_g1.json, sc_g1.npz.
  Within-item AUROC (committed attempts, right vs wrong of the same item over all 4 seeds); baseline register lg: trainC 0.642,
  dev 0.727, dev2 0.643 (rung-3 attempts incl. the adaptive extra tries are harder to rank than the K=32 atlas sets).
  Side-net alone (epoch picked on trainC OOF; dev/dev2 = 5-fold mean; `python3.11 off/blend.py job/out/sc_g1.npz`):

  | cfg (layers:feats:model:loss) | trainC OOF | dev | dev2 | blend 0.5 z(sn)+0.5 z(lg): dev / dev2 |
  |---|---|---|---|---|
  | l18:H:lin:bce (lg-like control) | 0.624 | 0.722 | 0.629 | 0.732 / 0.640 |
  | c4:HM:lin:bce | 0.681 | 0.736 | 0.665 | 0.754 / 0.671 |
  | c4:HM:lin:pair | 0.702 | 0.727 | 0.660 | 0.753 / 0.673 |
  | c4:HM:mlp:bce | 0.675 | 0.722 | 0.663 | 0.754 / 0.674 |
  | c4:HM:mlp:pair | 0.673 | 0.713 | 0.655 | 0.752 / 0.671 |
  | c4:HM:mlp:list | 0.665 | 0.696 | 0.673 | 0.740 / 0.680 |
  | c4:H:mlp:pair | 0.642 | 0.702 | 0.597 | 0.739 / 0.628 |
  | c8:HM:mlp:pair | 0.676 | 0.716 | 0.644 | 0.757 / 0.668 |
  | all:HM:mlp:pair | 0.667 | 0.724 | 0.638 | 0.757 / 0.663 |
  | all:HM:lin:pair | 0.702 | 0.723 | 0.652 | 0.754 / 0.668 |

  MLP (hid 256, drop 0.3/0.1) is NOT better than linear on the same features; alone no model gains >= +0.02 on dev. Blended with lg
  the gain is ~+0.025 dev / +0.03 dev2 (MLP and linear alike).
- 08:40Z offline re-pick g1 (`python3.11 repick_sn.py --sc job/out/sc_g1.npz --cfgs ... --modes B18,AUG,REP,BLD,P:AUG --l2 1e-3,1e-2`,
  off/repick_g1.log). Modes: B18 = fresh refit without side-net (control), AUG = + group stats of sn (Σσ, mean/max/min), REP = sn in
  place of lg, BLD = lg+sn in place of lg, P:AUG = AUG ridge toward current θ. Δ vs the in-stream VF pick (avf rows), f/b over 4 seeds:
  - control B18 l2 1e-3: dev −0.08 (5/7), dev2 +0.67 (9/1)  [the dev2 gain is the refit itself, as in lane vfrefit]
  - best dev: c4:HM:lin:bce AUG l2 1e-2 +0.17 (15/11), dev2 +0.58 (13/6); BLD l2 1e-3 dev +0.17 (14/10), dev2 +0.92 (15/4), pooled 29/14 p=0.016
  - MLPs: dev −0.21 … +0.00 for every mode (mlp:pair AUG l2 1e-2 −0.08, 14/16; mlp:list AUG −0.08); REP (drop lg) −0.6 … −0.75 dev.
  -> no candidate reaches +0.3 on dev.
- 08:42Z (prev session, logged now) j23442 `extract_sn.py --sets dev,dev2,trainC --extra 1` DONE: A = last seg token state, E = last
  line-end state before '####' (A.npy/E.npy next to H/M). g2+g3 training submitted as **j23447** (job_sn_tr2, see `gq show j23447`):
  g2 grid c4:A:lin:bce,c4:E:lin:bce,c4:HMA:lin:bce,c4:HMAE:lin:bce,c4:HMAE:lin:pair,c4:HMAE:mlp:bce,c4:HMAE:mlpr:bce,c4:HMA:mlpr:pair,
  c8:HMAE:lin:bce; g3 = `--hid 128 --drop 0.5 --idrop 0.2 --wd 0.1 --epochs 2,3,4,6,8` grid c4:HM:mlp:bce,c4:HM:mlpr:bce,
  c4:HMAE:mlp:bce,c4:HMAE:mlpr:pair,all:HMA:mlp:bce (mlpr = MLP + linear skip).
- 08:50Z session resumed (todo sidenet-fin). Note from repick_g1.log: the trainC-CV-best g1 entry is c4:HM:lin:pair BLD l2 1e-3
  (trainC CV +0.20, 56/44) -> dev +0.21 (19/14), dev2 +1.17 (17/3), pooled 36/17 p=0.006: still < +0.3 on dev.
- 08:57Z j23447 DONE (55 s; out/train_g2.json, sc_g2.npz, train_g3.json, sc_g3.npz). Within-item AUROC (`python3.11 off/blend.py
  job/out/sc_g{2,3}.npz` -> off/blend_g{2,3}.log); lg baseline trainC 0.642 / dev 0.727 / dev2 0.643. A = last seg token, E = last line
  end before '####'. g3 = smaller/stronger-regularised MLPs (hid 128, drop 0.5, idrop 0.2, wd 0.1, epochs 2..8).

  | cfg | trainC OOF | dev | dev2 | blend w(trainC) dev / dev2 | blend 0.5 dev / dev2 |
  |---|---|---|---|---|---|
  | g2 c4:A:lin:bce | 0.623 | 0.698 | 0.613 | 0.737 / 0.645 (w .3) | 0.734 / 0.640 |
  | g2 c4:E:lin:bce | 0.648 | 0.716 | 0.689 | 0.739 / 0.681 (w .6) | 0.742 / 0.676 |
  | g2 c4:HMA:lin:bce | 0.683 | 0.734 | 0.661 | 0.742 / 0.665 (w .9) | 0.755 / 0.668 |
  | g2 c4:HMAE:lin:bce | 0.686 | 0.743 | 0.684 | 0.750 / 0.687 (w .9) | 0.759 / 0.680 |
  | g2 c4:HMAE:lin:pair | 0.701 | 0.735 | 0.670 | 0.752 / 0.678 (w .8) | 0.760 / 0.677 |
  | g2 c4:HMAE:mlp:bce | 0.669 | 0.734 | 0.663 | 0.760 / 0.676 (w .7) | 0.761 / 0.675 |
  | g2 c4:HMAE:mlpr:bce | 0.679 | 0.734 | 0.672 | 0.751 / 0.682 (w .8) | 0.756 / 0.676 |
  | g2 c4:HMA:mlpr:pair | 0.675 | 0.733 | 0.637 | 0.760 / 0.659 (w .7) | 0.763 / 0.664 |
  | g2 c8:HMAE:lin:bce | 0.691 | 0.745 | 0.675 | 0.753 / 0.680 (w .9) | 0.761 / 0.678 |
  | g3 c4:HM:mlp:bce | 0.668 | 0.731 | 0.668 | 0.746 / 0.677 (w .8) | 0.757 / 0.673 |
  | g3 c4:HM:mlpr:bce | 0.674 | 0.735 | 0.665 | 0.748 / 0.675 (w .8) | 0.757 / 0.672 |
  | g3 c4:HMAE:mlp:bce | 0.663 | 0.737 | 0.665 | 0.750 / 0.676 (w .8) | 0.753 / 0.671 |
  | g3 c4:HMAE:mlpr:pair | 0.678 | 0.737 | 0.669 | 0.755 / 0.679 (w .7) | 0.756 / 0.676 |
  | g3 all:HMA:mlp:bce | 0.654 | 0.730 | 0.632 | 0.753 / 0.659 (w .7) | 0.754 / 0.660 |

  Adding A/E helps a little (best alone: c4:HMAE:lin:bce +0.016 dev / +0.041 dev2; c8:HMAE:lin:bce +0.018 / +0.032). Alone, still
  < +0.02 on dev for every config. Blended with lg, the AUROC gate (>= +0.02 both) is met by most configs, e.g. c4:HMAE:lin:bce w .9
  +0.023 / +0.044. Non-linear (mlp/mlpr, regularised or not) is again no better than linear on the same features.
- 09:02Z offline re-pick g2/g3 (local, 3.3 min each):
  `python3.11 repick_sn.py --sc job/out/sc_g2.npz --modes B18,AUG,BLD,P:AUG --l2 1e-3,1e-2 > off/repick_g2.log`;
  same for sc_g3.npz (modes AUG,BLD,P:AUG) -> off/repick_g3.log. Control B18 (fresh refit, no side-net) is unchanged:
  dev −0.08 (5/7), dev2 +0.67 (9/1). Highlights (Δ vs in-stream VF pick, f/b over 4 seeds):
  - g2 c4:HMAE:lin:pair P:AUG l2 1e-3: dev +0.17 (15/11), dev2 +1.00 (14/2), pooled 29/13 p=0.010
  - g2 c8:HMAE:lin:bce BLD l2 1e-2: dev +0.12 (15/12), dev2 +1.08 (18/5), p=0.016; BLD 1e-3: dev +0.17 (16/12), dev2 +0.83
  - g3 c4:HMAE:mlpr:pair P:AUG l2 1e-2: dev +0.25 (12/6), dev2 +0.75 (12/3) = best dev of all 114 entries (dev-selected, so optimistic)
  - g3 c4:HMAE:mlp:bce P:AUG l2 1e-3: dev +0.12 (9/6), dev2 +1.00 (15/3)
- 09:04Z trainC-only selection (no dev tuning) over g1+g2+g3, modes AUG/BLD/P:AUG x l2 {1e-3,1e-2} = 114 entries, ranked by trainC
  CV gain (off/select_trainC.log):
  | rank | entry | trainC CV | dev | dev2 | pooled |
  |---|---|---|---|---|---|
  | 1 | g1 c4:HM:lin:pair BLD 1e-3 | +0.20 (56/44) | +0.21 (19/14) | +1.17 (17/3) | 36/17 p=0.006 |
  | 2 | g1 c4:HM:mlp:pair AUG 1e-2 | +0.17 | −0.08 (14/16) | +0.67 (13/5) | 27/21 |
  | 3 | g2 c4:HMAE:lin:pair P:AUG 1e-3 | +0.13 | +0.17 (15/11) | +1.00 (14/2) | 29/13 p=0.010 |
  | 4 | g2 c4:HMAE:lin:pair AUG 1e-3 | +0.12 | +0.12 (16/13) | +0.75 (14/5) | 30/18 |
  Max dev Δ over all 114 entries = +0.25; entries with ≥ +0.3 on both sets = 0. Mean over all entries: dev −0.04, dev2 +0.67 —
  i.e. on average the side-net adds ~nothing over the plain refit control (B18: −0.08 / +0.67); the dev2 gain is the refit.

## VERDICT (09:05Z): gate FAILED -> no in-stack build
- AUROC part of the gate: side-net alone never reaches +0.02 within-item AUROC on dev (best +0.018, c8:HMAE:lin:bce); blended with
  lg it does (e.g. c4:HMAE:lin:bce w .9: +0.023 dev / +0.044 dev2).
- Offline part: no config/mode/l2 reaches +0.3 on dev (best trainC-selected +0.21, best dev-selected +0.25); dev2 gains (+0.7…+1.2)
  are mostly the ClusterLin refit itself (control +0.67). The trainC-selected entry (g1 c4:HM:lin:pair BLD 1e-3) is a consistent but
  small positive (pooled 36/17, p=0.006) — about +0.2 / +0.5 beyond the refit control; below the gate and not worth a
  multi-layer-state register build (4 layers x 2-4 states x 1024 accumulators per attempt slot).
- Non-linearity does not help: MLP / wide-and-deep (mlpr) / regularised MLPs <= linear on identical features in every comparison
  (AUROC and re-pick). The verifier ceiling here is information in the states, not the readout's capacity.
- If revisited: the cheap part worth keeping is the *linear* multi-layer blend (c4:HM or c4:HMAE, pairwise loss) as an extra per-slot
  register feature of the writer — a linear map, not a side-net — and only together with the ClusterLin refit (lane vfrefit), since
  the two gains overlap on dev2. Expected ≈ +0.2 dev / +1.0 dev2 offline; it would need a stronger signal (e.g. more trainC seeds,
  or per-step states) to clear +0.3 on dev.

## Orchestrator override (after the verdict above): BUILD g1 c4:HM:lin:pair BLD l2 1e-3 in-stack as a rung-4 candidate
Reason given: trainC-CV selected without dev tuning, offline +0.21 (19/14) dev / +1.17 (17/3) dev2, pooled 36/17 p=0.006 (stronger
than rung 3's 65/43 p=0.021), zero extra tokens; ladder rule = 4-seed confirmation, all weights. (The +0.3-both gate above was a
heuristic of the previous session.)
- 09:0xZ lane tree: `rsync -a --exclude out gsm_stack/implants/ladr3/aw/ gsm_stack/implants/sidenet/aw/`.
- Side-net weights = ONE fixed linear vector: the 5-fold MEAN of the trainC fold models (each fold model is linear on standardized
  features, so the mean folds into w = mean_k(wz_k)/sd, b = mean_k(bz_k) − w·mu). Chosen over a full-trainC refit because it is
  exactly the scorer whose dev/dev2 scores were evaluated offline (re-pick numbers above); trainC itself uses OOF scores for θ.
  `job/train_sn.py --save_w 1` writes out/w_TAG_CFG.pt (w: 8192 = H over layers 14,18,22,27 then M over the same, b, check =
  max |folded − fold-mean| on dev/dev2). GPU submit via gsub sat behind the cap -> ran on CPU instead (job/train_cpu.sh,
  `gq submit --gpu none --timeout 1800 --name job_cpu_sn_w -- bash train_cpu.sh python3 train_sn.py --tag g4 --save_w 1 --grid c4:HM:lin:pair`, j23452).
- In-stack (aw/stk, env `AW_SN=path/sn_w.pt`, requires AW_VF=1; unset = code path identical to ladr3):
  * aw_sn.py: loader (per layer a fixed (1024,2) [wH, wM] matrix), `vf_vote_sn` = aw_vf.vf_vote with per-slot Fv[0] = blend
    b = max(sn,−12) + max(lg,−12) and group Σsigmoid(b/2) (offline mode BLD), θ in vf_vote order (`to_vf_order`).
  * comp_serial.py (diff aw/aw_sn_comp_serial.diff): in the layer loop, right after layer i's first add_rms (x = layer-i input =
    HF hidden_states[i]) for i in {14,18,22,27}: u += x·[wH_i, wM_i]. Registers per row: s_snH (u_H at the token completing '####',
    as the L18 probe s_lg), s_snM/s_sncnt (sum/count of u_M at host-read tokens whose text contains '\n', segment before '####',
    gate computed in _sw: rd & NL[tok] & ~s_lgset & ~cap_probe & ~end — writer-phrase tokens are not host-read, = offline
    phrase_len <= off < hash_off; checked on the tables: inside that window only wf 0/1 tokens occur, calc tokens are host-read),
    reset at newseg; at END: R_sn[slot] = s_snH + s_snM/max(cnt,1) + b (with R_lg). Final writer: vf_vote_sn.
  * fastroll.py: harvest also records R_sn; aw_hook.py: rows get `sn` (per slot), file tag suffix `_sn`.
- aw/test_sn.py (CPU fuzz, compiled writer vs offline repick_sn St+pick BLD): random θ, 20000 trials: 0 mismatches, 0 near ties skipped.
- 09:15Z (session rung4) theta consistency: vfsn_BLD_0.001.json was saved by the g4 re-pick (off/repick_g4.log, `--sc job/out/sc_g4.npz
  --save c4:HM:lin:pair@BLD@0.001`; g1 log has no save) -> theta is fit on trainC OOF scores of the SAME (g4) weights. g4 vs g1 scores:
  corr 0.9991 trainC / 0.9998 dev (retrain noise); g4 BLD 1e-3: trainC CV +0.22 (56/43), dev +0.21 (20/15), dev2 +1.08 (17/4), pooled 37/19 p=0.011.
  `python3.11 mk_snw.py job/out/w_g4_c4_HM_lin_pair.pt vfsn_BLD_0.001.json` -> aw/stk/sn_w.pt (|w| 2.46, b -17.16, ep 16, fold check 5.7e-6).
  `python3.11 aw/test_sn.py` (real theta): 20000 trials, 0 mismatches, 0 near ties -> PASS.
- 09:15Z smoke `cd aw && <queue-submit> $PWD 2400 job_lad4_q -- bash lad4_job.sh off,snq dev` -> **j23454** (waited ~8 min on the cap).
- 09:33Z smoke j23454 DONE (5 min run): `off` dev s7 90.50, `snq` dev s7 90.67 (tok/q 1347.6 both).
  * `python3.11 aw/eq_cmp.py ../ladr3/aw/out/avf_dev aw/out/off_dev aw/out/snq_dev`: off = 600/600 token-identical to avf (gate-off
    exact); snq: 585/600 token-identical, the other 15 differ only in the final '####' writer line (sn_check [1]: 600/600 identical
    before the final line, natt identical); 6 fixes / 5 breaks vs avf s7.
  * `python3.11 aw/sn_check.py aw/out/snq_dev ../ladr3/aw/out/avf_dev dev job/out/sc_g4.npz c4:HM:lin:pair`: in-stack R_sn vs offline
    (teacher-forced extract + folded side-net) n=4417 committed attempts: corr 0.99883, MAE 0.080, p99 0.257, max 1.35 (1 attempt > 0.5,
    16 > 0.3; same bf16 noise class as the L18 probe extract check corr 0.998 / max 0.44, larger tail because 4 layers incl. L27).
    Pick vs the offline BLD writer on the row's own registers: 0/600 mismatches; pick vs the offline re-pick prediction (offline
    side-net scores on the avf rows): **0/600 pick changes**.
- 09:35Z 4-seed runs: `cd aw && <queue-submit> $PWD 3600 job_lad4_sn -- bash lad4_job.sh sn dev` -> **j23458**;
  `... job_lad4_sn2 -- bash lad4_job.sh sn dev2` -> **j23459** (rows -> aw/out/sn_dev, aw/out/sn_dev2).
- 09:50Z 4-seed run v0 (j23458 dev, j23459 dev2; rows archived as aw/out/sn0_{dev,dev2}): dev 90.67/90.67/90.17/90.33 = 90.46 (+0.17,
  f/b 19/15), dev2 89.00/87.67/88.33/88.33 = 88.33 (+1.00, 16/4), pooled 35/19 p=0.020; 100% of rows identical to avf before the
  final line. sn_check dev: R_sn corr 0.99872, MAE 0.080, max **5.48**; dev2 max 0.42; picks vs offline prediction 1/2400, 2/1200.
  BUG found from the dev max (seed 13 idx 254 k 12, 2nd outlier seed 7 idx 39 k 5): when an attempt ENDs on the token completing
  '####' (C=400 cap) and a repair/next attempt follows, the newseg reset happens in _sw BEFORE the layer loop, which then re-sets
  s_lgset/s_snH from that stale cap_probe -> the M line-end gate (~s_lgset) stayed closed for the whole next segment (M term = 0).
  (s_lgset/s_lg are base rung-3 behaviour and are left untouched; only lg's '####' value is overwritten later, so lg is unaffected.)
  FIX (SN-only): own flag s_snhs ('####' seen in this segment) for the M gate, reset at newseg; post-layer H capture uses
  cp = cap_probe & ~newseg(this step). Diff regenerated (aw/aw_sn_comp_serial.diff). Rerun: same commands -> **j23466** (dev),
  **j23467** (dev2), rows -> aw/out/sn_{dev,dev2}.
- 10:00Z rerun with the fix DONE: j23466 (dev, 8 min) / j23467 (dev2, 5 min). Rows aw/out/sn_{dev,dev2} are token-identical to the
  v0 rows (eq_cmp sn vs sn0: 600/600 per seed). The bug touched only 2/18336 dev R_sn values and no pick.
  `python3 ../ladr3/aw/summ.py ../ladr3/aw/out/avf aw/out/sn`; `python3.11 aw/sn_check.py aw/out/sn_{S} ../ladr3/aw/out/avf_{S} {S}
  job/out/sc_g4.npz c4:HM:lin:pair`; `python3.11 aw/sn_tail.py {S} m` (largest |ΔR_sn| and margins of the pick mismatches).

## RUNG-4 RESULT (10:00Z): g4 c4:HM:lin:pair BLD l2 1e-3 in-stack (AW_SN), 4 seeds 7/11/13/17, vs rung 3 (avf rows, same seeds)

| set | rung 3 (avf) per seed | rung 4 (sn) per seed | mean (Δ) | fixes/breaks | tok/q r3 -> r4 | natt |
|---|---|---|---|---|---|---|
| dev | 90.50/90.33/89.83/90.50 = 90.29 | 90.67/90.67/90.17/90.33 | **90.46 (+0.17)** | 19/15 | 1401.3 -> 1401.3 | 7.70 |
| dev2 | 87.67/87.67/87.00/87.00 = 87.33 | 89.00/87.67/88.33/88.33 | **88.33 (+1.00)** | 16/4 | 1522.4 -> 1522.4 | 8.17 |
| pooled | | | | **35/19, one-sided sign p = 0.020** | | |

Per seed f/b: dev 6/5, 4/2, 6/4, 3/4; dev2 5/1, 2/2, 5/1, 4/0. Offline prediction (g4 re-pick): dev +0.21 (20/15), dev2 +1.08 (17/4).
In-stack lost one fix per set to near ties (see below).
- Token identity: rows identical to rung 3 before the final '####' writer line = 2400/2400 dev and 1200/1200 dev2, with the same
  natt. Fully token-identical rows: 2352 / 1175. The other 48 / 25 rows differ only in the written answer (a different pick).
  tok/q is unchanged except dev2 s11 1481.1 -> 1481.0, a shorter written answer string. There is no other divergence: the side-net
  registers do not write into the residual stream, and the A/S/B/C decisions still read the base voter registers.
- Exactness vs offline, over committed attempts with '####' (dev n=18336, dev2 n=9748):

  | set | corr | MAE | p99 | max |
  |---|---|---|---|---|
  | dev | 0.99892 | 0.079 | 0.257 | 0.48 |
  | dev2 | 0.99894 | 0.079 | 0.259 | 0.42 |

  No attempt has a diff above 0.5 after the fix. Before the fix, dev max was 5.48.
  Picks vs the offline BLD writer on the row's own registers: 0/2400 and 0/1200 mismatches, so the writer is exact.
  Picks vs the offline re-pick prediction (offline side-net scores): 1/2400 dev and 2/1200 dev2. All three are near ties: in-stack
  group-score margin 0.014 / 0.008 / 0.042, offline margin 0.092 / 0.002 / 0.011. These flips come from the bf16 R_sn noise
  (MAE 0.08); 2 of them cost the in-stack run a fix that the offline prediction had (dev s11 idx 292, dev2 s11 idx 147).
- WHAT IS WEIGHTS: everything in rung 3, plus:
  * fixed linear vectors on the layer-input residuals of layers 14, 18, 22 and 27. Per layer this is a (1024, 2) [wH, wM] matrix
    (8192 weights + bias): the 5-fold mean of the trainC c4:HM:lin:pair side-net, with the standardisation folded in;
  * per-row registers:
    - s_snH: the H projection at the token completing '####';
    - s_snM / s_sncnt: the sum and count of M projections at host-read '\n' tokens before '####'. The gate uses the fixed
      token table NL and the flag s_snhs. All of these reset at newseg;
    - the slot register R_sn = s_snH + s_snM/max(cnt,1) + b, committed at END;
  * the final writer (ClusterLin over answer groups, fixed 18-dim θ + c0) with the BLD feature. θ was fit on trainC out-of-fold
    scores of the same weights.
  No host-side Python decides anything per row: every gate, counter, commit and the argmax writer is a batched tensor op inside the
  compiled decode step.
- VERDICT vs the rung criterion: the criterion is met. Mean gain is > 0 on both sets (+0.17 dev, +1.00 dev2) and pooled fixes exceed
  breaks (35 > 19, p = 0.020). Caveat: the dev gain is small (+0.17, 19/15); almost all of the signal is dev2. The dev2 gain partly
  overlaps with the plain ClusterLin refit (offline control B18: dev −0.08, dev2 +0.67). The orchestrator decides.
  Drop-in dir: gsm_stack/implants/rungs/rung4/ (rung-3 tree + aw_sn.py, sn_w.pt, patched comp_serial/fastroll/aw_hook,
  lad4_job.sh, test_sn.py, sn_check.py, README with the exact env).
