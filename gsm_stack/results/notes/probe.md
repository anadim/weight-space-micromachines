# Lane 2: predictability of right vs wrong (probe lane)

Data: 32 attempts per item (dev 600 items, dev2 300 items), plus split C (trainC = 1499 items × 16 attempts, used for training only).
Hidden states come from job_hsC (j23158) and the earlier dev/dev2 extractions.
Unless noted, probes are trained on trainC and tested on dev/dev2, which are fully held out.

Metric names used throughout:
- **pooled** = AUROC over all attempts.
- **within** = AUROC over (right, wrong) attempt pairs of the same mixed item, which removes item difficulty.
- Mixed items: trainC 635, dev 298, dev2 154.

## Headline

| question | best signal | dev | dev2 | baseline |
|---|---|---|---|---|
| final answer right? ('####' verifier) | hash_mnl_c4 + lg blend | pooled 0.854, within 0.773 | pooled 0.876, within 0.700 | current lg register: 0.817/0.761 (dev), 0.834/0.680 (dev2) |
| is this line the first divergent step D? | step detector, layers 18+22, h+Δh | 0.822 / 0.770 | 0.830 / 0.771 | text −logp of the line 0.63 / 0.59 |
| same, eq-lines-only control | step detector | 0.765 / 0.703 | 0.768 / 0.718 | text 0.63–0.65 / 0.58–0.60 |
| will it go wrong, at the line end *before* D? | anticipation probe, within item, same step | 0.694 | 0.682 | generic value probe 0.57–0.60 |
| does the generic value-probe logit drop at D? | Δlogit across the D line | 0.49–0.55 | 0.51 | — (no) |
| text branch token (sampled p_w < 0.1) | min-p proxy | 23.6% of branches | 21.3% | right attempts with any p<0.1 token: 58.7% / 60.7% |
| item difficulty at question end | ridge on c3 (layers 14/18/22) | p_right R² 0.286 | 0.295 | question length alone: 0.116 / 0.071 |

Answers in one line each:
- **Hidden state:** mostly yes. It knows item difficulty from the start. A step error is clearly readable *at and after* the error line, at roughly 0.77 within-item AUROC. Separation is moderate *before* the error (0.68), and the generic value probe misses the error entirely.
- **Text:** weakly. Low-probability branch tokens explain only about 20% of divergences, and low-probability tokens are almost as common in right attempts.
- **Question:** partially. About 29% of p_right variance is predictable at question end, and "always" items are separable with AUROC 0.81.

## 1. '####' verifiers (Q3): trainC-trained, held out

Cell format: pooled / within-item AUROC.

| verifier | dev | dev2 |
|---|---|---|
| lg (current L18 register, from rollouts) | 0.817 / 0.761 | 0.834 / 0.680 |
| L18 | 0.837 / 0.752 | 0.868 / 0.678 |
| L22 | 0.837 / 0.737 | 0.849 / 0.657 |
| c4 (layers 14, 18, 22, 27) | 0.836 / 0.748 | 0.862 / 0.668 |
| c6 | 0.838 / 0.754 | 0.868 / 0.678 |
| call (all 19 layers, 10–28) | 0.845 / 0.750 | 0.867 / 0.666 |
| stack (per-layer stack) | 0.845 / 0.754 | 0.865 / 0.672 |
| mnl_c4 (mean of line-end states only) | 0.840 / 0.693 | 0.868 / 0.688 |
| w_c4 (within-item-demeaned training) | 0.733 / 0.750 | 0.735 / 0.680 |
| **hash_mnl_c4** ('####' state + mean line-end state, c4) | **0.854** / 0.752 | **0.876** / 0.698 |
| **blend 0.3·z(lg) + 0.7·z(hash_mnl_c4)** | **0.854 / 0.773** | **0.876 / 0.700** |
| blend 0.5 / 0.5 | 0.849 / 0.779 | 0.871 / 0.700 |

Other results:
- Training on dev instead (cvdev, tested on dev2) gives hash_mnl_c4 0.872/0.697 vs lg 0.834/0.680. Training on trainC (2.5× more items, about 24k vs 19k attempts) moves dev2 only slightly (+0.004 pooled). More data of the same kind will not buy much.
- stack_lg is miscalibrated because of the mismatch between OOF training and the full fit. Do not use it.
- Within-item gains over lg are small: hash_mnl_c4 alone gives −0.01 (dev) and +0.02 (dev2), and the blend gives +0.01 (dev) and +0.02 (dev2). Most of the gain is pooled, i.e. cross-item calibration. That matters for weighted voting across items with one confident wrong cluster, but not for ranking attempts inside one item.

## 2. Value probe along the trajectory (Q1)

These are logits at reasoning line ends (the '\n' token), using the trainC-trained L18 probe.

| relative position in attempt | 0–10% | 30% | 60% | 90% | at '####' |
|---|---|---|---|---|---|
| within-item AUROC, dev | 0.52 | 0.53 | 0.56 | 0.63 | 0.73 |
| within-item AUROC, dev2 | 0.47 | 0.50 | 0.56 | 0.63 | 0.67 |

- Pooled AUROC is already 0.72–0.74 in the first decile. The probe tracks item difficulty early, but separates right from wrong attempts of the same item only near the end.
- All line ends, by probe (pooled / within-item step-matched):

  | probe | dev | dev2 |
  |---|---|---|
  | L18 | 0.789 / 0.574 | 0.810 / 0.567 |
  | c4 | 0.776 / 0.572 | 0.814 / 0.570 |
  | w18 (dev) / w22 (dev2), within-trained | 0.699 / 0.589 | 0.704 / 0.563 |
  | text −logp baseline | 0.658 / 0.537 | 0.669 / 0.535 |

- j23159 (within-item demeaned value probe, cvdev) gave no gain: on dev2, w18 scores pooled 0.654 / within 0.563 vs raw L18 at 0.57 within.

## 3. Predictability at the divergence point D

D = first divergent step of a wrong attempt vs the closest right attempt (atlas/diverge.py).
Coverage of wrong attempts in mixed items:
- dev: 3105 wrong → 2576 usable. Excluded: 473 with no D, 132 with D in the first line, 55 with D in the answer line.
- dev2: 1806 wrong → 1597 usable.
- trainC: 3208 labels (3180 usable), computed for this lane in probe/tables/.

### 3a. Hidden state *after* the error: yes, if you train for it

**Generic value probe:** the logit does not drop across the D line.
- AUROC of −Δlogit (D transition vs other transitions): 0.49–0.56 (c4: dev 0.546, dev2 0.513).
- P(Δ<0) is 0.52–0.57 at D vs 0.49–0.51 elsewhere.
- Conclusion: "Am I on track to the right answer" is not updated at the error.

**Dedicated step-error detector** (divprobe.py):
- Logistic regression on line-end states at layers 18+22.
- Features: h = state; Δh = h − previous line end; hdh = [h, Δh].
- Labels: D line = 1. Negatives are other lines of mixed-item attempts: right attempts plus pre-D lines of wrong attempts. Post-D lines are excluded.
- Trained on trainC (3056 positives in 47,950 lines).
- Cell format: test AUROC / within-item matched AUROC (D line vs right attempts of the same item at the same step).

| features | trainC OOF | dev | dev2 |
|---|---|---|---|
| h, all lines | 0.822 / 0.744 | 0.821 / 0.768 | 0.819 / 0.746 |
| Δh | 0.782 / 0.715 | 0.755 / 0.724 | 0.751 / 0.704 |
| **hdh** | 0.829 / 0.757 | **0.822 / 0.770** | **0.830 / 0.771** |
| h, eq-only lines (control) | 0.741 / 0.671 | 0.763 / 0.700 | 0.749 / 0.681 |
| hdh, eq-only lines (control) | 0.752 / 0.679 | 0.765 / 0.703 | 0.768 / 0.718 |
| text: line min/mean −logp | — | — | 0.63 / 0.61 pooled, 0.58–0.60 within |

- Eq-only = only lines containing a computed '… = v'. This controls for "D lines are where the computation happens".
- A dev-trained detector (tested on dev2) is a bit weaker: 0.805/0.717 for h and 0.804/0.719 for hdh.

By divergence category (hdh, test AUROC):

| set | arithmetic | dropped | relation | operand | plan | unit | early_stop |
|---|---|---|---|---|---|---|---|
| dev | 0.86 | 0.85 | 0.84 | 0.80 | 0.69 | 0.69 | 0.90 |
| dev2 | 0.95 | 0.87 | 0.84 | 0.81 | 0.77 | 0.77 | 0.73 |

Arithmetic and dropped-quantity errors are the most visible. Plan and unit errors are the least visible.

**Operating points** (hdh, all lines, dev2; threshold set at the trainC-OOF line FPR):
- "Right attempts flagged" = right attempts of mixed items with at least one flag.
- "Wrong flagged before D" = false early flags on wrong attempts.

| train line FPR | test line FPR | D-line TPR | right attempts flagged | wrong flagged before D |
|---|---|---|---|---|
| 1% | 1.2% | 11.1% | 6.9% | 2.7% |
| 2% | 2.2% | 17.6% | 11.4% | 5.3% |
| 5% | 5.3% | 32.4% | 26.0% | 12.5% |
| 10% | 11.0% | 47.8% | 47.3% | 25.4% |

- dev has similar points: 2% → TPR 19.5%, right attempts flagged 8.7%; 5% → 32.0% / 21.4%.
- The eq-only hdh detector is cleaner per attempt on dev2: 2% → 12.1% / 7.0%; 5% → 24.6% / 15.0%.
- Each attempt has about 5–6 lines, so per-line FPR compounds. A hard abort trigger catches at most about 1/5 of errors at about 10% collateral.

### 3b. Hidden state *before* the error: moderate

At the line end just before the D line, wrong attempts are compared with right attempts of the same item at the same step index.

Generic value probe, by line-end position rs relative to D (0 = line end before D, +1 = end of the D line):

| probe, set | rs −2 | rs −1 | rs 0 | rs +1 | rs +2 | rs +3 |
|---|---|---|---|---|---|---|
| c4, dev | 0.507 | 0.558 | 0.568 | 0.636 | 0.615 | 0.614 |
| c4, dev2 | 0.534 | 0.535 | 0.602 | 0.628 | 0.614 | 0.616 |
| wc4, dev | — | — | 0.602 | 0.645 | — | — |

Early_stop items are already separable at rs 0 (0.83–0.87). The model "knows" it is about to stop early.

Dedicated anticipation probe (trained with item×step demeaning, layers 18+22, within-item matched AUROC):

| features | trainC OOF | dev | dev2 |
|---|---|---|---|
| h | 0.674 | 0.694 | 0.682 |
| Δh | 0.626 | 0.660 | 0.672 |

There is real but modest pre-error signal. It is weaker than the post-error signal (≈0.77), so most of the information appears *in* the error line, not before it.

### 3c. Text: branch-token probability (min-p proxy)

The branch token is the first token where the wrong attempt departs from its closest right attempt (atlas). p_w is the sampled probability of the wrong branch token; p_r is that of the right branch. p_max is not stored, so p_w < thr is used as a proxy for a min-p floor (which is relative to p_max).

| dev (dev2) | p<0.05 | p<0.1 | p<0.2 |
|---|---|---|---|
| branch tokens with p_w < thr (n=2729) | 14.4% (12.9%) | 23.6% (21.3%) | 37.5% (36.5%) |
| … and p_r ≥ thr (block removes only the wrong branch) | 14.0% | 23.0% | 35.5% |
| critical branches (inside the D line, n=305) | 9.8% (11.0%) | 17.7% (17.9%) | 28.9% (34.5%) |
| right branch itself has p_r < thr | 8.2% | 14.2% | 25.8% |
| wrong attempts (mixed items) with any sampled p<thr token | 53.8% (54.2%) | 70.6% (72.9%) | 87.1% (89.4%) |
| same, pre-D prefix only | 33.4% | 48.2% | 65.9% |
| right attempts (mixed items) with any p<thr token | 40.0% (41.0%) | 58.7% (60.7%) | 79.1% (80.4%) |
| right attempts (all items) with any p<thr token | 29.4% | 45.7% | 65.7% |

- At most 10–18% of critical branches are low-probability samples. Most wrong branches are confident (p_w ≥ 0.2 in about 63–71%).
- Low-probability tokens are only slightly enriched in wrong attempts (54% vs 40% at p<0.05).
- A min-p floor would cut tail samples in roughly as many right attempts as wrong ones. It is not a targeted fix. The text-logp D-line detector (0.63 / 0.59 within) confirms this.

## 4. Question-end difficulty

Ridge regression on the question-end hidden state (last question token), trainC-trained.

| target | feature | trainC OOF | dev | dev2 |
|---|---|---|---|---|
| p_right R² | c3 (layers 14/18/22) | 0.264 | 0.286 | 0.295 |
| p_right R² | L16 | 0.245 | 0.272 | 0.278 |
| p_right R² | question length | — | 0.116 | 0.071 |

Class AUROC with L18 (dev / dev2):
- always vs rest: 0.811 / 0.807 (question length alone: 0.67)
- mixed vs rest: 0.777 / 0.790
- never vs rest: 0.690 / 0.704
- p_right ≥ 0.5: 0.768 / 0.794

## 5. Q4: value-gated early restart (offline simulation)

Abort an attempt when the line-end value probe falls below a threshold, and reuse the saved tokens for more attempts at an equal token budget.

On trainC selection, gating is worse than the ungated control at every threshold. For example, with layer 22, patience 5 and abort quantile 0.01–0.2, accuracy drops from 0.860 (ctrl) to 0.859 / 0.852 / 0.832 / 0.801 / 0.715. Aborts kill right attempts faster than the extra votes recover them. This matches §2 (no within-item signal early). The earlier held-out run (out/ana_cvdev.log, with the gating rule chosen on its training split) shows the same pattern. For the chosen rule, accuracy is 0.879 gated vs 0.880 plain on dev and 0.851 vs 0.855 on dev2. With an abort quantile of 0.05, gating drops accuracy to 0.855–0.863 (dev) and 0.823–0.827 (dev2).

The final dev/dev2 sim numbers are in probe/out/ana_trainC.json['q4'] once the background run finishes (out/ana_trainC_sim.log ends with ANA_DONE).

## 6. Implications for implants

1. **Verifier v2 (most solid, do first).** Replace the lg register with hash_mnl_c4, or the 0.3/0.7 blend with lg.
   - Pooled AUROC rises by 0.04 on both sets. Within-item rises by +0.01 (dev) and +0.02 (dev2).
   - Use it as the voter weight. It needs the per-attempt running mean of line-end states at layers 14/18/22/27, which is cheap to accumulate at each '\n' during generation.
2. **Step-error detector at line ends → voter feature, not a hard abort.**
   - At the '\n' that ends a line, the D detector (hdh, layers 18+22) flags the erroneous line with about 0.77 within-item AUROC.
   - As a hard abort or re-read trigger it is weak: about 18% of errors caught at about 11% right-attempt collateral.
   - Better uses:
     - (a) Down-weight attempts in the vote by their max D-score. This is still to be tested; it should add to the verifier, because it localizes errors the final '####' state already partly encodes.
     - (b) A *local* repair: on a flag, rewind only the flagged line and resample it, or inject a re-read ("Let me check this step"). This costs only one line, so collateral on right attempts is cheap. The atlas shows arithmetic, dropped and relation errors dominate, and those are exactly the categories the detector sees best (0.84–0.95).
   - The generic value probe must not be used for this. It is blind at D (AUROC ≈ 0.5).
3. **Early restart / value gating:** negative (Q4). The value probe has no within-item signal early in an attempt.
4. **Sampling-level fixes (min-p / temperature):** not targeted. At most 10–18% of critical branches are low-probability, and right attempts contain such tokens at nearly the same rate.
5. **Question-end difficulty:** usable for *allocation*, e.g. stop early or spend fewer attempts on predicted-always items (AUROC 0.81), or enable the implant / voter only on predicted-mixed items (0.78). It cannot pick out individual errors.

## 7. Verifier v2 spec (for the voter implant)

**Files:**
- probe/out/pred_trainC/probes.pt: torch dict, fp16 w in input units, float b.
- probe/out/pred_trainC/verifier_v2.npz: float32 numpy arrays `{name}__w`, `{name}__b`.
  - Contains L18, L22, c4, call, mnl_c4, hash_mnl_c4, stack_lg, w_c4.
  - Exported by probe/probes_dump.py (CPU gq job).

**Layer convention:**
- `hidden_states[L]` = input to decoder layer L of the host (28 layers).
- L28 = final-norm input (pre-norm output of layer 27).
- C4 = layers [14, 18, 22, 27].

**Score:** `score = x · w + b` on **raw** (unstandardized) fp32 hidden states. fit_logreg standardizes internally and the stored w/b are already de-standardized.

**probes.pt keys:**
- `PROBES['hash'][name] = {w, b, l2}`, names L10…L28, c4, w_L18, w_L22, w_c4, call, stack, stack_lg, mnl_c4, hash_mnl_c4.
- `PROBES['nl'][col] = {layers, w, b}`, cols '10'…'28', c4, w14, w18, w22, w27, wc4, we18, we22, wec4. These are line-end value probes.
- Probes named `w*` / `we*` are within-item trained. Their bias is meaningless; use them for ranking inside an item only.

**hash_mnl_c4 (recommended):** w has 8192 entries, b = −3.2435, l2 = 1.0. The input x is the concatenation:
1. The '####' state: hidden state of the token *completing* '####' (attempts table `hash_off`) at layers 14, 18, 22, 27, i.e. 4×1024 in that order.
2. The mean line-end state: the mean over the attempt's reasoning line ends, at the same 4 layers in the same order (4×1024).
   - Line ends are '\n' tokens with offset ≥ phrase_len (after the written attempt phrase) and offset < hash_off.
   - The mean is computed in fp32 then cast to fp16, matching training.
   - Use a zero vector if the attempt has no lines.

**Other probes:**
- c4: 4096 entries ('####' state only, layers 14/18/22/27), b = −6.227.
- L18: b = −1.964.
- L22: b = −2.775.

**Blend (best within-item):** `z = 0.3·(clip(lg, −10) − 1.287)/1.528 + 0.7·(s_hash_mnl_c4 − 2.006)/1.634`
- lg is the existing L18 register logit.
- The means and SDs are trainC statistics (hash_mnl_c4 from OOF scores).
- A 0.5/0.5 mix gives slightly higher dev within-item (0.779) but lower pooled.

**Per-attempt scores for offline voter work:** probe/out/scores_{trainC,dev,dev2}.jsonl.gz, written by `predict_ana.py --scores --hash_col hash_mnl_c4`.

The step-error detector weights (divprobe hdh) are **not** saved. Refit with divprobe.py, which takes 2 minutes on CPU, and add an np.save if an implant needs them.

## 8. Scripts / reproduction (all in probe/)

- `predict_fit.py --train trainC --test dev,dev2 --tag trainC`
  - Node GPU job; done in j23158.
  - Writes out/pred_trainC/{nl,hash}_{set}.npz, probes.pt and qend.json.
- `predict_ana.py --tag trainC --train trainC --test dev,dev2 --scores --hash_col hash_mnl_c4 [--skip_sim | --sims 200]`
  - Local (python3.11).
  - Covers Q1/Q3/Q4 and writes out/ana_trainC.{json,log}.
  - The cvdev version is in out/ana_cvdev2.log.
- `diverge_pred.py --tag {trainC,cvdev}`
  - Local; covers §3 (a, b, c) and the text baseline.
  - Writes out/divpred_{tag}.{json,log}.
- `mk_valtok.py {set}` and `tables/divlab_{set}.jsonl.gz`
  - Inputs for divprobe.
  - trainC divergence was produced by `atlas/diverge.run('trainC', …)` into probe/tables/diverge_trainC.jsonl.gz.
- `divprobe.py --train trainC --test {dev,dev2}`
  - CPU gq jobs j23181/j23182.
  - Writes out/divprobe_trainC_{dev,dev2}.json. The dev-trained version is in out/divprobe_dev.json.
- `probes_dump.py`
  - CPU gq job j23178; writes out/pred_trainC/verifier_v2.npz.
