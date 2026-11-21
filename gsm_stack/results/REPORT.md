# REPORT (the implant lane): what failures remain on GSM8K, and which implants could close them

> Release note: working notes as written during the runs (2026-11-20/21), with cluster paths mapped to this
> repository (`gsm_stack/...`) and host names replaced by GPU types. "task [n]" refers to the item numbers of the
> original task list; `j23xxx` are job-queue ids (labels only). Labeling slips found at release are listed in
> `gsm_stack/COMPLIANCE.md` (last section).

All numbers below are on dev600 ("dev") and dev2 (300). Both come from GSM8K train split B. No test runs were made. Probes and voters were trained on split C ("trainC", 1,499 items × 16 attempts), so dev and dev2 stay held out.
Stack: S3-serial = S2F inside N=8 serial tries at T=1, in one generation. Per-attempt answer register, L18 verifier probe, probe-weighted ReLU voter.

> **Compliance note (task [2], 02:39).** Every number in this report uses the S3 stack, whose RoPE position reset is a programmatic position-id offset. It is therefore **not** an all-weights result.
> The implant *conclusions* (what helps and what doesn't) are expected to carry over. The absolute numbers are being re-measured on the all-weights stack (comp_serial SerialWD, no reset); see LADDER.md.

## 0. Headline

> **Current all-weights top (LADDER.md, 4 seeds):** rung 4 = dev **90.46** / dev2 **88.33** (rung 0 u0: 81.04 / 79.08). See §7.1 and §7.5.


| | dev | dev2 |
|---|---|---|
| greedy S2F (N1_R2) | 82.50 | 82.00 |
| S3-serial N=8 vote (mean of 4 seeds) | 88.50 | 85.75 |
| per-attempt accuracy | 81.0 | 79.2 |
| oracle@8 / oracle@32 | 93.6 / 97.2 | 93.8 / 98.0 |
| vote wrong, but some attempt in the same stream right (selection gap) | 5.1% | 8.1% |
| vote wrong, no attempt in the stream right | 6.4% | 6.2% |
| never right in 32 attempts | 2.8% | 2.0% |

So the remaining ~10 points split about evenly:
- **selection**: a right attempt is in the stream but the vote picks a wrong one;
- **low yield**: few attempts are right (n_right small), so 8 tries often contain none.

Truly never-solved items are only 2–3%, and a third of those are gold-label or ambiguity problems.

**Implants (§4).**
- **Best confirmed: adaptive N + non-integer repair + cap 400 (A+B+C).** Over 4 seeds: dev **89.75** (+1.25), dev2 **86.42** (+0.67).
- **Same accuracy at base cost: early stop at 4 (stop4 + A16 + B + C).** Seeds 7 and 11: dev 89.50, dev2 86.67, at 6.6–6.9 attempts and base-level tokens (1181 / 1264 tok/q vs 1227 / 1243).
- **Small add-on (vfeat):** a zero-token answer-group voter, +0.50 / +0.58 over 4 seeds. Its confidence intervals include 0.

These were tried and gave null or negative results: tail truncation, the in-stream verifier v2, forced re-read, token-fork steering, and step-detector line rewind. All of them show the same thing: errors are semantic step choices that the model cannot fix locally. More independent tries for uncertain items is the lever that works.

## 1. Failure atlas (lane 1; K=32 = 4 seeds × N=8 on dev and dev2; details in notes/atlas.md, notes/atlas_counts.md)

Item classes (32 attempts each):

| set | always right | mixed | never |
|---|---|---|---|
| dev | 285 (47.5%) | 298 (49.7%) | 17 (2.8%) |
| dev2 | 140 (46.7%) | 154 (51.3%) | 6 (2.0%) |

**Categories of the first divergent step.** A wrong attempt of a mixed item is compared with the right attempts of the same item. Its first divergent step (D) is the first step whose computed value is neither a question number nor a value that the right attempts compute. Counts come from two sources:
- the "hand" column: a stratified hand-labelled sample of 120 mixed items, re-weighted to all mixed items;
- the "heuristic" column: an automatic classifier over all 4,760 wrong attempts. It agrees with the hand labels 52% exactly and 72% at coarse level (misread / omission / mechanical).

| category | hand (weighted) | heuristic (attempts) |
|---|---|---|
| wrong relation / direction (×↔÷, +↔−, double count, wrong reference) | 43% | 48% |
| dropped quantity / condition | 15% | 18–20% |
| unit / world knowledge (days per week, 4 tires, a pair = 2, ...) | 11% | 2–3% |
| wrong question target | 8% | 5–8% |
| plan (different decomposition) | 8% | 5–6% |
| wrong operand copied | 6% | 13–14% |
| arithmetic | 4% | 1% |
| early stop / truncation | 3% | 3% |
| rounding / format | 3% | 0–1% |

So the failures are overwhelmingly **semantic misreadings of one relation or quantity** (relation + dropped + operand + target ≈ 70%). Arithmetic is about 1–4%, because the calculator implant already handles it.

**Where trajectories diverge.**
- **Semantic divergence (D)** is late: median relative position 0.65–0.69 of the attempt. In 28% of attempts it is the very first step.
- **The token branch point is early.** This is the point where a wrong attempt first leaves the right attempt with the longest common token prefix in the same visible context. Its median relative position is 0.10, and 89% of branch points come before the D line.
  - Branch tokens are mostly words (79%), not digits (3%).
  - The wrong attempt is decided by an early *phrasing / plan choice* well before the wrong number appears.
- **The wrong branch is a low-probability sample.**

  | | p_w median | p_r median | p_w < 0.5 | p_w < 0.1 |
  |---|---|---|---|---|
  | branch points | 0.31 | 0.50 | 65% | 24% |
  | all sampled tokens (baseline) | | | 4–5% | 1% |

  - p_w and p_r are the probabilities of the wrong-path and right-path tokens under the same context; p_w + p_r ≈ 0.95.
  - So the step is a near-binary fork, and the right branch is the more likely one in 58% of cases.
  - For critical tokens (branch inside the D line), p_w has median 0.38 and is below 0.1 in 17%.
  - Implication: a **T=1 sampling tail** creates a large share of wrong attempts. A forward logit floor (min-p) or a lower temperature should raise per-attempt accuracy. Whether diversity survives is tested in §4.

**Wrong-answer structure.**
- A mixed item has a median of 2–3 distinct wrong answers.
- The most common wrong answer (the "attractor") takes 67% of wrong attempts.
- 70% of attractors are novel values; 18% are an intermediate value that the right attempts also compute (an early stop at the wrong quantity).

**Selection gap anatomy** (seed votes that fail although an attempt of that seed is right; dev 122 and dev2 97 item×seed cases):
- The winner is the item's 32-attempt wrong attractor in 84% (dev) and 65% (dev2) of cases.
- The winner's count is larger than the right count in 85% and 76%.
- The right count in the failing seed is 1 in 116/219 cases and 2 in 54/219.

The gap is therefore mostly "one right attempt against a wrong majority". Re-weighting agreement cannot fix it. It needs (a) a verifier that is much more confident than agreement, or (b) more right attempts per stream.

## 2. Never / near-never items (lane 4; notes/never.md)

The 76 items with n_right ≤ 4/32 were read by hand:

| category | items |
|---|---|
| dropped quantity / condition | 20 |
| question misreading | 17 |
| gold error / ambiguous | 13 |
| plan | 13 |
| world knowledge / units / calendar | 11 |
| wrong target / arithmetic | 2 |

- **Gold errors.** Six golds are plainly wrong: dev 6, 303, 488, 519 and dev2 115, 255.
- **No numeric trigger for many drops.** 9 dropped or misread items drop a *verbal* quantity ("splits with his friend", "morning and evening"), which no digit-based register can see.
- **Answer cap.** 9 items are dominated by the 320-token attempt cap.
- **Trigger audit:**

  | trigger | right attempts | wrong attempts | notes |
  |---|---|---|---|
  | non-integer `####` | 0 / 23,144 | 7.5% | 100% precision; 0/900 golds non-integer |
  | unused question number | 3–9% | 9–16% | precision 30–40%; recall 6/20 on dropped items. Rejected |
  | rare-word fact lookup (horseshoe, tire, couple, BC/born, round trip) | | | 2.4% of questions; 3–6 plausible fixes |

## 3. Predictability (lane 2; notes/probe.md, notes/branch.md)

Probes are trained on trainC and tested on held-out dev and dev2.
- *pooled* = AUROC over all attempts.
- *within* = AUROC over right/wrong pairs of attempts on the same item, which removes item difficulty.
- Cells read `pooled / within` (dev | dev2).

| question | best signal | dev | dev2 | baseline |
|---|---|---|---|---|
| final answer right? (`####` verifier) | 0.3·z(lg) + 0.7·z(hash_mnl_c4) | 0.854 / 0.773 | 0.876 / 0.700 | lg register 0.817 / 0.761, 0.834 / 0.680 |
| is this line the first divergent step D? | step detector, layers 18+22, h+Δh | 0.822 / 0.770 | 0.830 / 0.771 | text −logp of the line: 0.63, 0.59 |
| will it go wrong? (line end *before* D, within item, same step) | anticipation probe | 0.694 | 0.682 | generic value probe 0.57–0.60 |
| does the generic value logit drop at D? | Δlogit across the D line | 0.49–0.55 | 0.51 | no |
| which top-2 token is right at the branch point? | best probe on h (layers 14/18/22/27/28) | 0.700 | 0.712 | the host's own p: 0.698 / 0.710 |
| how hard is the item? (question end) | ridge on layers 14/18/22 | p_right R² 0.286 | 0.295 | question length alone 0.116 / 0.071 |

**Answers.**
- **Hidden state.** The verifier and the step detector can tell right from wrong *at and after* an error, at about 0.77 within-item AUROC. Before the error, separation is only moderate (0.68–0.69). The hidden state encodes item difficulty from the question end on; the generic value probe tracks difficulty, not the branch taken. The trainC-trained verifier improves pooled AUROC, i.e. cross-item calibration, but within-item ranking gains only +0.01/+0.02.
- **Text.** Weak signal. Low-probability branch tokens (p < 0.1) explain only about 22% of divergences, and 59–61% of right attempts contain one too.
- **Token forks are not decisions.** The atlas's "right branch more likely 58%" was a partner-selection artifact (0.633 with labels shuffled within each item). Child outcomes at a fork differ by only ΔP(right) ≈ 0.04–0.06. Token-level steering therefore has nothing to act on.
- **Question.** Partly predictable: about 29% of p_right variance at the question end, and "always right" items separate at AUROC 0.81. A value-gated restart sim gave no gain.

## 4. Implants built (lane 5; code + fuzz tests in implants/*)

Every implant is env-gated, runs in the single S3-serial generation, and is fuzz-tested against an offline reference implementation.
- *all-off* = the implant's code is present but every knob is off. All implants are bit-identical to base in that state.
- Vote accuracy is a seed mean. "fixes / breaks" counts items per seed against the base stream with the same seed.

| implant | what it changes | seeds | dev | dev2 | fixes / breaks per seed (dev; dev2) | attempts | tok/q vs base | verdict |
|---|---|---|---|---|---|---|---|---|
| base S3-serial | — | 7, 11, 13, 17 | 88.50 | 85.75 | — | 8.0 | 1223 / 1241 | |
| **A+B+C** (implants/adapt) | A: adaptive N — while the winner's probe-weight share is < 0.7, add tries in blocks of 4, up to 16. B: up to 3 " We need to find" repairs for a non-integer `####`. C: attempt cap 320 → 400 | 7, 11, 13, 17 | **89.75** | **86.42** | 13.2 / 5.8; 9.0 / 7.0 | 9.1 / 9.3 | +22% / +27% | **best confirmed; +1.25 / +0.67** |
| A alone | as above | 7 (dev); 7, 11 (dev2) | 90.00 | 86.17 | 13 / 4; 8.5 / 6.0 | 9.1 / 9.4 | +19% / +23% | matches the offline sim (+0.9) |
| B alone (3 repairs) | | 7, 11 | 89.17 | 85.17 | 9.0 / 4.5; 4.5 / 5.0 | 8.0 | +3% | fires on 2–4% of rows; noise |
| stop4 + A(Nmax 24) + B + C | stop after 4 tries if they agree with probe ≥ 0.5, else as A but up to 24 tries | 7 (dev); 7, 11 (dev2) | 88.50 | 87.67 | 9 / 9; 10.5 / 3.5 | 7.7 / 8.0 | +15% / +19% | early stop is safe (0 fixes / 0 breaks on 715 stopped rows); extension mixed |
| **stop4 + A16 + B + C** (implants/adapt) | stop after 4 tries if they agree with probe ≥ 0.5; else A+B+C | 7, 11 | **89.50** (A+B+C same seeds 89.58) | **86.67** (A+B+C 86.67) | 14.0 / 7.5; 9.5 / 5.5 | 6.6 / 6.9 | **−4% / +2%** | **recommended: A+B+C accuracy at base tokens** |
| voter v2 (implants/voter2) | in-stream hash_mnl_c4 verifier register; blended voter sigmoid(1.5z) | 7, 11 | 88.25 → 88.33 | 85.33 → 85.50 | same-stream: 4.5 / 4; small | 8.0 | +0% | null: pooled AUROC gain is cross-item calibration and does not change within-item choice |
| min-p / T floor (implants/trunc) | min-p 0.05 / 0.1 / 0.2, T 0.7 | 7, 11 | 87.67–88.50 | 85.00–86.50 | | 8.0 | ~0% | null: wrong and right branch tokens are the host's top-2, so no tail exists to cut |
| forced re-read (implants/reread) | attempts k≥1 start by re-copying the question (full / qa) or with a "read carefully" phrase | 7, 11 | 88.17–88.92 | 84.83–86.17 | 29–36 / 23–35 per 2 seeds | 8.0 | +13–40% | negative: per-attempt accuracy falls 0.7–4.6 pt; misreading is not caused by distance from the question |

Exactness:
- A/B/C: dev2 600/600 rows bit-identical to base with everything off. Rows where nothing fired diverge only after an implant event earlier in their batch (sampler RNG is shared per batch position).
- Re-read: bit-identical through the end of attempt 0 (256/256 per set and variant).
- min-p at m=0: bit-identical (128/128).
- voter v2: in-stream vs offline corr 0.9996.

**Line rewind (implants/linefix): the gate failed, so no in-stream implant was built.**
- Method: the step detector (2% line false-positive rate) flags about 10% of attempts. At the first flagged line, the attempt resamples from the end of the previous line, with K=4 samples.
- Flagged wrong attempts became right 14% of the time. Flagged right attempts broke 20% of the time (5% for control rewinds).
- Net change per attempt: dev +0.06, dev2 −0.07 points, against a +1.0 gate. With zero breaks the ceiling would be +0.86.
- 42% of rewinds went back to the same wrong answer. A fresh attempt (~32% right on these items) is about 2× better than a rewind.

**Voter features (implants/vfeat, `SER_VF=1`): small and unconfirmed.**
- Question-number coverage is useless: pooled AUROC 0.54. Wrong answer groups cite every question number as well.
- Close-fork density is real (5.2 vs 6.8 per 100 tokens) but redundant with the existing voter inputs.
- The best voter scores each answer group as a whole by length, distinct-number count and coverage; it was trained on trainC.
  - Offline, 4 seeds: dev +0.71, dev2 +0.58 (95% CIs include 0).
  - In-stream, same-stream, seeds 7, 11, 13, 17: dev **+0.50** (89.04 vs 88.54; 30 fixes / 18 breaks), dev2 **+0.58** (86.33 vs 85.75; 20 / 13), at zero token cost.
  - 95% CIs: dev [−0.17, +1.17], dev2 [−0.33, +1.58]. Per-seed results range from −0.67 to +2.0.
  - Weak but consistent. Stacking with A+B+C is untested.
- Step-detector max score as a voter feature: at most +0.17 on dev2 (fit on dev).

## 5. Voter upgrades, offline (lane 3; voter/)

All voters were trained on trainC; the table reports the vote accuracy on dev and dev2 (mean over 4 seeds) with fixes and breaks against the stack voter:

| voter | dev | dev2 |
|---|---|---|
| stack (probe-weighted) | 88.50 | 85.75 |
| count only | 87.58 | 85.58 |
| listwise, all 21 features | 88.67 (+24 / −20) | 86.25 (+19 / −13) |
| listwise exp-weight, all features | 89.08 (+34 / −20) | 86.25 (+19 / −13) |
| lg + agreement only | 88.67 | 86.50 |

**Ceiling simulation.** This holds attempts fixed and replaces the verifier with a synthetic one of a given AUROC:

| synthetic verifier AUROC | dev | dev2 |
|---|---|---|
| 0.85 | ≈ 91.2 | ≈ 90.5 |
| 0.90 | ≈ 92.0 | ≈ 91.7 |
| 0.99 (oracle@8) | 93.4 | 93.7 |

**Adaptive extra tries** (offline, by resampling the 32 attempts): with tau 0.7 and Nmax 24, dev reaches 89.47 and dev2 86.88 at about 10 tries on average. A fixed N=16 gives 89.38 / 86.53.

## 6. Ranked next implants

The ranking is by expected gain per unit of risk, given the findings above.

What the findings say:
- Errors are **semantic step choices**: relation, dropped quantity and target. They are not token accidents and not arithmetic.
- The hidden state recognises an error *after* it happens (within-item AUROC ~0.77), but barely before it (0.68).
- Re-reading the question, steering at token forks, and rewinding a line do not help.
- What works is more *independent* tries for items with a split vote, chosen by the probe register's own uncertainty.

1. **Adaptive budget, both ways.** Stop early when the first 4 tries agree, and extend only split items to Nmax 16.
   - Confirmed: A+B+C gives +1.25 / +0.67 over 4 seeds. stop4 + A16 keeps that accuracy at base tokens, and its early stop fires on about 62% of rows with ≈0 fixes / breaks.
   - Next: a difficulty-aware first block. The question-end probe (R² 0.29) decides whether to start at 4 or 8 tries, and stops more aggressively on "always" items (AUROC 0.81). That frees budget for low-yield items.
   - Expected: about +1.0–1.5 at ≤ base tokens.
2. **Diversity-forcing extra tries.** For items the voter has split on, mask (forget) the leading wrong attractor's key lines in the extra attempts. 65–84% of selection failures are a dominant wrong attractor, and right_count = 1 is common.
   - Measure first, offline from the K=32 pool: the right rate of attempts that do *not* share the attractor's D line. Build only if the conditional right rate beats the unconditional rate by more than 5 pt.
3. **Answer-group voter features** (vfeat: +0.50 / +0.58 same-stream over 4 seeds, zero tokens, CIs include 0). Next: stack with A+B+C and test on a fresh dev3 split. Per-attempt features (coverage, fork density, step-detector max) did not help. A trainC-scored step detector inside a group-level voter is the remaining variant.
   - Within-item ranking is the bottleneck. The ceiling simulation says a within-item AUROC gain of +0.05 is worth about 1 pt.
4. **Exact fact lookup for rare-word conventions** (horseshoe = 4, couple = 2, round trip ×2, BC/born dates).
   - This would be a single-call written fact line triggered on the token. It touches 2.4% of questions, with 3–6 plausible fixes per 900 items, i.e. ≤ +0.5.
5. **Gold-label hygiene.** dev 6, 303, 488, 519 and dev2 115, 255 are gold errors. They are not fixable and should be excluded from headroom estimates; they cost 0.5–0.7 pt of the "never" bucket.

Tried, negative or null, not recommended:
- tail truncation (min-p / T);
- in-stream verifier v2 (calibration only);
- forced re-read of the question;
- token-branch steering (forks are not decisions);
- step-detector line rewind (fixes 14%, breaks 20%);
- value-gated restart.

**Final stack candidate for the coordinator's single test run:** S3-serial + stop4 + A16 + B + C.
- Env: `SER_NMAX=16 SER_WS_E=64 STK_MAXN=7936 SAN_SMAX=8192`
- STK_SER: `8:1:1:1:0:T1.0:A=0.7/16/4:S=4/0.5:B=3:C=400`
- Code: implants/adapt/job/stk; launcher run_adapt.sh.

Expected gain over base S3-serial is about +1.0–1.3 on dev-like data, at the same token count and about 2.7× wall time.
The fallback is A+B+C without `S=4/0.5`, which has 4-seed confirmation. Optionally add `SER_VF=1`, but its stacking with A+B+C is untested.

## 7. Session 3: the all-weights ladder and mechanistic lanes (task [5]–[9])

### 7.1 All-weights ladder (LADDER.md; 4 seeds 7/11/13/17; dev / dev2)

| rung | what it adds (all weights) | dev | dev2 | tok/q |
|---|---|---|---|---|
| 0 | u0 stack (SerialWD, no position reset) | 81.04 | 79.08 | 1079 / 1085 |
| 1f | + weight-space position reset (WPR: offset register + fixed rotary table R(−off) on q / new K) | 88.79 | 85.92 | 1229 / 1252 |
| 2 | + adaptive tries as a compiled comparator (A/S) + non-integer repair counter (B) + cap 400 (C) | 89.58 | 86.92 | 1196 / 1299 |
| 3 | + Nmax 24 + answer-group voter as registers (AW_VF: number-set automaton registers + fixed-θ ClusterLin final writer) | 90.29 | 87.33 | 1401 / 1522 |
| 4 | + side-net verifier (fixed linear vector on L14/18/22/27 residuals → registers → blended into the ClusterLin writer; AW_SN) | **90.46** | **88.33** | 1401 / 1522 |

- Rung 3 vs rung 2: 38/21 and 27/22 fixes/breaks; pooled 65/43, one-sided sign p = 0.021.
  - Nmax 24 alone (+0.12 / +0.17) and VF alone (+0.04 / +0.58) are both noise. Extra tries pay only with the better final writer.


- Rung 2 reaches the non-compliant S3 reference (89.75 / 86.42) entirely in weights.
- **WPR bug (fixed 03:40Z).** A helper `arange(B_max+1)` was treated by fastroll compaction as a per-row register and permuted. Later seeds in a process then decayed (dev 88.7 → 73).
  - The fix is two lines. Multi-run output is now bitwise equal to fresh runs (300/300).
  - Handed to the stack lane in implants/wpr/HANDOFF.md.
- Without WPR, adding attempts hurts: A+B+C on rung 0 gives dev2 ~76.7.
- Neutral variants:
  - WPR_K=2 (kernel reads the rotary row at pos−off; bit-identical to the programmatic reset): 89.33 / 86.75.
  - WPR_K=1 (fp32): 89.17 / 86.50.
  - AW_VF=1 (answer-group voter as registers): 89.63 / 87.50, i.e. +0.04 / +0.58, within noise.

### 7.2 Failure atlas of rung 1f (notes/atlas_aw.md)

- Per-attempt accuracy is flat across attempt index (81 / 79), so after WPR there is no position lever left.
- Error mix is unchanged: relation ~40%, dropped quantity ~19%, target 9–16%, unit 3–14%.
- Selection gap is 4.8 / 7.0 pts. In 80% / 67% of gap items the wrong winner has *more* votes than the right answer.
- Ranked ceilings (dev / dev2): perfect selection 4.8 / 7.0, relation guard 2.3 / 3.3, quantity coverage 1.5 / 2.5, unit facts 2.0 / 1.5.

### 7.3 Mechanistic and implant lanes (all weights, gate-off bitwise checks passed)

| lane | mechanism | result (vs same-tree base) | verdict |
|---|---|---|---|
| sel-check | in-stream verification writer for contested votes | On gap items, the model's own checks (yes/no, backward check, conditioned re-solve, probes) rank right above winner at 0.37–0.64, i.e. chance. Best eval-tuned simulated gain +0.25 / +0.92 (gate +1). Not built. | negative |
| M5 | first-step diversity: registers of earlier first lines; fixed −β logit bias | First lines become distinct (4.0 → 5.3–7.9 per 8) but distinct answers do not (1.80 → 1.88). 4 seeds: −0.50 / +0.50 and +0.09 / −0.25. | null |
| M4 | unit/fact skill: compiled fact-word DFA + value table + trained side map at the fired question tokens | Map trained on gold answers: −17 / −3 (copies the terse gold style). DPO map on the host's own attempts: −0.7 (26 / 39 fixes / breaks), rare facts −0.2. Real fact errors ≈ 0.07% of attempts. The compiled trigger itself is exact (causal_lw 97.3 / 96.7). | negative |
| M1 | attention forensics → head patching → fixed value-vector write | p2: copying the right attempt's outputs of 8 heads (L17–23) flips the wrong critical token in ~60% of pairs (controls 2–24%). Copying question attention does not (≤0.12, = shuffled control), and role-keyed attention-logit bias does nothing. p3 (j23432): a fixed mean(zR−zW) write on those or all heads gives ≤ +2.4 / +2.5 net over random. It sharpens the current choice (it pushes W more wrong and R more right). | negative: the head-output lever is specific to each attempt's own context; a fixed write or attention bias cannot carry it. Lane closed. |
| M2 | trigger-gated fixed residual vector at the operator commit (right-vs-wrong diff-of-means at D, trainC, L14–27) | Dev offline sweep j23414: best non-oracle +3.5 net vs random control +2.0. Any config above +4 breaks 2.5–25% of right steps (13× weight in use). The oracle per-case right-op direction scores +10.7 (129 / 22) and flips 121/121 wrong-operator cases. | negative: operator directions are causal, but the right sign is not readable at D (AUROC .55–.67), so detection is the bottleneck. No implant built. |
| M3 | question-structure latent slot: fixed parser + trained side map at the 3 question-end tokens, layers 7–27 | 4 seeds vs 1f: −0.83 / +0.50 (mix), −0.58 / −0.08 (CE). It cuts first-attempt accuracy to 65–69 and does not reduce dropped/target errors. The constant-vector control was +0.54 / +0.50 (noise level); see §7.4. | negative |
| diff-block | question-end difficulty register (ridge on L14/18/22, fixed linear map) driving the A/S budget comparators | Offline replay simulator (0 vote mismatches vs real rung-2 rows), 690 policies, knobs chosen on one set and scored on the other: +0.05 / +0.05. Near-oracle difficulty: +0.2. Hard-item extra budget sits 0.1–0.25 below uniform knobs at equal tokens. Side lead: uniform A=0.6/20/4:S=3/0 matches rung 2 at −15% tokens. | negative (not built) |
| const tail vector | m3qs constant trained vector at the question-end tokens, on rung 2 | −0.42 / −0.33 (54 / 68). The earlier +0.5 on rung 1f was noise. | negative |
| vf-refit | refit the rung-3 ClusterLin writer on trainC rows under adaptive N (j23422; 4 seeds, 89.14) | Share instead of count/8, probe share, prior-pulled fits, subset training: trainC CV −0.02…−0.12, dev −0.25…+0.17, dev2 +0.4…+1.0 | negative: the writer is saturated on trainC |
| ladr3 extra budget | on rung 3: Nmax 32 + VF, const vector, tau 0.6 S=3/0 (+VF, Nmax 24/32) | a32vf −0.42 / −0.25; avfqc −0.04 / +0.42; c32vf −0.79 / +0.33; cvf −0.75 / +0.75 at −28% / −22% tokens (= rung-2 accuracy at −15% / −9% tokens: efficiency point) | budget saturated at Nmax 24 (diff-block sim: rung 2 within 0.3–0.5 of the 32-try limit) |

**Reading so far.**
- The 0.6B host's remaining errors are *beliefs*. Its own checks, re-reads, diverse openings and injected facts reproduce or disturb those beliefs; they do not correct them.
- The only reliable lever remains *more independent tries where the vote is split*, which is rung 2.

### 7.4 Failure atlas of rung 3 (notes/atlas_r3.md)

| bucket (rows over 4 seeds, pts) | dev | dev2 | trainC |
|---|---|---|---|
| failures | 233 (9.71) | 152 (12.67) | 651 (10.86) |
| selection gap (a right attempt exists, writer picks wrong) | 108 (4.50) | 91 (7.58) | 386 (6.44) |
| · right group outnumbered / singleton / larger but lost / tied | 50 / 34 / 20 / 4 | 47 / 18 / 22 / 4 | 177 / 119 / 69 / 21 |
| generation miss (no right attempt in up to 24) | 125 (5.21) | 61 (5.08) | 265 (4.42) |
| · gold-label error or ambiguous gold | 1.0 | 1.67 | – |
| oracle@slots | 94.79 | 94.92 | 95.58 |

- **Selection gap.** The right group has fewer votes and lower Σσ(lg) in 71–83% of gap rows.
  - No compiled re-pick on the existing registers is CV-positive on trainC.
  - 60–70% of gap rows already used all 24 tries.
- **Generation misses**, hand-labelled (dev / dev2 pts):
  - world fact: 1.29 / 0.17
  - dropped quantity: 0.83 / 1.42
  - misread relation: 1.04 / 1.25
  - plan: 1.00 / 0.17
  - non-integer answer needing ceil: 0 / 0.42
- **Stack-caused failures** (cap, repairs, sanity filter, parse, `####` copy) total ≤ 0.3 pts.
- **Signature errors.** The picked value is 2× or ½ of the gold in 10–11% of failing rows (also on trainC), and gold ± 1 in 4–7%.
- **Ranked next ideas.**
  - operand-use register (question quantities actually used as operands, incl. number words) as a VF feature and extension trigger: 0.2–0.5
  - relation-cue implant (each / per / both / round trip; the 2× / ½ signature): 0.3–0.6
  - world-fact memory: dev only
  - ceil writer: ≤ 0.4 on dev2

### 7.5 Sessions 5–6: rung 4 and the closed lanes

**Rung 4: side-net verifier (implants/sidenet).**
- A linear map on hidden states at layers 14/18/22/27. It reads two places per attempt: H, the state at the token completing `####`, and M, the mean over the body's line-end tokens. It was trained on trainC with a pairwise within-item loss and folded into one fixed vector.
- In-stack, register accumulators feed a per-slot score R_sn. The final ClusterLin writer uses max(sn, −12) + max(lg, −12), with fixed θ fit on trainC out-of-fold scores.
- 4 seeds vs rung 3: dev +0.17 (19 / 15), dev2 +1.00 (16 / 4); pooled 35 / 19, p = 0.020. Zero extra tokens.
- Within-item AUROC of the verifier is only ~0.73 / 0.66 (dev / dev2) and MLPs never beat linear. The ceiling is the information in the states, not the readout.

**Closed lanes (negative).**

| lane | idea | result | why |
|---|---|---|---|
| operand | register of question quantities used as operands → writer feature / extension trigger | trainC CV ≤ +0.03, dev ≤ +0.12 | in gap rows the right group is no better covered than the wrong winner |
| relcue | compiled relation-cue DFA (each / per / both / pair / round trip; exact on 7.5k questions) + DPO-trained write at the cue tokens, aimed at the ×2 / ½ errors | held-out pair accuracy 0.55 / 0.47 = scrambled-cue control; SNIS sign flips; −0.5 nats drift on always-right attempts | the map memorises train items; whether a cue means ×2 depends on its referent in each question, so a fixed per-cue write cannot carry it |
| shotctx | (A) different 8-shot exemplar sets per attempt (prompt ensemble); (B) retrieved similar split-A exemplars | (A) single alternative prefixes are equivalent to v1 (signs flip); shot-mix at equal budget (exact writer replay) +0.07 / +0.54, gate +0.5 on both; cross-prefix same-wrong-answer rate 0.31 → 0.29. (B) −3.1 / −0.7, per-attempt −4 / −2 pts | the attractor errors are properties of question × model, not of the exemplars |
| sndata | retrain the side-net on 2.5k split-A items (rung-3 rollouts, 2 seeds) + trainC | within-item AUROC .742 / .664 vs .727 / .660; every re-pick below g4 (pooled 8/17 … 16/24); flat learning curve | the verifier is not data-limited |
| ladr5 | on rung 4: A/S comparators read the blended sn+lg score; Nmax 32 | −0.63 / +0.08 (40/54); −0.29 / −0.25 (17/27) | the budget is saturated even with the better writer |
| attnver | 8,960 attention summaries per attempt (28 layers × 16 heads × 20, incl. attention coverage of question numbers) as verifier features | alone AUROC .725 / .657, but within-item corr .80 with sn+lg; increment +0.005 / +0.004; re-pick +0.04 / −0.08 | attention carries the same right/wrong signal as the residual states |
| pathdiv | distinct reasoning paths per answer group (body number-sets, Jaccard, dedup count) as writer features | wrong winner groups have MORE distinct paths (2.4 vs 1.7); best buildable +0.08 / −0.17 (14/14) | paths per member just tracks group size |
| headmix | per-attempt model perturbation (head drop / head-scale noise / MLP dropout, a fixed table indexed by the attempt counter) | mixed-mask vote vs rung 4: −8.6 / −8.8, −0.17 / −1.03, −0.15 / −0.37 (σ 0.05: +0.39 / −0.66); same-wrong rate −22% | perturbations add scattered errors; the attractor-to-right ratio is unchanged, because attractors belong to question × model family |
| prmstep | linear step-correctness probe on line-end residuals (labels: each line's result ∈ gold intermediate values; trainC) | line AUROC .78 but within-item .54–.65 (sn+lg .70–.75); trainC-picked weight 0 → re-pick = rung 4; fixed weights ≤0 / ≤+.08 | gold step labels would separate attempts (AUROC .89) but residuals don't encode them within an item |
| r4eff | rung 4 at the cheap budget (A .6/24 : S 3/0 + sn) | 89.50 / 88.08 at 1015 / 1180 tok/q = cvf (f/b 17/18) | the writer needs the full budget to have something to re-pick |
| chain | cross-attempt chain containment (winner is an intermediate of the right chain, or the reverse) as writer group features | early-stop signature as common in control rows as in gap rows (dev 17% vs 19%); trainC-selected writer −0.04 / +0.17 (3 / 2) | the gap is group size (≈7 vs 4 attempts), not chain structure |

**Rung 4 on 8 seeds** (implants/r4seeds). Rung 3 can be re-picked exactly from the same streams (0/3600 mismatches), so the comparison is paired: dev 90.34 vs 90.00 (41/25), dev2 88.17 vs 87.59 (22/8), pooled 63/33, p = 0.0014. Seed-to-seed spread of a 4-seed mean is about 0.5 pts, so only paired comparisons are meaningful at this level.

**Reading.**
- On the selection gap, the only lever that has paid is a stronger *verifier* signal read from hidden states (rung 4).
- Every other readable signal carries the *same* ~0.73 / 0.66 within-item right/wrong information and adds nothing on top: residual states at more layers, attention patterns, text structure (coverage, operand use, chain containment, path diversity, fork density, length), and 2.7× more training data.
- Diversifying the attempts, through exemplar prompts or weight perturbations, does not weaken the attractor relative to the right answer.
- The remaining ~10 / 12 pts therefore split into:
  - a selection gap at the verifier's information ceiling for this 0.6B host;
  - generation misses that are shared beliefs of the model family.
- **Ranked next ideas:**
  - (1) A stronger host, via the RL arms with WPR (the stack lane). Rung 4 drops in unchanged.
  - (2) *Tested and NEGATIVE:* a verifier trained on step-level correctness labels (prmstep). The labels are informative, but the host's residual states don't encode them within an item.
  - (3) The efficiency variant of rung 4 (r4eff) is NEGATIVE: A 0.6/24 : S 3/0 + sn scores 89.50 / 88.08 at 1015 / 1180 tok/q, the same as rung-3 cvf (f/b 17/18). The writer has nothing to re-pick at natt ~5.5.
