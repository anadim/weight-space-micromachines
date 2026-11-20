# Answer selection and verification over K samples (gsm_push, 2026-11-20)

**Question.** The RL+calc host (a2_u300_s2) has GSM8K test pass@8 of 89.3 but pass@1 of 71.0. An implant that draws K
samples inside the engine and *picks* one should recover part of that gap. How much do majority vote, an
impossible-answer filter, and a learned hidden-state verifier add, at what token cost, and does an adaptive K help?

**Short answer.** (DEV = 600 split-B GSM8K train items, DEV2 = 300 more; greedy 80.33 / 77.00; T = 1.0 samples)
- **Majority vote implant works and scales with K.** Greedy 80.33 / 77.00 -> majority 81.9 / 81.0 (K = 4),
  83.9 / 82.0 (K = 8), 84.8 / 84.0 (K = 16), 85.3 / 83.3 (K = 32). Cost is K x the greedy tokens (one answer is about
  136 tokens). At K = 2 a plain vote is worse than greedy (ties go to a T = 1 sample).
- **The sanity filter is a free +1.0 to +1.9** at every K on both sets (drop negative / non-integer / missing answers
  before the vote). It adds no tokens.
- **The learned verifier adds on top.** A logistic probe on the host's layer-18 state at the `####` token, fit on
  split C only (11,992 samples), weights the vote. Probe vote vs majority: +3.2 / +3.6 at K = 2, +1.9 / +1.8 at K = 8
  (CIs exclude 0). On top of the sanity filter it adds +2.0 / +2.3 at K = 2, +1.0 / +0.4 (n.s. on DEV2) at K = 8:
  the probe matters most at small K, where the filter has little to vote with.
- **Best rule = probe vote + sanity:** DEV 82.0 / 84.2 / **86.1** / 87.2 / 87.7 and DEV2 80.4 / 83.2 / **84.4** /
  85.7 / 86.7 at K = 2 / 4 / 8 / 16 / 32. At K = 8: **+5.8 [3.7, 8.1] on DEV and +7.4 [4.0, 10.8] on DEV2** over greedy,
  at 8x tokens. 42 fixes / 6 breaks on DEV, 29 / 7 on DEV2.
- **Adaptive K saves about 25% of the samples at the same accuracy** (k0 = 4, stop if the winner holds 75% of the
  weight: 86.7 at 6.0 samples on DEV, 85.0 at 6.3 on DEV2). The cheapest useful point is a sanity-gated vote: keep the
  greedy answer unless it is impossible, then sample and vote: +2.2 / +3.3 at 1.7-1.8x tokens, which beats the
  sanity implant's single restart (+1.1 on test).
- **Not general.** On a GSM-Hard-like slice (split-B scaled numbers) majority still helps (+4.9 at K = 8), but the
  sanity filter (-3.1) and the probe (-2.0; AUROC 0.58 vs 0.65 for log-prob) both hurt. T = 0.7 is no better than
  T = 1.0. Selecting the single highest-probe sample ("best-of-N") is worse than voting.

## 1. Data, host, discipline
- Host: `rl_implants_v2/runs/long/a2_u300_s2/step_000300` (sha256 ee82ee1e...), calculator implant attached (arm 2,
  calc_s4, hand-wired end write), 8-shot prompt, 320-token budget, RL engine (`rl_wiring/engine.py`, bf16, CUDA graphs).
- Splits (`common.part` = md5(question) mod 10; A = RL pool, B = 20%, C = 20%):
  - **DEV**: split-B GSM8K items (shots removed, prompt + 320 <= 544 tokens), `random.Random(0).shuffle`, items 0-599.
  - **DEV2**: the same shuffle, items 600-899. Used only to confirm configs chosen on DEV.
  - **Verifier training**: all 1,499 split-C GSM8K items (disjoint from A, B and test). Nothing learned touches DEV,
    DEV2 or test. GSM8K test was not run.
  - Note: split-B items are easier for this host than test (greedy 80.3 / 77.0 vs test 71.2), as split A was (83.0).
- Samples: K = 16 per item at T = 1.0 (the RL temperature) and at T = 0.7; greedy once. K < 16 numbers are the mean
  over 30 random K-subsets of the 16 (the first subset is samples 0..K-1), so they are expected accuracies at K.
- CIs: paired bootstrap over questions (4,000 resamples), 95%.

## 2. The implant
Everything runs in the engine's decode, except the final regex parse of the answer number (see "disclosed").
1. **K rows per question.** `engine.generate(G=K)`: the question's KV is prefilled once and copied to K rows; the rows
   decode in one batch (shared 8-shot prefix KV). Cost = K x the tokens of one answer.
2. **Answer-marker latch + state capture** (`selection/probe_engine.py`, `ProbeEngine`, inside the captured CUDA graph).
   A token table `nh[tok]` (number of '#' in an all-'#' token) drives a per-row counter; the first time it reaches 4
   (the token that completes `####`), the residual stream at that position is copied out at layers 14 / 18 / 22
   (layer-L input = HF `hidden_states[L]`). It only reads x; when nothing is captured the decode is the plain engine's:
   greedy ProbeEngine == plain Engine on 16/16 rows; the captured state equals a teacher-forced HF forward at that
   position (cos 0.9998-0.9999 at all three layers, bf16).
3. **Verifier probe.** One affine read of the layer-18 state: p = sigmoid(w . h + b) (logistic regression,
   standardisation folded into w, b; `out/verifier_probe_L18.pt`). Fit on split C only (T = 1.0 samples, G = 8, 11,992
   rows, label = "this sample's final answer is right"). Layer and C chosen by 5-fold question-grouped CV of the
   probe-vote accuracy on split C: L18, C = 0.001 (CV vote 85.3, AUROC 0.846; L14 0.837/0.800, L22 0.846/0.842).
4. **Group reduction** (`group_select` in `probe_engine.py`, tensor ops over the K rows): each row's answer value is
   compared exactly (relative 1e-6) with every other row's; a row's score is the summed weight of rows with the same
   answer; the earliest row with the top score wins. Weight = 1 (majority) or p (probe vote). With the sanity filter,
   rows whose answer is negative, non-integer or missing are dropped first, unless every row is dropped.
   The tensor version agrees with the Python reference on 600/600 DEV and 300/300 DEV2 questions.
- **Disclosed:** the answer value is read with the scorer's own regex on the decoded row, then fed to the tensor
  reduction. An in-graph equivalent is the sanity implant's DFA (`dig/sanity/san_engine.py`) plus a digit accumulator
  (Qwen writes one digit per token); not built. The probe reads the host's hidden state, not text.
- **Generic.** f(samples) = "the answer most of the attempts reach, weighted by how much the host's own state at the
  answer marker looks like a correct attempt". Nothing in it refers to GSM8K except the sanity filter's prior
  (answers are non-negative integers), which is wrong on GSM-Hard-like data (§5).

## 3. Results, T = 1.0

Greedy: DEV **80.33**, DEV2 **77.00** (136 tokens per answer). Oracle (any of 16 right): 93.8 / 94.0.
Probe fit on split C at T = 1.0, L18. Token cost = K x one answer (about 136 tokens); "g+" = greedy row + K-1 samples.

**DEV (600)**

| K | tokens vs greedy | majority | majority + sanity | probe vote | **probe vote + sanity** | probe select (best single) | log-prob vote | g+ probe vote + sanity |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.0x | 78.58 | 78.58 | 78.58 | 78.58 | 78.58 | 78.58 | (greedy 80.33) |
| 2 | 2.0x | 78.75 | 80.03 | 81.98 | **82.02** | 81.98 | 79.42 | 82.82 |
| 4 | 4.0x | 81.91 | 82.92 | 83.94 | **84.19** | 83.56 | 82.16 | 84.45 |
| 8 | 8.0x | 83.88 | 85.09 | 85.74 | **86.11** | 84.78 | 83.74 | 86.26 |
| 16 | 16x | 84.83 | 85.83 | 86.67 | **87.17** | 84.17 | 84.67 | 87.08 |
| 32 (2 seeds) | 32x | 85.33 | 86.50 | 87.00 | **87.67** | 85.50 | 85.50 | 87.45 |

**DEV2 (300, confirmation; nothing chosen here)**

| K | tokens vs greedy | majority | majority + sanity | probe vote | **probe vote + sanity** | probe select | log-prob vote | g+ probe vote + sanity |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.0x | 76.39 | 76.39 | 76.39 | 76.39 | 76.39 | 76.39 | (greedy 77.00) |
| 2 | 2.0x | 76.68 | 78.04 | 80.26 | **80.36** | 80.26 | 78.12 | 80.02 |
| 4 | 4.0x | 80.96 | 82.48 | 82.68 | **83.23** | 81.30 | 80.87 | 82.47 |
| 8 | 8.0x | 81.99 | 83.93 | 83.77 | **84.36** | 81.10 | 81.89 | 84.26 |
| 16 | 16x | 84.00 | 85.67 | 85.00 | **85.67** | 81.00 | 83.33 | 85.49 |
| 32 (2 seeds) | 32x | 83.33 | 86.00 | 86.00 | **86.67** | 80.33 | 83.33 | 86.61 |

Paired differences (pp, 95% CI):

| comparison | DEV K=2 | DEV K=4 | DEV K=8 | DEV K=16 | DEV2 K=2 | DEV2 K=4 | DEV2 K=8 | DEV2 K=16 |
|---|---|---|---|---|---|---|---|---|
| majority - greedy | -1.6 [-3.4, 0.4] | +1.6 [-0.2, 3.4] | +3.5 [1.6, 5.5] | +4.5 [2.3, 6.8] | -0.3 [-3.1, 2.5] | +4.0 [1.0, 7.0] | +5.0 [1.9, 8.2] | +7.0 [3.3, 10.7] |
| sanity filter: maj+san - majority | +1.3 [0.9, 1.6] | +1.0 [0.7, 1.4] | +1.2 [0.6, 1.9] | +1.0 [0.3, 1.8] | +1.4 [0.9, 1.9] | +1.5 [0.8, 2.4] | +1.9 [0.9, 3.2] | +1.7 [0.3, 3.3] |
| probe: probe vote - majority | +3.2 [2.6, 3.9] | +2.0 [1.5, 2.6] | +1.9 [1.2, 2.6] | +1.8 [0.8, 3.0] | +3.6 [2.6, 4.6] | +1.7 [0.9, 2.6] | +1.8 [0.7, 2.9] | +1.0 [-0.7, 2.7] |
| probe on top of sanity: pv+san - maj+san | +2.0 [1.4, 2.6] | +1.3 [0.8, 1.7] | +1.0 [0.5, 1.5] | +1.3 [0.5, 2.3] | +2.3 [1.5, 3.2] | +0.8 [0.1, 1.3] | +0.4 [-0.1, 1.0] | +0.0 [-1.3, 1.3] |
| **pv+san - greedy** | +1.7 [-0.3, 3.6] | +3.9 [1.9, 5.9] | **+5.8 [3.7, 8.1]** | +6.8 [4.7, 9.2] | +3.4 [0.5, 6.3] | +6.2 [3.3, 9.3] | **+7.4 [4.0, 10.8]** | +8.7 [5.0, 12.7] |
| g+pv+san - greedy | +2.5 [1.4, 3.7] | +4.1 [2.6, 5.8] | +5.9 [4.0, 7.8] | +6.8 [4.6, 8.9] | +3.0 [1.3, 4.9] | +5.5 [3.2, 7.8] | +7.3 [4.4, 10.3] | +8.5 [5.1, 12.1] |

K = 32: pv+san - greedy +7.3 [5.0, 9.7] DEV, +9.7 [6.3, 13.3] DEV2; pv+san - maj+san +1.2 [0.3, 2.2] / +0.7 [0.0, 1.7].
The curve saturates: doubling K from 8 to 16 to 32 adds about +1.0 then +0.5 on DEV. Oracle at 32 is 95.8 / 95.7, so
selection, not coverage, is what is left (DEV: 87.7 picked vs 95.8 reachable).

Other numbers:
- Probe quality: within-question AUROC (questions with both right and wrong samples) 0.770 DEV / 0.768 DEV2, vs 0.595 /
  0.611 for the sample's mean token log-prob. Log-prob voting is no better than majority (-0.4 to +1.4).
- "probe select" (take the single highest-p sample) is worse than voting from K = 8 on, and gets worse with K on DEV2
  (81.0 at K = 16): with more samples there are more confident wrong ones. The probe is a good *weight*, a poor *judge*.
- Fixes / breaks of probe vote + sanity at K = 8 (first 8 samples) vs greedy: DEV 42 / 6, DEV2 29 / 7.
- A pooled probe (split C at T = 1.0 and 0.7, 23,984 rows) gives the same numbers within 0.2 pp (DEV K = 8: 86.12).


## 4. Adaptive K

Adaptive rule: draw k0 samples; stop if the winning answer holds at least a share `thr` of the (probe-)weight of the sane
answers; otherwise grow to 8, then 16. Simulated exactly from the dumped samples in their drawn order (samples are
independent, so the first n of 16 *are* an n-sample run), one realisation per question, so these are noisier than the
fixed-K table. "x" = mean samples drawn, i.e. tokens / one answer. Config chosen on DEV, read on DEV2.

| rule | DEV acc | DEV x | DEV2 acc | DEV2 x | fixed-K reference (DEV / DEV2) |
|---|---|---|---|---|---|
| probe vote + sanity, k0 = 4, thr 0.75 (**DEV best below 8 samples**) | **86.67** | 6.0 | **85.00** | 6.3 | K = 8: 86.11 / 84.36 |
| probe vote + sanity, k0 = 2, thr 1.0 (stop if 2 agree) | 84.33 | 4.6 | 84.67 | 4.7 | K = 4: 84.19 / 83.23 |
| majority + sanity, k0 = 4, thr 0.6 | 85.67 | 5.4 | 85.67 | 5.8 | K = 8: 85.09 / 83.93 |
| greedy first; sample only if greedy's answer is impossible (7, then 15 samples, probe vote + sanity) | 82.50 | 1.8 | 80.33 | 1.7 | greedy 80.33 / 77.00 |
| greedy + 1 sample; stop if they agree, else 8, 16 (probe vote + sanity) | 85.00 | 4.0 | 83.33 | 4.7 | |
| greedy + 2 samples; stop if all agree, else 8, 16 (probe vote + sanity) | 85.83 | 5.8 | 84.67 | 6.6 | |

- Adaptive K saves about a quarter of the samples at the same accuracy as fixed K = 8 (86.7 at 6.0 vs 86.1 at 8 on DEV;
  85.0 at 6.3 vs 84.4 at 8 on DEV2). It does not move the accuracy frontier; it moves along it.
- The cheapest useful point is the **sanity-gated vote**: keep the greedy answer unless it is impossible, and only then
  draw samples and vote. +2.2 (DEV) / +3.3 (DEV2) at 1.7-1.8x tokens. Compare the sanity implant's in-decoder restart:
  +1.1 on test at 1.08x (`dig/sanity_implant.md`); voting over fresh samples beats one restart because a single restart
  re-derives the same wrong reading 23/75 times.


## 5. Temperature and GSM-Hard-like check

**T = 0.7 vs T = 1.0** (same rules; probe fit on split C at both temperatures):

| set, K = 8 | majority | maj + sanity | probe vote | probe vote + sanity | pv+san - greedy |
|---|---|---|---|---|---|
| DEV, T = 1.0 | 83.88 | 85.09 | 85.74 | **86.11** | +5.8 |
| DEV, T = 0.7 | 82.93 | 84.22 | 85.21 | 85.52 | +5.2 |
| DEV2, T = 1.0 | 81.99 | 83.93 | 83.77 | **84.36** | +7.4 |
| DEV2, T = 0.7 | 82.03 | 83.71 | 83.60 | 84.27 | +7.3 [4.5, 10.2] |

T = 0.7 is equal or slightly worse at every K (DEV K = 16: 86.33 vs 87.17). The RL host's T = 1 samples are already
as good as greedy (DEV sample-0 78.6 vs greedy 80.3), so the extra diversity of T = 1 is free. Keep T = 1.0.

**GSM-Hard-like slice (devS).** GSM-Hard itself is a test set, so I used 300 split-B *scaled-number* copies (the RL
pool's "scaled" kind; large operands, and per the failure analysis about 40% of their golds are negative or
non-integer, many of them noise). Greedy 60.0.

| devS, K | majority | maj + sanity | probe vote | probe vote + sanity | probe select | log-prob vote |
|---|---|---|---|---|---|---|
| 4 | 63.38 | 61.46 | 62.22 | 61.67 | 60.00 | 63.88 |
| 8 | **64.90** | 61.77 | 62.94 | 61.79 | 59.04 | 64.81 |
| 16 | **65.33** | 61.33 | 62.33 | 61.00 | 57.67 | 65.00 |

- Majority vote transfers: +4.9 [1.7, 8.1] over greedy at K = 8.
- The sanity filter costs -3.1 [-5.1, -1.5]: the GSM8K prior (non-negative integer) is false here, as the sanity
  implant found on GSM-Hard.
- The probe does not transfer: within-question AUROC 0.584 vs log-prob 0.651, probe vote -2.0 [-3.8, -0.2] vs majority.
  Same as the earlier base-host study on GSM-Hard (`verifier/NOTEBOOK.md`: probe 0.66 vs log-prob 0.67). The L18 probe
  reads "this looks like a typical correct GSM8K-scale solution", not "the arithmetic is right".
- So the probe and the sanity filter are in-distribution parts. Plain majority is the out-of-distribution-safe part.


## 6. Examples (DEV, first 8 samples, probe vote + sanity vs greedy)

Answers of the 8 samples, with the probe's p(correct) under each:

```
FIX  DEV #53  gold 3     greedy 10 (wrong)
     answers 10 | -28 | -11 | 3 | -11 | 3 | -11 | -10
     p       .02 | .02 | .00 |.55 | .01 |.60 | .00 | .01   -> sanity drops the negatives; 3 wins (majority alone: -11)
FIX  DEV #76  gold 3     greedy 3.47619
     answers 3 | 3 | 3.47619 | 3.47619 | 3 | 3.47619 | 3 | 3
     p      .72 |.78 | .15 | .11 | .76 | .09 | .74 | .78        -> the non-integer reading is both flagged and low-p
FIX  DEV2 #24 gold 405   greedy 225
     answers 400 | 405 | 100 | 405 | 75 | 405 | 405 | 380   -> plain majority already finds it
BREAK DEV #20 gold 1     greedy 1 (right)
     answers 2 | 2 | 1.333 | 1 | 1.333 | 1 | 2 | none
     p      .78 |.86 | .19 |.78 | .22 |.59 |.84 | 0           -> a confident, stable wrong reading (2) out-votes greedy
BREAK DEV2 #29 gold 22   greedy 22 (right)
     answers 22 | 33 x6 | 22                                    -> the samples prefer the wrong reading; nothing can fix this
```
Breaks are of the second kind: the host's own sample distribution puts more mass on a wrong answer than on the greedy
one. 42 fixes / 6 breaks on DEV, 29 / 7 on DEV2 (K = 8).


## 7. Recommendation

**Keep** the selection implant, as the last stage of the stack:
1. **Main setting: K = 8, T = 1.0, probe vote + sanity filter.** DEV +5.8, DEV2 +7.4 over greedy, 8x tokens. If the
   coordinator wants a single test number, this is the config to run (once). Expected on test: the failure analysis saw
   majority@8 add +3.7 on test-300 (72.0 -> 75.7); here majority@8 adds +3.5 / +5.0 and the full rule +5.8 / +7.4, so
   roughly 71.2 -> 76-78 is the prior, not a measurement.
2. **Token-matched variant: adaptive (k0 = 4, thr 0.75)**, same accuracy at about 6 samples on average.
3. **Cheap variant: sanity-gated vote** (greedy unless impossible): +2.2 / +3.3 at 1.7x. Use it instead of the sanity
   restart if the budget is tight.
4. **Larger K** buys little: +1.0 for 16, +0.5 more for 32.

How to stack:
- Selection composes with anything that improves single samples (calculator v2, re-read implants): it votes over
  their outputs. Run those implants inside each of the K rows; the vote sits after them. Gains will overlap where the
  other implant also removes impossible answers (the sanity restart becomes redundant: the filter already drops them).
- The probe reads the host's layer-18 state, so **refit it if the host or the in-row implants change** (any change
  to what the rows compute). Refit = 12k samples on split C (about 14 min on one H100 at B 64) + a 1-minute logistic fit;
  `analyze.py` does it and writes `verifier_probe_L18.pt` (w, b already folded).
- **Domain switch needed.** On GSM-Hard-like data turn the sanity filter and the probe off and keep plain majority.
  Both encode GSM8K priors (integer answers, GSM8K-typical solution states).
- Not done: an in-graph number reader (the vote reads the value with the scorer's regex on the decoded text), the
  probe dot product inside the graph (computed from the captured state, same math), a base-host replication (the
  earlier base-host study found the same ordering: probe vote +5.2 over majority at N = 8, `verifier/NOTEBOOK.md`).



## 8. Single-call versions (follow-up: one generation stream, forward-only, no cross-sample voting, no rewind)

The rule changed: an implant must act inside ONE generation stream. K-sample voting (sections 2-7) is therefore out;
it remains the reference for what independent samples are worth. Three single-call designs were built and measured on
the RL+calc host. **None of them helps. The reason is the same in every case: once the host sees its own earlier
attempt in context, it copies it.**

### 8a. Serial tries + answer register + voter (`selection/serial_engine.py`, `serial_hook.py`)
- **Writer.** When an attempt's `#### X` line closes, or the host would end the answer, or the attempt reaches 320
  tokens, the end token is replaced by `["\n"] "Wait, let me solve it again.\n"`. The host writes the next attempt
  in the same context. This repeats until there are N attempts.
- **Register.** An exact per-row automaton reads each attempt's first `#### X` (it implements the scorer's regex;
  Qwen writes one digit per token). It stores the value, an impossible-answer flag, the L18 probe logit at that
  `####`, and the answer's own tokens.
- **Voter.** It picks a slot (count, probe-weighted, or last; impossible answers dropped first) and writes
  `["\n"] "####" + that slot's tokens + eos`. Only that final line is graded: the last `#### number` in the stream.
- **Exactness checks.**
  - The in-stream final answer equals the offline reduction over the logged register on 600/600 DEV and 300/300
    DEV2 rows.
  - The first attempt of the N=4 stream is the same text as the N=1 run. N=2 and N=3 are read from the first 2 or 3
    attempts of the N=4 stream, because attempt k does not depend on N.
  - Baselines are N=1 under the same engine settings (B 128, S_max 1600). bf16 batch noise makes those differ from
    the B 64 plain greedy run by -0.7 (DEV) and +1.0 (DEV2).

Accuracy (probe-weighted vote + sanity filter; the count vote is equal or lower):

| set, decoding | N=1 | N=2 | N=3 | N=4 | oracle any-of-4 | tokens / item N=1 -> 4 | items whose 4 attempts all give the same answer |
|---|---|---|---|---|---|---|---|
| DEV greedy | 79.50 | 80.17 | 80.33 | 80.33 | 81.33 | 134 -> 436 | 93.8% |
| DEV T=0.7 | 79.00 | 79.33 | 79.33 | 79.17 | 80.00 | 134 -> 431 | 95.0% |
| DEV2 greedy | 78.00 | 78.33 | 78.00 | 78.67 | 79.33 | 137 -> 454 | 94.3% |
| DEV2 T=0.7 | 80.00 | 80.33 | 80.33 | 80.33 | 81.00 | 136 -> 443 | 95.0% |

- **Diversity is the problem.** Attempts 2-4 repeat attempt 1's answer on 93.7% of DEV items, and usually its text
  word for word (with T = 0.7 too). The oracle rises only from 79.5 to 81.3 over four attempts. Four *independent*
  samples with the same voter reach 84.2 (section 3, K = 4 probe vote + sanity), at the same token cost.
- **Matched compute.** Plain greedy with the same 4x token budget is plain greedy: 0.3% of rows hit the 320 cap. So
  the matched-compute baseline is about 80.3 / 77.0. The serial voter is within +0.8 of it.
- **The probe degrades on copied attempts.** Within-attempt AUROC is 0.835 / 0.814 / 0.723 / 0.663 for attempts 1-4.
  A copied attempt sits in a different state from a fresh one.

### 8b. Verifier-triggered rethink (`selection/ver_engine.py` circuit; measured with `rethink_proxy.py`)
- **Trigger.** It fires when the L18 probe logit at `####` is below a threshold, or when the answer is impossible.
  The threshold was set on split C only (5-fold out-of-fold logits on split-C greedy answers; AUROC 0.840).
  - At target precision 0.6 the threshold is logit < -0.042. On split C that fires 7.7% probe-only and 12.5%
    combined, with combined precision 0.76.
  - The repo's earlier rethink detector fired on 68% of answers. This one fires on about 13%, and 71-76% of those
    answers really are wrong.
- **Act.** It writes "Wait, let me check that. We need to find" after the `#### X` line, and the host continues.
- **How it was measured.** The rows where it fires were measured by an exact token-stream proxy. The prefill is the
  host's own greedy attempt up to the end of its `#### X` line, plus the phrase, and greedy continuation follows.
  Under greedy decoding this is the stream the implant produces; rows where it does not fire keep the greedy answer.

| set | trigger | fire rate | precision (fired & wrong) | fixes | breaks | answer unchanged after the phrase | acc (base) | extra tokens / item |
|---|---|---|---|---|---|---|---|---|
| DEV | sanity only, typed phrase | 4.5% | 1.00 | 1 | 0 | 17 / 27 | 80.50 (80.33) | +6 |
| DEV | probe or sanity, "let me check" | 12.8% | 0.71 | 2 | 3 | 53 / 77 | 80.17 (80.33) | +12 |
| DEV | unused question number, named hint | 4.8% | 0.41 | 2 | 2 | 23 / 29 | 80.33 (80.33) | +6 |
| DEV | all three, typed phrases | 16.8% | 0.61 | 4 | 5 | 73 / 101 | 80.17 (80.33) | +19 |
| DEV2 | sanity only | 4.3% | 1.00 | 0 | 0 | 12 / 13 | 77.00 (77.00) | +5 |
| DEV2 | probe or sanity | 13.7% | 0.76 | 1 | 2 | 33 / 41 | 76.67 (77.00) | +13 |
| DEV2 | unused number | 6.0% | 0.33 | 0 | 0 | 17 / 18 | 77.00 (77.00) | +8 |
| DEV2 | all three | 18.0% | 0.63 | 0 | 2 | 48 / 54 | 76.33 (77.00) | +20 |

- **The detector is good now.** At 13% fire rate it has 71-76% precision, against the old detector's 68% fire rate
  and no fixes.
- **The repair is not.** After the phrase the host restates the same answer in 69-89% of fired rows. Fixes and
  breaks cancel. This matches the sanity study, where the phrase gained +0.5 and a KV restart gained +1.1. Only a
  fresh context changes the answer.

### 8c. Failure diagnosis (30 DEV items the voter cannot help: greedy wrong and at most 1 of 16 independent samples right)
DEV has 118 greedy failures; 50 of them are this "hard core". These are 29 of them, read by hand
(`selection/results/hard_failures_dev.txt`). Each is listed by its first wrong step and the side implant that could
fix it:

| class | n | items | candidate implant | verdict |
|---|---|---|---|---|
| wrong plan / misread the situation | 14 | 170 88 217 36 533 254 215 445 117 225 374 29 213 530 | none small: needs a better reader | no implant |
| dropped a stated quantity or multiplier | 5 | 107 (x20 birds) 386 (split with friend) 444 90 361 | "unused number" checklist register + named hint | built (8b): 41% precision, 2 fixes / 2 breaks |
| right steps, wrong reference in the last step | 2 | 564 (N+5 for "sum now"), 240 (70-28 instead of 56-28) | operand-source tracker | not built (2/30) |
| unit / world knowledge | 4 | 317 (quarts, oz) 136 (couples = 2) 432 (pairs) 6 | unit-fact table, like the calculator | not built: 2 of 4 are unit facts |
| relation / direction | 3 | 598 412 216 | all 3 already flagged by the sanity filter | the sanity repair changes 0/3 |
| gold or answer-unit ambiguity | 2 | 567 ($3.50 vs 350 cents), 488 | none | |

- About half of the hard core is plan-level misreading. No side function fixes that.
- The quantity-checklist implant (the best-grounded candidate) was built and measured (8b): +0 on DEV and DEV2.
- An activation-steering retry (push the retry away from the first attempt) was not built. 8a shows the retry copies
  the attempt's text verbatim through attention to the context, and a residual steering vector does not remove that
  copy path. The one change that does is to take the attempt out of attention (the sanity restart, or K-sample voting),
  and that is exactly what the single-call rule forbids.

### 8d. Two more single-call variants (DEV only, since neither moved DEV)
- **Retry phrase that asks for a new method** ("Wait, let me solve it again using a different method."), T = 1.0
  attempts, N = 4. Diversity rises a little: all 4 attempts agree on 88% of items (94% before), and the oracle reaches
  81.0. Final accuracy is 78.33, against 77.17 for N = 1 at T = 1, and still below greedy 80.3. With greedy attempts it
  gives 80.33, the same as the plain phrase.
- **Probe-adaptive serial (N <= 2).** It retries only if the answer is impossible or the probe logit is below -0.042
  (split-C threshold), then votes. It retried on 14% of items and gave 79.67, equal to its N = 1 baseline, at +14%
  tokens.

### 8e. Verdict for the single-call rule
- **Drop all single-call selection and rethink implants for the stack.** The serial-tries voter (+0.7 to +0.8 at 3.3x
  tokens, within noise), the verifier-triggered rethink (0 to -0.3), the unused-number hint (+0) and the adaptive
  serial (+0) all fail for one reason: in-context retries copy.
- **What is still worth having:** the L18 probe. It is a calibrated in-graph detector: 13% fire rate, 71-76%
  precision, split-C AUROC 0.84. It could gate a costlier act if a later implant finds an act that changes the
  answer. Under the old rules that act was a fresh context (independent samples: +5.8 / +7.4 at K = 8).
- **A note on grading:** the serial voter writes its answer in the final `####` line, so a scorer must read the LAST
  `#### number`, not the first.


## 9. Serial tries + forget mask + register + voter: the vote gain comes back in one stream

**Change from section 8a:** when the writer starts attempt k ("Wait, let me solve it again."), every earlier attempt
is hidden from attention. Attempt k then sees only the 8-shot prefix, the question, the phrase and itself. Two
registers do this:
- **Attempt register (forget).** Each cached key carries the attempt label it was written in; question and prompt
  keys carry none. The query of attempt k adds -M on keys from any other attempt.
  - It was run first with the reading agent's masked suffix-attention kernel (key range [question end, attempt start)
    hidden).
  - Then it was run with the stack agent's weight-space version (`final/code/forget_ws.py`: E extra, un-rotated q/k
    dims per head, W_K = I, W_Q = -M (J - I), M = 1e4). I extended it to E = 8 attempts and ported it into
    `selection/serial_engine.py` (`SER_WS=1`), with a rowid indirection because this engine compacts rows.
- **Position register (needed; new).** With the mask alone, attempt k's RoPE positions keep counting past the hidden
  attempts. The question then sits hundreds of positions back, and quality decays attempt by attempt. On DEV2 at
  T = 1 the per-attempt accuracy was 76 / 61 / 58 / 52 / 43 / 35 / 36 / 25, and N = 8 gave only 78.3. The fix gives
  the new attempt positions that continue from the question end: pos = slot - (hidden span), with the span held in a
  per-row register. Per-attempt accuracy is then flat (76-79% for all 8 attempts) and the probe stays calibrated
  (AUROC 0.81-0.90 on every attempt).
- The writer, answer register and voter (probe-weighted vote + sanity filter) are unchanged from 8a. Sampling is
  T = 1.0 or 0.7 inside every attempt. N = 1/2/4 are the first 1/2/4 attempts of the N = 8 stream (attempt k does not
  depend on N).

**Exactness.**
- The weight-space forget (attempt labels on extra q/k dims) is **token-identical** to the masked kernel: 300/300 DEV2
  rows, 8 attempts each (1,141 tokens per item; same GPU and batch).
- The in-stream final `####` equals the offline vote over the logged register on 600/600 and 300/300 rows.

**Results** (RL+calc host, no calc v2; position register on; A100, batch 128). "All-same" = items where all N
attempts give one answer. The independent-sample references are from section 3 (T = 1, same voter): K-vote accuracy,
and the all-same rate of K independent samples on DEV is 83.2 / 72.3 / 63.5 at K = 2 / 4 / 8.

| set, T | N=1 | N=2 | N=4 | N=8 | oracle N=8 | tokens / item (N=1/2/4/8) | all-same N=2/4/8 | independent K-vote (K=2/4/8) |
|---|---|---|---|---|---|---|---|---|
| DEV, T=1.0 | 79.00 | 81.67 | 85.17 | **88.50** | 92.83 | 133 / 279 / 561 / 1121 | 77.3 / 65.7 / 56.8 | 82.02 / 84.19 / 86.11 |
| DEV, T=0.7 | 78.17 | 82.17 | 84.50 | 85.83 | 91.33 | 133 / 279 / 561 / 1122 | 78.5 / 67.2 / 63.0 | 81.98 / 83.88 / 85.52 |
| DEV2, T=1.0 | 77.00 | 80.00 | 84.00 | **87.00** | 90.67 | 136 / 282 / 567 / 1136 | 77.3 / 64.7 / 56.3 | 80.36 / 83.23 / 84.36 |
| DEV2, T=0.7 | 78.00 | 81.33 | 82.33 | 85.00 | 92.00 | 136 / 285 / 570 / 1146 | 77.0 / 68.3 / 60.3 | 81.06 / 83.16 / 84.27 |
| DEV2, T=1.0, weight-space forget (batch 32) | 78.67 | 80.33 | 83.67 | 85.67 | | 136 / 285 / 571 / 1140 | 78.0 / 64.3 / 56.3 | |

Against plain greedy (DEV 80.33, DEV2 77.00; paired bootstrap; fixes / breaks):

| | N=4 | N=8 |
|---|---|---|
| DEV, T=1 | +4.83 [2.16, 7.50], 50 / 21 | **+8.17 [5.67, 10.83], 59 / 10** |
| DEV2, T=1 | +7.00 [2.67, 11.33], 34 / 13 | **+10.00 [5.67, 14.33], 38 / 8** |
| DEV, T=0.7 | | +5.50 [3.00, 8.17], 49 / 16 |
| DEV2, T=0.7 | | +8.00 [4.00, 12.00], 31 / 7 |

- **The vote gain is back under the single-call rule.** At equal sample count, serial + forget matches the
  independent K-sample vote: N = 8 gives 88.5 / 87.0 vs K = 8 at 86.1 / 84.4. The N = 8 row is one realisation per
  item and the K row is an average over subsets, so read them as "equal within noise", not "better". The all-same
  rate falls to the independent level and slightly below it (56.8 vs 63.5 at 8 on DEV), so copying is gone.
- T = 1.0 is better than 0.7 again (+2.7 / +2.0 at N = 8).
- N = 2 is weak (+1.3 / +3.0, not significant): with two tries the vote is mostly a coin flip between them. The
  useful points are N = 4 (4.2x tokens, about +5 / +7) and N = 8 (8.3x tokens, +8 / +10).
- The batch-32 weight-space run gives 85.67 on DEV2 vs 87.00 at batch 128. The token streams differ only through
  bf16 batch numerics in sampling; at equal batch the weight-space and kernel runs are token-identical. The spread
  (±1.3) is the realisation noise of a single N = 8 run.
- **Compared with the final stack S2F** (FINAL.md: DEV 83.17 / DEV2 81.00 at about 146 tokens): serial + forget at
  N = 4 is 85.17 / 84.00 (and 84.50 / 82.33 at T = 0.7) at about 4x tokens; at N = 8 it is 88.50 / 87.00 at about 8x.
  It is a compute-for-accuracy knob on top of the single-sample stack, not a free gain.
- **Not done: serial + forget on top of S2F** (calc v2 inside every attempt). calc v2 is a Python-side row tracker
  in its own `_wave` loop (`C2Engine`), which does not compose with this engine's loop without a port. The two
  should add roughly: calc v2 is +0.6 per sample, and S2F's sanity restart is subsumed by the voter's sanity filter
  plus fresh attempts. To get the number, the stack agent should port `SerialEngine._act` / `_read` and the position
  register into StackH.
- **Disclosed non-weight pieces** (as in reading.md §7):
  - the decode loop and per-attempt token cap (320);
  - registers held as indices or values between steps;
  - the position register enters RoPE as an integer offset;
  - the answer register is an exact automaton over token ids, the same kind of object as the calculator's reader.
- Files: `selection/serial_engine.py` (writer, answer register, voter, forget, position register, weight-space
  forget), `serial_hook.py` (`SER_FORGET=1 SER_POSFIX=1 [SER_WS=1] SER_NMAXGEN=8 SAN_SMAX=2960`),
  `ser_analyze.py`, `ser_ci.py`. Dumps: `A100:<work>/gsm_sel/out/ser_*_fmp.pt`, `*_ws.pt`, `*_mask32.pt`.
- **Grading note:** the answer is the LAST `#### number` in the stream (the voter's line).

## Files
- Code: `gsm_push/selection/` (`probe_engine.py`, `sel_hook.py` exec'd inside
  `dig/sanity/tw_dump_lean.py --eval_ckpt`, `run_sel.sh`, `analyze.py`, `check_cap.py`).
- Results (json + analysis logs + fitted probe): `gsm_push/selection/results/`; final table =
  `sel_final_T1_K32.json` (+ `sel_T1_trainT1.json` for the 16-sample tables, `sel_devS.json`, `sel_dev2_T07.json`).
- Dumps + probe: `H100:<work>/gsm_push/selection/out/` (`*_G*_s*.pt`, `verifier_probe_L18.pt`, `sel_*.json`),
  checkpoint copy at `H100:<work>/gsm_push/selection/ckpt_a2s2` and `A40:<work>/gsm_sel/ckpt_a2s2`
  (sha256 matches A100). Compute: all generation ran on H100 GPU1 before 08:30 CT in free memory
  (mem cap 0.2, own watchdog at 6 GB free); A40 was not needed and no job on it was touched.
