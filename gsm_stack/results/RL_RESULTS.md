# RL with the all-weights GSM8K stack — results (the stack lane, task [6] [7] [8] [9] [10])

> Release note: working notes as written during the runs (2026-11-20/21), with cluster paths mapped to this
> repository (`gsm_stack/...`) and host names replaced by GPU types. "task [n]" refers to the item numbers of the
> original task list; `j23xxx` are job-queue ids (labels only). Labeling slips found at release are listed in
> `gsm_stack/COMPLIANCE.md` (last section).

Host for everything: Qwen3-0.6B-Base after 300 GRPO updates with the calculator (a2_u300_s2), except arm C (base host).
Stack (all weights, one generation call): calc_s4 + chain2 ({+,-}) calculators, compiled answer-sanity DFA + pointer
phrase writer, weight-space forget mask (4 fixed q/k dims per head), and for the serial arms N tries in one stream with a
compiled answer register, L18 probe voter and final writer. **No position reset** except arm Ar (marked NON-COMPLIANT).
Decoding T = 1, seed 7 (test), dev seed 4242; grading = last number. CIs: paired bootstrap, 4000 resamples, 95%.
Code: eval/trees/v42 (arms), engine (new arms, exact vs v4.2), eval/ (test harness; frozen v4.2 copy).

## 1. Arms and dev curves (dev = 600 GSM8K-train items, dev2 = 300 more; never test)

| arm | what | host | prompts | N (train) | seed |
|---|---|---|---|---|---|
| A | group-relative reward = final vote | a2_u300_s2 | 64 × G8 | 8 | 3 |
| A1 (A') | as A + per-attempt advantage | a2_u300_s2 | 64 × G8 | 8 | 3 |
| Ar | as A + programmatic RoPE position reset per attempt — **NON-COMPLIANT, reference only** | a2_u300_s2 | 64 × G8 | 8 | 3 |
| B | S2F single attempt (token-matched control) | a2_u300_s2 | 256 × G8 | 1 | 1 |
| C | as A, base host | Qwen3-0.6B-Base | 64 × G8 | 4 | 5 |
| A1D | as A1 + diversity term (−0.5 on a wrong try repeating an earlier answer when the vote is wrong) | a2_u300_s2 | 64 × G8 | 8 | 0, 1 |
| Bb ([13a]) | as B (one-attempt S2F stack), base host | Qwen3-0.6B-Base | 256 × G8 | 1 | 5, 6, 7 |
| P ([14]2) | plain RL, NO implants (calc / chain2 / compiled triggers off), stop eos / blank line / 320 tok | Qwen3-0.6B-Base | 256 × G8 | 1 | 1, 2 |
| C600 | C continued u300 → u600 (fresh AdamW, warmup 5) | C u300 | 64 × G8 | 4 | 5 |
| CW600 ([14]4) | CW_s0 continued u300 → u600 (WPR=1, fresh AdamW) | CW_s0 u300 | 64 × G8 | 4 | 0 |

Dev vote8 (dev / dev2) for A, A1, Ar; vote4 for C; single-attempt acc for B:

| u | A | A1 | Ar (NON-COMPL.) | B | C (vote4) |
|---|---|---|---|---|---|
| 0 | 81.33 / 78.00 | 81.33 / 78.00 | 88.83 / 86.67 | 82.00 / 82.00 | 36.83 / 35.33 |
| 100 | 83.83 / 82.67 | 85.17 / 83.00 | 86.67 / 86.67 | 82.50 / 82.67 | 72.50 / 68.33 |
| 200 | 80.67 / 78.67 | 86.50 / 85.67 | 87.67 / 86.00 | 82.00 / 83.67 | 82.17 / 82.33 |
| 300 | 83.00 / 81.67 | **86.67 / 85.00** | 85.67 / 85.00 | 80.83 / 81.33 | 84.33 / 83.00 |

Full per-25 curves: `python3 eval/results_tools/devtab.py A=the v4.2 arm runs/arm_A/steps.jsonl ...` (A, A1, Ar, B in
rel/v4.2/out, C in rel/v4.1/out). Attempt-8 accuracy (how good the 8th try is without reset): A 29 → 57.5, A1 29 → 70.7;
all-same rate at N = 8: A 9 → 48, A1 9 → 54.

**[6](b) choice, made on dev before any test item was touched (STATUS 02:30):** mean of dev / dev2 at vote8 and vote4 at
u300: A 82.17, **A1 85.38** → A1 is the stack arm that goes to test.

### Seeds (task [10]/[14]; curves: figures/rl_dev_curves.png; data: eval/results_tools/steps/ARM.jsonl via the steps copies in results/curves/)

| arm | seed | u300 dev / dev2 (vote8; B acc) | all-same@8 at u300 | try-8 acc at u300 | note |
|---|---|---|---|---|---|
| A | 3 | 83.00 / 81.67 | 48.3 | 57.5 | |
| A_s2 | 4 | 79.83 / 78.33 | **99.2** | **0.3** | **collapse:** from u225 all 8 tries give the same answer and try 8 is empty; peak was 85.17 at u25 |
| A_s3 | 6 | **77.00 / 73.67** | 37.8 | – | **late decline, no collapse:** tracks A to u250 (vote8 84.8 at u125, 81.8 at u250; all-same 44–47%), then vote8 −4.8 / dev2 −5.0 at u300. vote1 slips 79.3 → 77.2 and oracle8 91.2 → 87.7; all-same *falls* to 38%. Dev-only seed (A not dev-selected) |
| A1 | 3 | 86.67 / 85.00 | 54.3 | 70.7 | |
| A1_s2 | 4 | **86.00 / 85.67** | 51.2 | 65.3 | replicates A1 (86.67 / 85.00); test/hard j23371 |
| A1_s3 | 6 | 85.00 / 83.67 | 52.3 | 67.7 | test N=8 79.68 / N=4 78.47 / Hard 62.47 (j23493); 3-seed A1: test 80.06 (sd 0.59), Hard 63.31 (sd 0.92) |
| A1D_s0 (RL_DIV 0.5) | 0 | 85.83 / 84.00 | 45.3 | 64.8 | diversity penalty lowers all-same (45 vs 51) but not acc; test 80.36 / hard 63.08 (§2) |
| A1D_s1 (RL_DIV 0.5) | 1 | 87.17 / 83.00 | 52.2 | 66.5 | test 80.36 / hard 63.15 (§2) |
| B | 1 | 80.83 / 81.33 | – | – | |
| B_s2 | 2 | 82.83 / 80.33 | – | – | test/hard (compiled trigger) j23371 |
| B_s3 | 3 | 82.67 / 80.67 | – | – | flat 81–84 from u25 like B/B_s2; test 73.62 / Hard 57.62 (seed_Bs3.list, j23538); 3-seed B: test 73.94 (sd 0.31), Hard 57.82 (sd 0.49) |
| C_s3 | 6 | 83.83 / 82.00 (vote4) | – | 70.8 (try 4 = att 4) | replicates C (84.33 / 83.00); test 75.13 / Hard 52.39 (§10) |
| C_s2 | 7 | 81.17 / 82.00 (vote4) | – | 68.3 (try 4) | test 74.68 / Hard 51.86 (§10) |
| C600 (C u300 → u600) | 5 | u600: 83.00 / 79.00 (start 84.83 / 83.33) | 66.8 (vote4) | 67.2 (try 4) | **plateau:** C's dev is flat from u300 (82.2-85.2 at every 25). Test 75.06 [72.71, 77.41], +0.45 [−1.67, 2.58] vs C u300; GSM-Hard 55.12, +0.45 [−1.74, 2.65]. More updates give nothing |
| CW600 (CW_s0 u300 → u600) | 0 | u600: 85.67 / 81.33 (start 87.17 on dev) | | | **plateau too:** dev vote4 is 83.5–87.5 at every 25 from u325 and never above the start. Test 77.79, +0.61 [−1.29, 2.50] vs CW_s0 u300; GSM-Hard 55.65, +1.97 [−0.15, 4.09] (n.s.). [14]4 answer: both C and C+WPR are saturated by u300 |

With the final-vote reward alone (A), RL is unstable across seeds. At u300, dev / dev2 vote8 is:
- seed 3: 83.00 / 81.67
- seed 4: 79.83 / 78.33, collapsed into one repeated try
- seed 6: 77.00 / 73.67, late decline without collapse

Only seed 3 ends above u0 (81.33 / 78.00) on both dev sets. The attempt-level credit of A1 holds on all 3 seeds (86.67 / 86.00 / 85.00 on dev). This is the reason A1, not A, is the headline arm.

**Caveat for "base 37 → 84" (C):** the 37 is the base host *with the stack attached* at u0, on dev. On TEST
(base.list j23321, T=1 seed 7):
- base host alone, no implants: 28.05;
- + calc/chain2/sanity: 28.66;
- + S2F stack: 28.43;
- + serial N=4: 28.43.

So the stack does not hurt the untrained base on test, and the base starts at ~28 test, about 37 dev.
These 28s are for our zero-shot chat format (T=1, 320 tokens). Under the Qwen3 report protocol the same base model is
much stronger. That protocol is OpenCompass 4-shot CoT, greedy, 512 tokens (bc/gsmevals j23369; the report says 59.59):
- no implants: **56.48** [53.83, 59.14];
- with the calc implant: 56.63 (+0.15 n.s.).

So "base → RL" gains measured in our format overstate RL relative to a few-shot base. The fair base for the blog is
~56 (4-shot) next to the RL hosts' zero-shot ~75–81. My earlier note of
"50–62 on a 40-item P smoke" was a 40-item training-batch reading. It is not a dev number, and is superseded.
The P arm (no implants, from base, 2 seeds) and B_base give the split of the final number into:
- RL alone;
- RL + tools in one attempt;
- RL + the serial stack.

Their u0 dev evals give the same-harness base dev number.

## 2. Test (GSM8K test, 1319 items) — each config run once

| config | test acc [95% CI] | Δ vs u0 same N [CI] | fixes / breaks | tok/q | oracle | all-same | attempt acc 1 → N |
|---|---|---|---|---|---|---|---|
| u0 S2F (B stack, 1 attempt; missing trigger = swap hook; compiled trigger: 72.63, §6) | 72.25 [69.75, 74.68] | – | | 151.6 | | | |
| u0 serial N=4 | 72.02 [69.52, 74.45] | – | | 586.0 | 81.20 | 32.9 | 71.5 → 45.8 |
| u0 serial N=8 | 71.87 [69.37, 74.15] | – | | 1093.8 | 85.52 | 6.5 | 72.9 → 22.7 |
| **A1 u300 N=4** | **79.23** [77.03, 81.35] | **+7.20** [5.00, 9.33] | 151 / 56 | 613.5 | 85.52 | 61.6 | 73.2 → 70.6 |
| **A1 u300 N=8** | **79.76** [77.56, 81.88] | **+7.88** [5.91, 9.86] | 145 / 41 | 1222.7 | 89.08 | 48.1 | 73.8 → 65.7 |
| B u300 (S2F, RL, 256 prompts; swap hook; compiled trigger: 74.22, §6) | 73.84 [71.42, 76.12] | +1.59 [−0.53, 3.64] vs u0 S2F | 117 / 96 | 141.9 | | | |
| C u300 N=4 (base host) | 74.60 [72.18, 76.95] | +2.58 [0.30, 4.85] vs u0 N=4 | 128 / 94 | 545.3 | 82.26 | 58.0 | 66.3 → 62.7 |
| base host alone, no implants (ladder H on base) | 28.05 | | | 102.6 | | | |
| base host + calc + chain2 + sanity (ladder D on base) | 28.66 | | | 101.2 | | | |
| base host + S2F stack (1 attempt) | 28.43 | | | 113.3 | | | |
| base host, no RL, N=4 | 28.43 [25.93, 30.93] | | | 691.1 | 42.68 | | 29.9 → 6.4 |
| *u0 N=8 + position reset (NON-COMPLIANT)* | *80.97* [78.77, 83.09] | *+9.10 vs u0 N=8* | | 1273.5 | 90.14 | 52.4 | 73.9 → 71.7 |
| *Ar u300 N=8 + reset (NON-COMPLIANT)* | *77.79* [75.51, 79.91] | *−3.18 vs u0+reset* | | 1192.4 | 88.1 | 52.1 | 71.6 → 68.8 |


**Seeds at test** ([14]3; T=1 seed 7, once each; paired Δ vs the same-N u0 config; B rows use the compiled missing trigger, vs compiled u0 S2F 72.63 / 56.03):

| arm | RL seed | test | Δ vs u0 [CI] | GSM-Hard | Δ vs u0 [CI] |
|---|---|---|---|---|---|
| A1 N=8 | 3 | 79.76 | +7.88 [5.91, 9.86] | 63.15 | +5.61 [3.56, 7.66] |
| A1 N=8 | 4 | **80.74** [78.62, 82.79] | **+8.87 [6.82, 10.99]** | **64.29** [61.64, 66.87] | +6.75 [4.78, 8.79] |
| A1D N=8 (diversity 0.5) | 0 | 80.36 [78.24, 82.49] | +8.49 [6.44, 10.54]; vs A1 s3 +0.61 [−1.06, 2.35] | 63.08 [60.42, 65.66] | +5.53 [3.34, 7.73]; vs A1 s3 −0.08 |
| A1D N=8 (diversity 0.5) | 1 | 80.36 [78.17, 82.49] | +8.49 [6.44, 10.54]; vs A1 s3 +0.61 [−1.14, 2.27] | 63.15 [60.58, 65.66] | +5.61 [3.56, 7.73]; vs A1 s3 0.00 |
| A1 N=8 | 6 | 79.68 [77.48, 81.80] | +7.81 [5.76, 9.93] | 62.47 [59.74, 65.13] | +4.93 [2.81, 7.05] |
| **A1 N=8, 3 seeds pooled** | 3/4/6 | **80.06** (sd 0.59) | **+8.19 [6.39, 10.01]** | **63.31** (sd 0.92) | **+5.76 [4.02, 7.61]** |
| A1 N=4 | 3 | 79.23 | +7.20 [5.00, 9.33] | 62.09 | +6.44 [4.40, 8.42] |
| A1 N=4 | 4 | 79.08 [76.88, 81.20] | +7.05 [4.93, 9.17] | | |
| A1 N=4 | 6 | 78.47 | | | |
| B S2F | 1 | 74.22 | +1.59 [−0.61, 3.79] | 57.47 | +1.44 [−0.68, 3.49] |
| B S2F | 2 | 74.00 [71.57, 76.27] | +1.36 [−0.83, 3.64] | 58.38 | +2.35 [0.30, 4.47] |
| B S2F | 3 | 73.62 [71.11, 75.89] | +0.99 [−1.06, 3.11] | 57.62 [54.97, 60.20] | +1.59 [−0.53, 3.71] |
| **B S2F, 3 seeds pooled** | 1/2/3 | **73.94** (sd 0.31) | +1.31 [−0.51, 3.18] | **57.82** (sd 0.49) | **+1.79 [0.05, 3.46]** |
| C N=4 (u300 / u600 continuation) | 5 | 74.60 / 75.06 | | 54.66 / 55.12 | |

A1's +8 on test over the untrained stack holds across 3 seeds (pooled N=8: 80.06, sd 0.59; +8.19 [6.39, 10.01]; Hard +5.76 [4.02, 7.61]).
Pooled over seeds, A1 vs the untrained WPR rungs:
- test: −0.61 [−2.12, 0.96] vs rung 1; −1.52 [−3.03, −0.03] vs rung 2.
- Hard: +1.14 [−0.40, 2.73] vs rung 1; +0.68 vs rung 2.
- A1 → A1W (WPR inside RL): test −0.94 [−2.12, 0.25], Hard −0.56 (both 3 seeds, n.s.). The diversity penalty (A1D, 2 seeds) does not change accuracy: both seeds give 80.36 test (+0.6 vs A1, n.s.) and 63.1 hard. It lowers test all-same 48.1 → 37.5 / 43.4 at +9–21% tokens per question, because later attempts are weaker (s0 attempt 8: 57.5 vs 65.7). B's single-attempt RL gain is
+1.0 to +1.6 on test per seed and not significant on any seed. Pooled over 3 seeds it is +1.31 [−0.51, 3.18] on test (n.s.) and +1.79 [0.05, 3.46] on Hard (just significant).
Pooled A1 (N=8) − B (1 attempt): test **+6.12 [4.70, 7.56]**, Hard **+5.48 [3.99, 7.03]**.
Reading:
- Without the reset, the serial stack at u0 adds nothing over one attempt (71.9 vs 72.3): later attempts decay
  (72.9 → 22.7 at attempt 8) because the host copies its earlier tries. **RL with per-attempt credit (A1) repairs this
  inside the weights**: attempt 8 goes 22.7 → 65.7 and the vote reaches 79.8, within 1.2 of the programmatic-reset
  number (81.0) that the old 80.4 headline needed.
- C: RL with the full stack from the base host reaches 74.6 at N=4 (base host + stack, no RL: 28.4).
- Ar (reset + RL) is *below* its own u0 on test (77.8 vs 81.0), matching its dev decline (88.8 → 85.7).
- Harness validation (dev, arms' dev seed): u0 81.33 = training log exactly; A1 u300 86.00 vs 86.67 logged; A u300 82.33 vs
  83.00 logged; greedy B u0 82.00 = exact. The u300 gap (−0.67 for both) is not explained (weights are bitwise the saved
  bf16 copy); it is the same sign for both arms, so the A-vs-A1 ranking is unaffected.

## 3. GSM-Hard (1319 items) — serial: plain majority vote, no sanity filter, repair missing answers only (GSM-Hard answers can be negative / non-integer)

| config | GSM-Hard acc [CI] | paired Δ [CI] | tok/q | attempt acc 1 → 8 |
|---|---|---|---|---|
| u0 S2F, missing-only triggers (STK_MISSING_ONLY=1; B has no missing trigger → ≈ chain2 + forget) | 55.19 | – | 174.3 | |
| B u300, same settings | 57.39 [54.74, 60.12] | +2.20 [0.00, 4.32] vs u0 S2F | 161.4 | |
| u0 serial N=8 | 57.54 [54.89, 60.12] | +2.35 [0.38, 4.32] vs u0 S2F | 1236.6 | 56.2 → 15.0 |
| **A1 u300 N=8** | **63.15** [60.50, 65.81] | **+5.61 [3.56, 7.66] vs u0 N=8**; +7.96 [5.69, 10.08] vs u0 S2F | 1440.0 | 58.6 → 50.0 |
| u0 serial N=4 | 55.65 [53.07, 58.30] | – | 689.9 | |
| A1 u300 N=4 | 62.09 [59.51, 64.75] | +6.44 [4.40, 8.42] vs u0 N=4 | 736.6 | 59.4 → 55.5 |
| C u300 N=4 | 54.66 [51.93, 57.32] | −0.99 [−3.34, 1.29] vs u0 N=4 | 703.8 | 47.8 → 44.5 |
| base host alone (no RL, no implants) | 10.46 | | 126.7 | |
| ladder H (host only, no implants) | 39.42 [36.84, 42.08] | – | 174.1 | |
| ladder C4 (+ chain2 + calc) | 53.90 [51.25, 56.56] | +14.48 [12.28, 16.68] vs H | 174.6 | |
| ladder D (+ compiled sanity) | 54.74 [52.08, 57.39] | +15.31 [12.81, 17.74] vs H | 174.9 | |
| *u0 N=8 + programmatic reset (NOT compliant, reference)* | *62.09 [59.44, 64.67]* | *+4.55 vs u0 N=8* | *1629.5* | *55.4 → 54.1* |

On GSM-Hard, the calculator implants give the biggest jump (+14.5 over the host). RL with the stack attached (A1) adds
another +6.4 / +5.6 at N=4 / 8 and matches or beats the non-compliant reset reference with fewer tokens. C (RL'd for the
sanity filter, which is off here) does not transfer to GSM-Hard.

## 4. [8] All-weights implant ladder on TEST (host a2_u300_s2, T=1, seed 7, once per rung)

| rung | test acc [CI] | Δ vs previous rung [CI] | tok/q |
|---|---|---|---|
| no implants (calc off) | 67.70 [65.20, 70.13] | – | 137.2 |
| + calc_s4 | 68.69 [66.11, 71.11] | +0.99 [−0.15, 2.05] | 136.9 |
| + chain2 | 69.45 [66.87, 71.80] | +0.76 (vs H: +1.74 [0.53, 2.88]) | 137.0 |
| + sanity + forget (compiled) = S2F | 72.25 [69.75, 74.68] | +2.80 (vs H: +4.55 [2.58, 6.44]) | 151.6 |
| + retry / register / voter, N=4, no reset | 72.02 [69.52, 74.45] | −0.23 | 586.0 |
| + retry / register / voter, N=8, no reset | 71.87 [69.37, 74.15] | −0.38 [−2.43, 1.74] vs S2F | 1093.8 |
| + RL with the stack attached (A1 u300), N=8 | 79.76 [77.56, 81.88] | +7.88 [5.91, 9.86] | 1222.7 |
| + WPR weight-space position reset, N=8, no RL (rung 1, [17]) | **80.67** [78.47, 82.71] | +8.79 [6.75, 10.84] vs N=8 no reset | 1276.9 |
| programmatic position reset, N=8 (NON-COMPLIANT reference) | 80.97 [78.77, 83.09] | +9.10 [7.13, 11.07] | 1273.5 |
| rung 2: WPR + adaptive tries (A) + stop-at-4 (S) + integer guard (B) + cap 400 (C), all weights, no RL | **81.58** [79.38, 83.70] | +0.91 [−0.45, 2.35] vs rung 1; +9.70 [7.66, 11.68] vs N=8 no reset | 1481.8 (7.87 tries/q) |
| rung 3: rung 2 + Nmax 24 + answer-group voter as registers (AW_VF), all weights, no RL | **82.71** [80.59, 84.76] | +1.14 [−0.15, 2.50] vs rung 2; +10.84 [8.79, 13.04] vs N=8 no reset | 1773.7 (9.37 tries/q) |

The S2F rung here has no missing-answer trigger (arm B's config). [12] item 1 adds the compiled END trigger (section 6).

## 5. [7] Weight-space compliance audit of the all-weights arms

Rule: nothing outside the weights decides anything except the generator (sampling, token budget, stop when the stack
wrote eos). Read: comp_serial (`_sw`, `_step_triton`, SerialW done rule), stackh, forget_ws, attn3, fastroll, c2_engine,
wiring `_book`, rl_hook.

| part | verdict |
|---|---|
| serial registers (answer register, attempt counter, voter, final writer, probe readout) | fixed tables + tensor ops on registers each step, no host sync, no data-dependent Python → compliant (same class as the compiled DFA) |
| SerialW stop rule (done = d0 \| register wrote eos \| budget) | compliant: overrides `_book`'s host stop tokens for the serial stack |
| forget mask (weights mode, 4 extra q/k dims) | compliant; token-identical to the masked kernel (600/600, 300/300) |
| fastroll admission / compaction / harvest, TK flag check | scheduling and diagnostics only (bitwise-identical rows to the plain loop) |
| **first sampled token: row ended if it was eos or contained "\n\n"** (fastroll, c2_engine) | **host stop rule → flagged.** Fired 0 / 600 on dev (every row has 8 attempts). Removed for the serial stack in engine (rel/v5) (new arms). Old arms' rows are unaffected in practice |
| B (StackHD): stop on host stop tokens; budget grows with the fire count | standard generator stop + budget (allowed); disclosed |
| B's missing-answer trigger (m_on: stop-token swap after sampling) | programmatic, but **off** for arm B and all B test rows above; compiled replacement in section 6 |
| Ar: RL_POSRESET (RoPE offset reset per attempt) | **programmatic → NON-COMPLIANT** (reference only) |
| rl_hook reward / advantage / RL_DIV | training-time only (not part of the deployed stack) |

## 6. [12] item 1: S2F all-weights with a compiled END trigger for missing answers — DONE

The original S2F used a post-sampling swap: if the DFA says "no answer yet" and the sampled token is a stop token, replace
it with the first token of " We need to find" and start the phrase writer + forget. Compiled version (eval/trees/v42/stk/
stackh.py `_miss_w`, FE_MISS=w): in the armed register state (DFA 'missing', no phrase in flight, fires < max, room left),
a fixed logit write moves the stop set's mass onto the phrase's first token (greedy: max → the same token as the swap;
sampling: T·logsumexp → fires with the stop probability), and the emitted token drives the same register updates. Validation:
- **Greedy u0 S2F, compiled (`FE_MISS=w`) vs the swap hook**: token-identical on 600/600 dev and 300/300 dev2 items,
  24 / 12 fires each; acc 82.00 / 82.00 (j23322 vs j23276).
- Sampled (T=1 seed 7): 81.00 vs 81.00. 590/600 identical, 598/600 same correctness. The logit write changes the RNG
  path only where the trigger fires.
- With this, S2F has no post-sampling step left.
- **Result: test + GSM-Hard once with the compiled trigger** (miss3 j23342, T=1 seed 7; paired vs the swap-hook run):

  | config | test | Δ vs hook | GSM-Hard (missing-only triggers) | Δ vs hook |
  |---|---|---|---|---|
  | u0 S2F, compiled | **72.63** [70.13, 75.06] | +0.38 [−0.99, 1.82] | **56.03** [53.37, 58.68] | +0.83 [−0.68, 2.43] |
  | B u300 S2F, compiled | **74.22** [71.80, 76.42] | +0.38 [−1.06, 1.82] | **57.47** [54.81, 60.12] | +0.08 [−0.99, 1.06] |
  | B vs u0, both compiled | +1.59 [−0.61, 3.79] | | +1.44 [−0.68, 3.49] | |

  The compliant all-weights S2F matches the old hook numbers within noise. [12] item 1 is done.

## 8. [14]5 Implant ablation at TEST (GSM8K test, T=1 seed 7, once; eval/lists/abl_test.list; paired Δ vs the full config [CI])
Each row turns off one weight-space part at inference on the same RL checkpoint, with no retraining.
- nocalc: calculator implant off. nochain: chain2 implant off. notools: both off, host + stack mechanics only.
- nosan: compiled sanity filter off. nofgt: forget mask off (later tries see earlier ones). vote0: plain majority instead of the compiled voter.
- nocomp: compiled triggers off (B only).

| part off | A1 u300 N=8 (79.76) | C u300 N=4 (74.60) | B u300 S2F (73.84) |
|---|---|---|---|
| calculator | 79.00, −0.76 [−2.27, 0.76] | 74.75, +0.15 [−1.59, 1.90] | 73.84, 0.00 [−1.67, 1.67] |
| chain2 | 78.62, −1.14 [−2.35, 0.15] | 74.07, −0.53 [−2.12, 1.06] | 73.24, −0.61 [−2.05, 0.83] |
| both tools | **77.26, −2.50 [−4.25, −0.76]** | 75.36, +0.76 [−1.06, 2.73] | 73.31, −0.53 [−2.35, 1.21] |
| sanity filter | 79.83, +0.08 (1 item changes) | 74.60, 0.00 (never changes an answer) | – |
| compiled triggers | – | – | 73.54, −0.30 [−1.90, 1.21] |
| forget mask | **71.57, −8.19 [−10.16, −6.29]** | **65.43, −9.17 [−11.37, −6.90]** | 72.48, −1.36 [−2.96, 0.15] |
| compiled voter → plain majority | 78.39, −1.36 [−2.65, −0.15] | 73.62, −0.99 [−2.12, 0.15] | – |
| *SC×4, same host (programmatic)* | *75.44, −3.79 (vs N=4)* | *71.19, −3.41 [−5.61, −1.21]* | *77.10, +3.26 [1.51, 5.01] (4× tokens)* |

Reading:
- On GSM8K test, after RL, the calculator and chain2 implants add little: −0.8 to −2.5 together on A1, and nothing
  significant on C or B. GSM8K numbers are small, and the RL'd host computes most of them itself.
- The test-time gain comes from the RL'd serial stack mechanics:
  - forget mask: −8 to −9 when off;
  - compiled voter: −1 to −1.4.
- The tools' value is on GSM-Hard (abl_hard.list, j23341, paired vs the full config):

  | part off | A1 u300 N=8 hard (63.15) | B u300 hard (57.39) |
  |---|---|---|
  | calculator | **47.99, −15.16 [−17.51, −12.74]** | |
  | chain2 | 62.55, −0.61 [−2.05, 0.91] | |
  | both tools | **47.76, −15.39 [−17.74, −12.96]** | **40.18, −17.21 [−19.64, −14.71]** |

  On GSM-Hard's large numbers the RL'd host still depends on the in-weight calculator (−15 to −17). chain2 adds nothing
  once the calculator is on.
- A1 and C one-call serial beat 4 separate calls of the same host (programmatic SC). For B, single-attempt S2F,
  SC×4 beats one call, at 4× the tokens.

## 9. [16]/[17] WPR: weight-space position reset (the implant lane's implant, implants/wpr/)
**What it is.** Each serial attempt's query and new key are rotated by R(−off) with a fixed fp32 rotary table, so
attempt k sees the question at the attempt-0 distance. The whole mechanism lives in weights: fixed tables, a register
offset, and the existing rope kernel. It is applied in comp_serial.SerialW._step_triton. My copies are byte-identical
to implants/wpr/comp_serial.wpr_fixed.py, and WPR unset gives a bitwise-unchanged path (the implant lane: 600/600).

**Rung 1 on TEST / GSM-Hard** (u0 host a2_u300_s2, N=8, T=1 seed 7, once; eval/lists/wpr_rung1.list, j23378):

| config | test [CI] | Δ [CI] | GSM-Hard [CI] | Δ [CI] | tok/q test |
|---|---|---|---|---|---|
| u0 N=8, no reset | 71.87 [69.37, 74.15] | – | 57.54 [54.89, 60.12] | – | 1093.8 |
| **u0 N=8 + WPR (rung 1, all weights)** | **80.67** [78.47, 82.71] | **+8.79 [6.75, 10.84]** | **62.17** [59.51, 64.75] | +4.62 [2.65, 6.60] | 1276.9 |
| u0 N=8 + programmatic reset (NON-COMPLIANT ref.) | 80.97 [78.77, 83.09] | +9.10 [7.13, 11.07] | 62.09 [59.44, 64.67] | +4.55 [2.65, 6.52] | 1273.5 |
| WPR − programmatic reset | | −0.30 [−1.67, 1.14] | | +0.08 [−1.52, 1.59] | |
| A1 u300 N=8 (RL, no reset) | 79.76 [77.56, 81.88] | +7.88 [5.91, 9.86] | 63.15 [60.50, 65.81] | +5.61 [3.56, 7.66] | 1222.7 |
| A1 u300 N=8 + WPR at inference only | 80.36 [78.17, 82.49] | +0.61 [−0.99, 2.12] vs A1 | 63.61 [61.03, 66.19] | +0.45 [−1.21, 2.05] vs A1 | 1267.9 |

Per-attempt accuracy on test:
- u0 without a reset: 73.8 at attempt 1, falling to the mid-60s by attempt 8.
- u0 with WPR: flat at 71–73.
- A1 + WPR: flat at 74–76.

Reading:
1. The compliant weight-space reset reproduces the programmatic reset at test (−0.30 n.s.) and on hard (+0.08).
2. Untrained, it gives the same test gain as 300 RL updates of A1 (80.67 vs 79.76). Most of what A1's RL bought on
   test is the fix for the position drift of later attempts.
3. On top of A1, WPR adds +0.6 / +0.45 (n.s.). RL and WPR fix overlapping failures: A1 had already learned to keep later
   attempts usable. The blog must say this plainly.
4. On GSM-Hard, A1 RL is +1.0 above WPR alone (63.15 vs 62.17, unpaired).

**Rung 2 on TEST / GSM-Hard** ([17](2); u0 host, T=1 seed 7, once; eval/lists/wpr_rung2.list, j23419).
Code: eval/trees/r2, i.e. the implant lane's implants/rungs/rung2 (comp_serial + aw_adapt + fastroll) and fe_hook with `AWA.setup`.
Env: `WPR=1 SW_NS=16 SW_MAXN=7936 AW_ADAPT=A=0.7/16/4:S=4/0.5:B=3:C=400`, B_max 128, SAN_SMAX 8192.
- CPU fuzz (test_aw.py) passes in the job.
- Rung 1 rerun on the same stk_r2 code with an empty AW_ADAPT (r1f, j23420) is **bitwise identical** to r1_u0_N8 on both
  sets (1319/1319 rows). So stk_r2 is a drop-in, and the handoff's fastroll eos fix never fires in these runs.

| config | test [CI] | Δ vs rung 1 | Δ vs A1 u300 N=8 | tok/q, tries/q | GSM-Hard [CI] | Δ vs rung 1 | Δ vs A1 u300 N=8 | tok/q, tries/q |
|---|---|---|---|---|---|---|---|---|
| rung 1 (WPR, N=8) | 80.67 | – | +0.91 | 1277, 8 | 62.17 | – | −0.98 | 1628, 8 |
| **rung 2** (WPR + A/S/B/C) | **81.58** [79.38, 83.70] | +0.91 [−0.45, 2.35] | +1.82 [0.15, 3.56] | 1482, 7.87 | **62.62** [59.97, 65.13] | +0.45 [−1.14, 2.05] | −0.53 [−2.35, 1.29] | 2365, 10.89 |

Other paired comparisons for rung 2:
- vs the u0 N=8 stack without a reset: test +9.70 [7.66, 11.68], Hard +5.08 [3.11, 7.13].
- vs SC8 (8 independent B-stack samples, plain majority): test +1.90 [0.08, 3.64], Hard +2.58 [0.76, 4.40].

How many tries rung 2 uses:
- test: 640/1319 items stop at 4 (stop rule S), 363 at 8, 280 run to the 16 cap;
- Hard: 227 at 4, 486 at 8, 567 at 16.

On Hard the RM=1 protocol (repair only missing answers) is kept, so B (repairing non-integer answers) is inert.

Reading:
- On test, rung 2 matches the implant lane's dev gain over rung 1 (+0.8 on dev, +1.0 on dev2): +0.9, n.s. at n=1319, at +16% tokens.
  It is the best untrained all-weights stack, and +1.8 above A1's 300 RL updates (significant).
- On Hard, the adaptive tries spend +45% tokens for +0.45 (n.s.). The vote rarely gets a confident share on Hard
  items, so 43% of them run to 16 tries.

Figures: figures/wpr_test.png and figures/wpr_hard.png. The rung is also in figures/test_bars and figures/hard_bars.

**Rung 3 on TEST / GSM-Hard** (ladder rung 3; u0 host, T=1 seed 7, once; eval/lists/wpr_rung3.list, j23472, 11.5 min).
Code: eval/trees/r3 = stk_r2 plus the implant lane's implants/rungs/rung3 (comp_serial, aw_vf, numreg, fastroll, vf_w.pt).
fe_hook also logs `pick0` (the base voter's slot) next to `win` (the VF pick).
Env: rung 2 plus `SW_NS=24 AW_ADAPT=A=0.7/24/4:S=4/0.5:B=3:C=400 AW_VF=1`. The CPU fuzz (test_aw) passes in the job.
Dev (the implant lane, 4 seeds): 90.29 / 87.33, sign p=0.021 vs rung 2.

| config | test [CI] | Δ vs rung 2 | Δ vs rung 1 | tok/q, tries/q | GSM-Hard [CI] | Δ vs rung 2 | Δ vs rung 1 | tok/q, tries/q |
|---|---|---|---|---|---|---|---|---|
| rung 2 | 81.58 | – | +0.91 | 1482, 7.87 | 62.62 | – | +0.45 | 2365, 10.89 |
| **rung 3** (+ Nmax 24 + VF) | **82.71** [80.59, 84.76] | +1.14 [−0.15, 2.50] | +2.05 [0.61, 3.56] | 1774, 9.37 | 60.42 [57.77, 63.08] | **−2.20 [−3.94, −0.53]** | −1.74 [−3.56, 0.08] | 3014, 13.79 |

Other paired comparisons for rung 3:
- vs the u0 N=8 stack without a reset: test +10.84 [8.79, 13.04], Hard +2.88 [0.91, 4.85].
- vs A1 u300 N=8: test +2.96 [1.21, 4.70], Hard −2.73 [−4.70, −0.61].
- vs SC8 u0: test +3.03 [1.36, 4.70], Hard +0.38 [−1.59, 2.35].

VF vs the base voter on the same rollouts: score the `pick0` slot's register answer against the VF `win` slot.
- test: +1.21 [0.15, 2.27], 32 fixes / 16 breaks; the picks differ on 88 items.
- Hard: −3.03 [−4.78, −1.36], 45 fixes / 85 breaks; the picks differ on 324 items.
- Caveat: on 37 Hard rows the 16-symbol register decode disagrees with the text score. On test it agrees on 1319/1319.

Decomposition run: rung 2 + Nmax 24 without VF (r2n24; eval/lists/wpr_r2n24.list, j23492). On test it gives the same rollouts as rung 3 (same tok/q).

| set | r2n24 | Nmax 24 effect (r2n24 − rung 2) | VF effect (rung 3 − r2n24) |
|---|---|---|---|
| test | 81.50 [79.23, 83.55] | −0.08 [−1.21, 1.06] | **+1.21 [0.23, 2.27]** |
| Hard | **63.53** [60.88, 66.11] | +0.91 [−0.38, 2.20] | **−3.11 [−4.78, −1.36]** |

- So all of rung 3's test gain and all of its Hard loss come from the voter. Nmax 24 is neutral on test and n.s. +0.9 on Hard.
- r2n24's 63.53 is the highest compliant Hard number from the untrained host, but it is n.s. above rung 2.

**RL + rung-2 stack** (A1W u300 hosts, i.e. WPR inside RL, seeds 0–2, with rung-2 env; eval/lists/wpr_r2rl.list, j23499, once).
The A/S/B/C thresholds are the ones tuned on u0 dev and are not retuned for the RL host.

| set | per seed | pooled (sd) | vs A1W N=8 (same hosts) | vs rung 2 u0 | vs A1 N=8 (3 seeds) | tries/q |
|---|---|---|---|---|---|---|
| test | 79.00 / 80.52 / 80.36 | 79.96 (0.84) | +0.83 [0.13, 1.57] | **−1.62 [−3.06, −0.18]** | −0.10 [−1.34, 1.09] | 7.1 (≈800 stop at 4) |
| Hard | 63.68 / 64.22 / 64.67 | **64.19** (0.49) | **+1.44 [0.71, 2.20]** | +1.57 [−0.08, 3.16] | +0.88 [−0.45, 2.25] | 9.6 (≈445 at 16) |

- On Hard, this is the best compliant number: 64.19 over 3 seeds. Adaptive tries add +1.4 on top of RL+WPR (significant).
- On test, the untrained rung-2 stack still beats RL + rung 2 by 1.6. RL makes the votes agree earlier (about 800/1319 stop at 4, vs 640 for u0), so the stack spends fewer tries, but accuracy does not rise.
- The same rung-2 stack on the RL-from-base CW hosts (§10; j23509, 3 seeds): test 80.26 (sd 1.76), Hard 58.76 (sd 2.26).
  - vs CW N=4: test +3.74 [2.73, 4.75], Hard +3.69 [2.58, 4.88]. This mixes more tries (CW trained at N=4) with the adaptive rules.
  - vs rung 2 u0: test −1.31 [−2.73, 0.10], Hard −3.87 [−5.53, −2.27].
  - vs A1W + rung 2: test +0.30, Hard −5.43 [−6.97, −3.89].
  - On Hard the CW hosts rarely reach a confident vote: 41–88 items stop at 4, and 600+ run to 16.
- Blog reading: on GSM8K test, an untrained weight-space stack (rung 2/3) ≥ any RL arm. On GSM-Hard, RL with WPR plus adaptive tries is best, by a small margin (+0.7–1.6 over the untrained rungs).

How many tries rung 3 uses:
- test: 648 stop at 4, 358 at 8, 233 at the 24 cap;
- Hard: 235 at 4, 496 at 8, 451 at 24.

Reading:
- On test, rung 3 is the best compliant number: 82.71, +2.05 over rung 1 (significant) and +1.14 over rung 2 (n.s.).
  The VF pick contributes about +1.2 of that. The dev gain does carry over to the GSM8K test distribution.
- On GSM-Hard, rung 3 is worse than rung 2 (−2.2, significant), and the loss comes from the voter.
  - VF was trained on dev rollouts (GSM8K-train items). Its answer-group features (number novelty, question-copy) do not transfer to Hard's large numbers.
  - The extra 8 tries (Nmax 24) cost +27% tokens on Hard and do not offset this.
- For the blog: report rung 3 for test and rung 2 (or r2n24) for Hard, and say plainly that the learned voter is distribution-specific.

**RL with WPR** ([16]/[17](1)): arms A1W_s{0,1,2} (A1 config, RL host) and CW_s{0,1,2} (C config, base host), WPR=1.
With WPR=1, rl_hook.build_docs trains on the reset positions, so the trainer forward matches the rollout.
Smokes (2 updates, full dev at u0):

| smoke | FIRST_UPDATE_RATIO | u0 dev | u0 dev without WPR |
|---|---|---|---|
| A1W_s0 (j23376) | 0.999947 | vote8 88.67 | 81.33 (A1 u0) |
| CW_s0 (j23377) | 0.999886 | vote4 51.67, attempt acc 41.0 / 40.0 / 39.5 / 40.7 | 36.8, attempts 36 / 19 / 13 / 9 (C u0) |

The A1W u0 dev of 88.67 equals the implant lane's rung-1f seed-7 dev. The 6 arms are at the top of QUEUE.md and run as GPUs free.

**CW (C config + WPR, base host, N=4) — 3 seeds done (07:23Z / 07:26Z / 07:58Z).**
Dev (vote4, 600 dev items) by update:

| u | 0 | 50 | 100 | 150 | 200 | 250 | 300 |
|---|---|---|---|---|---|---|---|
| CW_s0 | 51.67 | 73.00 | 79.50 | 84.83 | 84.83 | 86.33 | 86.50 |
| CW_s1 | 51.67 | 73.83 | 79.50 | 83.83 | 85.33 | 86.17 | 86.67 |
| CW_s2 | 51.67 | 74.50 | 82.33 | 84.67 | 86.50 | 86.17 | 84.17 |
| C (no WPR) | 36.8 | | | | | | 84.83 |

The C value at u300 is C600's u0 dev, i.e. the C checkpoint at u300. Full table: `python3 eval/results_tools/devcmp.py dev_vote4 CW_s0 CW_s1 C600`.

TEST and GSM-Hard (u300, N=4, T=1, seed 7, once each; eval/seed_CWs{0,1,2}.list, j23417/j23418/j23428), paired Δ [95% CI]:

| config | test | Δ vs C | Δ vs Bb_s5 | Δ vs Pb_s1 | tok/q | GSM-Hard | Δ vs C | Δ vs Bb_s5 |
|---|---|---|---|---|---|---|---|---|
| C (no WPR, u300) | 74.60 | – | +2.50 [0.23, 4.62] | +4.24 | 545 | 54.66 | – | −1.74 |
| **CW_s0** | **77.18** [74.91, 79.45] | +2.58 [0.38, 4.70] | +5.08 [2.88, 7.35] | +6.82 [4.55, 9.10] | 595 | 53.68 | −0.99 [−3.18, 1.36] | −2.73 [−5.00, −0.53] |
| **CW_s1** | **77.33** [74.98, 79.45] | +2.73 [0.68, 4.78] | +5.23 [3.03, 7.43] | +6.97 [4.70, 9.33] | 647 | 55.12 | +0.45 [−1.74, 2.65] | −1.29 [−3.71, 0.99] |
| **CW_s2** | **75.06** [72.63, 77.33] | +0.45 [−1.67, 2.58] | +2.96 [0.61, 5.23] | +4.70 [2.35, 6.97] | 588 | 56.41 | +1.74 [−0.38, 3.87] | 0.00 [−2.27, 2.27] |
| CW mean of 3 seeds | **76.52** (seed sd 1.3) | +1.92 | +4.42 | +6.16 | 610 | 55.07 | +0.41 | −1.34 |

CW_s1 − CW_s0 = +0.15 [−1.90, 2.20] on test.

Reading (3 seeds; C has one seed here, C_s3 runs and C_s2 is queued, so the C-vs-CW gap still carries C's seed noise):
1. With the position reset in the weights during RL, the serial stack from base reaches 75.1–77.3 on test (mean 76.5).
   That is +1.9 over C without WPR (one C seed), and the serial step over single-attempt tools (Bb_s5) grows from
   +2.5 to +4.4 on average (+3.0 to +5.2 per seed), at about 4× the tokens. Dev at u300 ranks the seeds like test (86.5 / 86.7 / 84.2).
2. WPR fixes the start: u0 dev 51.7 vs 36.8. The later attempts no longer collapse (attempt acc 41/40/40/41 vs 36/19/13/9).
   RL then ends about 1.7 dev points higher.
3. GSM-Hard is unchanged (−1.0 / +0.5 vs C, n.s.). The serial stack still does not beat single-attempt Bb_s5 on Hard.
   That matches rung 1, where WPR gave +4.6 on Hard from the u0 host but nearly nothing on top of A1 RL.

**A1W (A1 config + WPR, RL host a2s2, N=8) — 3 seeds done (07:58Z / 08:07Z / 08:08Z).**
TEST/Hard (u300, T=1 seed 7, once; eval/seed_A1Ws{0,1,2}.list, j23431/j23437/j23438), paired Δ [95% CI]:

| config | test N=8 | Δ vs A1 u300 N=8 | Δ vs rung 1 (u0 + WPR, no RL) | test N=4 | Δ vs A1 N=4 | GSM-Hard N=8 | Δ vs A1 | Δ vs rung 1 |
|---|---|---|---|---|---|---|---|---|
| A1W_s0 | 78.70 [76.50, 80.89], 1158 tok/q | −1.06 [−2.88, 0.68] | **−1.97 [−3.79, −0.15]** | 78.01 | −1.21 [−3.11, 0.68] | 62.70 | −0.45 n.s. | +0.53 n.s. |
| A1W_s1 | 79.91 [77.56, 82.03], 1208 tok/q | +0.15 [−1.52, 1.82] | −0.76 [−2.50, 0.99] | 78.24 | −0.99 [−2.73, 0.76] | 62.62 | −0.53 n.s. | +0.45 n.s. |
| A1W_s2 | 78.77 [76.42, 80.89], 1187 tok/q | −0.99 [−2.65, 0.76] | −1.90 [−3.71, 0.00] | 79.00 | −0.23 [−2.05, 1.59] | 62.93 | −0.23 n.s. | +0.76 n.s. |
| A1W mean of 3 | **79.13** (seed sd 0.68) | −0.63 | **−1.54** | 78.42 | −0.81 | 62.75 | −0.40 | +0.58 |

Dev for A1W_s0 vs A1 seed 2 (no WPR):

| u | A1W_s0 vote8 | vote1 | oracle8 | allsame8 | A1_s2 vote8 | vote1 | allsame8 |
|---|---|---|---|---|---|---|---|
| 0 | 88.67 | 80.00 | 92.67 | 60.17 | 81.33 | 79.00 | 9.33 |
| 100 | 88.17 | 82.83 | 92.83 | 64.83 | 86.00 | 81.50 | 48.00 |
| 200 | 88.00 | 82.83 | 91.33 | 69.50 | 85.67 | 81.33 | 52.50 |
| 300 | 87.17 | 82.83 | 92.67 | 71.83 | 86.00 | 82.00 | 51.17 |

A1W_s1/s2 dev vote8 at u0 → u300: 88.67 → 88.00 / 88.00; allsame8 60.2 → 68.5 / 70.7; vote1 80.0 → 81.7 / 83.0.
Vs rung 2 (81.58), A1W N=8 is −2.88 / −1.67 / −2.81 on test.

Reading (3 seeds):
1. With WPR on, RL does not improve the N=8 stack. Single attempts get better (vote1 +2.8 on dev, every attempt slot
   80 → 83), but the 8 attempts agree more (dev allsame8 60 → 72; on test, all 8 valid and identical: 49.8% for rung 1
   vs 58.9% for A1W_s0). So vote8 falls on dev (88.67 → 87.2–88.0) and on test the 3 seeds are −0.8 to −2.0 vs the untrained rung 1
   (mean −1.54; two of three are at or past the CI edge).
2. Without WPR, RL had something to fix: later attempts collapsed (attempt 8 at 29% at u0) and RL repaired them. WPR
   removes that fix, and what is left for RL is to sharpen the policy, which costs vote diversity.
3. Dev is flat: vote8 stays at 86.2–89.5 for all seeds and never clearly beats u0 (88.67). The single higher points
   (s1 88.83, s2 89.50 at u125) are within dev noise (±1.3 at n=600). The mean over u25–u300 is below u0 for every seed. For the blog, the strongest compliant serial stack on the
   RL host is rung 2 with no RL (81.58), not RL + WPR.

## 10. [13a]/[14]2 Splitting "base → C": plain RL alone (P) vs RL + tools (B_base) vs RL + serial stack (C)
Test and GSM-Hard, T=1 seed 7, run once each. The Pb_s1 rows come from eval/lists/seed_Pbs1.list (j23392); the greedy rows from greedy_H.list (j23393).
CIs are 4000-resample paired bootstraps (eval/results_tools/pboot.py, numpy-free, same protocol as ana_rltest.py).

| config (host) | test [CI] | Δ vs base H [CI] | Δ vs P [CI] | GSM-Hard [CI] | Δ vs P [CI] | tok/q test |
|---|---|---|---|---|---|---|
| base host, no implants (H) | 28.05 | – | | 10.46 | | |
| **P = plain RL, no implants, base host, u300 (Pb_s1)** | **70.36** [67.93, 72.71] | **+42.30** [39.35, 45.34] | – | **44.66** [41.93, 47.31] | – | 120.4 |
| P seed 2 (Pb_s2, u300; seed_Pbs2.list, j23448) | 71.19 [68.69, 73.46] | +43.14 [40.11, 46.10] | +0.83 [−1.14, 2.81] vs Pb_s1 | 45.34 [42.61, 48.07] | +0.68 [−1.52, 2.81] vs Pb_s1 | 128.6 |
| B u300 (RL host + S2F stack + RL; compiled trigger) | 74.22 [71.80, 76.50] | | +3.87 [1.67, 6.14] | 57.47 [54.74, 60.20] | +12.81 [10.31, 15.39] | 142.5 |
| B_s2 u300 | 74.00 [71.57, 76.35] | | +3.64 [1.29, 5.84] | 58.38 [55.72, 61.03] | +13.72 [11.14, 16.38] | 147.1 |
| B_s3 u300 (seed_Bs3.list, j23538) | 73.62 [71.11, 75.89] | | +3.26 [0.99, 5.46] | 57.62 [54.97, 60.20] | +12.96 [10.46, 15.62] | 143.5 |
| C u300 N=4 (base host + full stack + RL) | 74.60 [72.18, 76.95] | | +4.25 [1.97, 6.52] | 54.66 [51.93, 57.32] | +10.01 [7.43, 12.51] | 545.3 |
| C_s2 u300 N=4 (seed 7; seed_Cs2.list, j23465) | 74.68 [72.33, 76.95] | | +4.32 [2.05, 6.60] | 51.86 [49.13, 54.66] | +7.20 [4.55, 9.93] | 515.9 |
| C_s3 u300 N=4 (seed 6; seed_Cs3.list, j23462) | 75.13 [72.71, 77.41] | | +4.78 [2.43, 7.05] | 52.39 [49.66, 55.12] | +7.73 [5.16, 10.31] | 583.9 |
| C600 N=4 | 75.06 [72.63, 77.33] | | +4.70 [2.35, 6.97] | 55.12 [52.39, 57.85] | +10.46 [7.88, 13.04] | 561.1 |
| CW_s0 u300 N=4 (C + WPR in the weights during RL, §9) | 77.18 [74.91, 79.45] | | +6.82 [4.55, 9.10] | 53.68 [51.02, 56.41] | +9.02 [6.44, 11.60] | 594.5 |
| CW_s1 u300 N=4 | 77.33 [74.98, 79.45] | | +6.97 [4.70, 9.33] | 55.12 [52.39, 57.77] | +10.46 [7.81, 13.04] | 647.4 |
| CW_s2 u300 N=4 | 75.06 [72.63, 77.33] | | +4.70 [2.35, 6.97] | 56.41 [53.68, 59.06] | +11.75 [9.10, 14.33] | 587.7 |
| CW600 N=4 (CW_s0 u300 → u600; seed_CW600.list, j23508) | 77.79 [75.44, 79.98] | | +7.43 [5.08, 9.70] | 55.65 [52.99, 58.45] | +10.99 [8.34, 13.65] | 744.5 |
| CW + rung-2 stack (A/S/B/C, up to 16 tries; wpr_r2cw.list, j23509), seeds 0/1/2 | 81.12 / 81.43 / 78.24, **pooled 80.26** (sd 1.76) | | | 56.71 / 61.18 / 58.38, **pooled 58.76** (sd 2.26) | | 1612 |
| **Bb_s5 u300 (B_base: base host + S2F stack + RL, 1 attempt)** | **72.10** [69.67, 74.45] | +44.05 (vs H) | +1.74 [−0.38, 3.87] | **56.41** [53.83, 59.06] | +11.75 [9.33, 14.33] | 144.1 |
| Bb_s6 u300 (seed_Bbs6.list, j23456) | 73.39 [70.96, 75.74] | | +3.03 [0.99, 5.16] | 55.80 [53.07, 58.45] | +11.14 [8.57, 13.80] | 142.1 |
| Bb_s7 u300 (seed_Bbs7.list, j23457) | 73.39 [70.96, 75.66] | | +3.03 [0.83, 5.31] | 54.97 [52.24, 57.70] | +10.31 [7.73, 12.89] | 141.1 |

**Seed-pooled (eval/results_tools/pbootm.py).** Per-item mean over seeds, paired item bootstrap; the CI does not include seed variance, so the seed sd is listed separately:

| comparison | test A → B (seed sd) | Δ test [CI] | Hard A → B (seed sd) | Δ Hard [CI] |
|---|---|---|---|---|
| P (2 seeds) → Bb (3 seeds) | 70.77 (0.59) → 72.96 (0.74) | **+2.19 [0.73, 3.70]** | 45.00 (0.48) → 55.72 (0.72) | **+10.73 [8.73, 12.79]** |
| P (2) → C (3) | 70.77 (0.59) → 74.80 (0.29) | **+4.03 [2.38, 5.62]** | 45.00 (0.48) → 52.97 (1.49) | **+7.97 [5.94, 10.10]** |
| Bb (3) → C (3) | 72.96 (0.74) → 74.80 (0.29) | **+1.84 [0.48, 3.16]** | 55.72 (0.72) → 52.97 (1.49) | **−2.75 [−4.45, −1.19]** |
| Bb (3) → CW (3) | 72.96 → 76.52 (1.27) | **+3.56 [2.10, 4.98]** | 55.72 → 55.07 (1.37) | −0.66 [−2.22, 0.83] |
| C (3) → CW (3) | 74.80 → 76.52 | **+1.72 [0.56, 2.93]** | 52.97 → 55.07 | **+2.10 [0.76, 3.41]** |
| Bb (3) → B (3) (same S2F stack + RL; base host → RL host) | 72.96 → 73.94 (0.31) | +0.99 [−0.38, 2.27] | 55.72 → 57.82 (0.49) | **+2.10 [0.76, 3.39]** |

Greedy, single attempt, no implants: base 52.54, P u300 70.74 (+18.20 [15.54, 20.77]), u0 RL host a2s2 70.51.
GSM-Hard greedy: base 26.54, P 46.70 (+20.17 [17.66, 22.67]).

Dev curves (greedy, single attempt, dev / dev2, from the arms' steps.jsonl):

| u | 0 | 50 | 100 | 150 | 200 | 250 | 300 |
|---|---|---|---|---|---|---|---|
| P (Pb_s1) | 62.2 / 58.3 | 74.2 / 70.7 | 78.0 / 79.7 | 79.7 / 79.0 | 80.5 / 80.3 | 81.5 / 81.3 | 79.5 / 81.7 |
| P (Pb_s2) | 62.2 / 58.3 | 76.2 / 72.3 | 78.7 / 79.7 | 81.3 / 84.0 | 79.0 / 81.7 | 81.2 / 83.7 | 81.8 / 82.0 |
| B_base (Bb_s5) | 63.0 / 60.3 | 74.2 / 74.3 | 77.8 / 78.3 | 82.0 / 81.0 | 81.2 / 81.3 | 82.5 / 78.7 | 82.0 / 78.0 |
| B_base (Bb_s6) | 63.0 / 60.3 | 77.0 / 77.7 | 79.7 / 79.0 | 82.7 / 80.0 | 81.7 / 82.7 | 81.2 / 81.3 | 82.5 / 82.3 |
| B_base (Bb_s7) | 63.0 / 60.3 | 74.8 / 75.3 | 80.8 / 80.7 | 82.2 / 81.3 | 83.2 / 84.3 | 82.8 / 84.3 | 82.2 / 83.0 |
| C (vote4; seed 5) | 36.8 / 35.3 | 52.7 / 48.3 | 72.5 / 68.3 | 80.0 / 76.3 | 82.2 / 82.3 | 82.5 / 80.3 | 84.3 / 83.0 |
| C_s2 (vote4; seed 7) | 36.2 / 35.7 | 58.8 / 57.3 | 73.5 / 72.0 | 80.0 / 78.3 | 81.2 / 79.3 | 80.7 / 81.0 | 81.2 / 82.0 |
| C_s3 (vote4; seed 6) | 36.2 / 35.7 | 58.0 / 60.0 | 73.5 / 71.3 | 79.0 / 79.7 | 82.7 / 83.3 | 82.7 / 80.3 | 83.8 / 82.0 |

Both reach a dev plateau of ~80–82 by u150. The stack (Bb) is at most +1–2 over plain RL (P) on greedy dev.

**Split of base → C (one seed each so far; paired Δ, T=1):**

| step | test | GSM-Hard | tok/q test |
|---|---|---|---|
| base (H) | 28.05 | 10.46 | ~113 |
| + RL alone (P) | 70.36 (+42.30 [39.35, 45.34]) | 44.66 (+34.19) | 120 |
| + tools in one attempt (Bb vs P) | 72.10 (+1.74 [−0.38, 3.87], n.s.) | 56.41 (+11.75 [9.33, 14.33]) | 144 |
| + serial stack N=4 (C vs Bb) | 74.60 (+2.50 [0.23, 4.62]) | 54.66 (−1.74 [−4.25, 0.53], n.s.) | 545 |
| + serial stack N=4 with WPR (CW_s0 / s1 / s2 vs Bb) | 77.18 / 77.33 / 75.06 (+5.08 [2.88, 7.35] / +5.23 [3.03, 7.43] / +2.96 [0.61, 5.23]) | 53.68 / 55.12 / 56.41 (−2.73 [−5.00, −0.53] / −1.29 n.s. / 0.00 n.s.) | 595 / 647 / 588 |

Figures: figures/base_split.png (test) and figures/base_split_hard.png. New seeds are picked up automatically (make_figs.py; seed dots).

Reading (P has 2 seeds; Bb, C and CW have 3; points 1 and 5 were written at one seed each):
0. **With seeds: tools in one attempt (Bb) beat plain RL (P) by +2.2 [0.7, 3.7] on test and +10.7 on Hard.** The one-seed
   comparison (+1.7, n.s.) was the low end. The serial-stack-with-WPR (CW) adds +3.6 [2.1, 5.0] on test over Bb and nothing on Hard.
   **C replicates tightly on test** (74.60 / 74.68 / 75.13, sd 0.3): serial N=4 without WPR adds +1.8 [0.5, 3.2] over Bb
   on test at ~3.8× the tokens, and **loses −2.8 [−4.5, −1.2] on GSM-Hard** (C_s2/C_s3 Hard 51.9 / 52.4 vs 54.7 for seed 5).
   WPR in the weights during RL fixes that: CW vs C is +1.7 [0.6, 2.9] test and +2.1 [0.8, 3.4] Hard. So the
   per-step split at 3 seeds is: base 28 → P 70.8 (RL alone) → Bb 73.0 (+tools, +2.2) → C 74.8 (+serial, +1.8) → CW 76.5
   (+WPR, +1.7); GSM-Hard 10.5 → 45.0 → 55.7 → 53.0 → 55.1.
1. **Most of "base → C" on GSM8K test is RL alone.** Plain RL from the base, with no implants and 1 attempt at ~120
   tok/q, reaches 70.4 test. That is +42 over the base at T=1, and +18 over the base greedy (52.5). C adds +4.3 on top,
   at 4.5× the tokens.
2. The base's T=1 number (28) is far below its greedy number (52.5). Much of the zero-shot T=1 "37 → 84" dev gain is
   RL sharpening the sampling distribution, which greedy decoding already partly gets. Report base greedy and 4-shot
   (56.5, §1) next to it.
3. **The implants pay on GSM-Hard.** The stacks are +10 to +14 over P there (calculator: large-number arithmetic), and
   +3.6 to +4.7 on test.
4. P is single-attempt at ~120 tok/q, so the fair matched-token comparison is B (single attempt, ~142 tok/q): +3.9 test, +12.8 hard.
5. From the base host, the serial stack (C) adds +2.5 test over the single-attempt stack (Bb) at 3.8× the tokens.
   It adds nothing on Hard. Same-token SC (§8) is the matched-token check. WPR arms (CW) are the base-host serial
   stack with the position fix; they come next.

## 7. Concerns for the write-up ([6](d))
- Ar's dev decline under RL (88.8 → 85.7) and test below its u0: RL with the reset does not help on top of the reset.
- A peaks early on dev (u150 85.0) and returns to 83.0; A1 is flat-to-rising after u50. Single seed each so far: seeds
  A_s2 (u~230), B_s2, A1_s2, A1D (diversity) are running; seed CIs follow ([10](1)).
- All-same rate rises with RL (A1 N=8: 9% → 54% dev, 48% test) — the vote gains come from a better first attempt plus
  less decay, not from diverse attempts; oracle at N=8 is 89.1 vs vote 79.8. A1D (diversity penalty) attacks this.
- The u300 harness-vs-training dev gap of −0.67 (both arms) — see section 2.
