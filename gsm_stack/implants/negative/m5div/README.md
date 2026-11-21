# lane m5div: diversity of the first steps (task [6] M5)

## Progress log (newest last)

### 03:10Z: step 1 (CPU), how similar are the first steps, and what is the ceiling?
Data: the rung-1 rows (WPR, all weights, `implants/wpr/fe/out/fe/{set}_wpr_s{7,11,13,17}.jsonl`). That gives 32 attempts per item (4 seeds × 8).
- Parser: `m5lib.py`. Attempts are split at the written retry phrase. The attempt answer comes from the slot register (sym/rok/bad/lg).
- Voter replica `m5lib.vote` matches the stored winner and `ok` on 600/600 and 300/300 rows.
- Script: `python3 ana1.py fe 100` writes `ana1_fe.md` (full tables).

First-step keys:
- **W4 / W8**: the first 4 / 8 words of the first line.
- **L1**: the whole first line.
- **L1N**: the numbers in the first line.
- **N2**: the first 2 numbers of the attempt.
- **EQ**: the operator/number left side of the first '='.

**Similarity, dev (dev2 is similar):**
- Within an item and seed, 8 attempts have 2.8 distinct W4, 4.1 distinct L1 and 3.0 distinct EQ.
- Pairs of attempts share their first line 37% of the time (31% on mixed items) and their first equation 51% of the time.
- Many first lines restate the question, e.g. "Hayden spends 5 minutes ironing ...".

**Is the first step predictive of the answer? Only weakly.**
- P(same answer | same L1) = 0.84 vs P(same answer | different L1) = 0.70.
- On mixed items: 0.68 vs 0.53.
- For EQ: 0.86 vs 0.64.
- H(answer | L1) / H(answer) = 0.40 (EQ 0.54, W4 0.57).
- So two attempts with different first lines still reach the same answer 70% of the time. The fork usually comes later (atlas: the divergent step D is the first line in only 28% of cases; median branch position is 0.10 of the attempt).

**Gap cases** (seed vote wrong but some attempt right; dev 116, dev2 75):
- In 65% / 76% of them, a right attempt has a first line used by no winner-answer attempt of that seed. For EQ the figures are 59% / 65%; for W4, 47% / 49%.
- But only 84/191 (dev) of the right attempts in gap seeds have a first line that is unique within the seed.

**Ceiling: resample 8 of the 32 attempts per item (100 draws/item)**

The sequential constraint is: the next attempt's key must differ from all earlier keys (or from those of the plurality group). This is an ideal implant that keeps the host's own conditional distribution.

| policy | dev vote | dev oracle@8 | dev per-att | dev2 vote | dev2 oracle@8 | dev2 per-att |
|---|---|---|---|---|---|---|
| random 8 of 32 (= base) | 87.96 | 92.77 | 77.83 | 85.70 | 91.89 | 73.44 |
| L1-distinct | 88.12 | 93.15 | 76.28 | 85.58 | 91.35 | 71.59 |
| EQ-distinct | 88.17 | 93.68 | 76.75 | 85.25 | 92.41 | 72.01 |
| W4-distinct | 88.26 | 92.58 | 75.40 | 85.40 | 90.95 | 69.83 |
| L1-distinct only if earlier answers unanimous | 87.98 | 93.11 | 77.51 | 85.62 | 92.02 | 73.18 |
| W4: avoid plurality-group first words if its mean probe logit < 2 | 88.42 | 92.94 | 76.38 | 85.61 | 91.66 | 71.25 |
| answer-distinct (upper bound on diversity) | 84.41 | 94.42 | 66.78 | 80.66 | 93.71 | 60.56 |

=> **The ceiling is about zero:**
- Vote: −0.45 to +0.46 pt.
- Oracle@8: at most +0.9 / +0.5 (EQ-distinct).
- Per-attempt accuracy drops by 1–4 pt, because the rarer first steps are worse.

Even perfect answer diversity raises oracle@8 by only +1.7 / +1.8 and costs 3.5–5 vote points. The vote needs repeated right answers; diversity removes them.

I still build the token-level implant (design a) to measure the real effect: an implant can also push attempts to first lines outside the 32-attempt pool.

### 03:20Z: found the WPR multi-seed leak (told ladder-r2-debug)
`fastroll._f_rowregs` moves every tensor with first dim B_max+1 on compaction. comp_serial's `_wpr_ar = arange(B_max+1)` has that shape. It is listed in "fastroll compaction moves: [..., '_wpr_ar']" in aw/out/r2_dev/log.txt.

As a result:
- After a compaction, `Kc[rar, :, slot] = R(-off) K` writes the rotated K rows into the wrong batch rows.
- This persists into the next generate() call.
- It also touches the tail rows of the first seed after compaction.

In my copy, `stk/fastroll.py` excludes `_wpr_ar` (1 line). My base rows are regenerated with that fix. Runs are still one process per seed.

### 03:25Z: implant (design a + gates c), code
`aw/stk/m5div.py` (pure torch). comp_serial.py has 7 hook lines, all behind `M5DIV` (unset means the module is not imported and `m5_on=False`). Diff: `aw/m5_comp_serial.diff`.

Env: `M5DIV=beta/P/mode/tau`.

**What is weights:**
- **Fixed tables:**
  - `canon[tok]`: the leading-space variant maps to the bare token, used at position 0 only. Attempt 0 starts " Tom", later attempts start "Tom".
  - `NL[tok]`: the token contains '\n'.
  - `WS[tok]`: the token is whitespace only.
- **Registers per row:**
  - `R_fl[slot, P]`: the token ids of each attempt's first line, first P host tokens.
  - `s5_m[slot]`: prefix-match bits; the current first line still equals slot i's first line so far.
  - `s5_pm[slot]`: repel bits, set at the retry trigger.
  - `s5_l1`: in-first-line bit.
  - `s5_seen`: a non-whitespace token was seen.
  - The position counter is the stack's existing `s_c` (host tokens in the segment). The repair counter `s_rep` turns everything off in repair segments.
- **Gate (fixed comparator on the answer register at the retry trigger):**
  - mode 0: all earlier attempts.
  - mode 1: only if the valid earlier answers are unanimous (symbol-equality ReLU, as in the voter).
  - mode 2: the plurality group (count ≥ 2) whose mean probe logit is < tau.
- **Action:** a fixed one-hot output-bias table. `logits[R_fl[i, j]] -= beta` for every repelled slot i still prefix-matched at the current position j. Multiplicity is normalized, so each distinct token gets −beta. It is active only while the host writes the first line of a non-repaired attempt k ≥ 1 and j < P.
- **No host decision:** every op is a tensor op inside the captured decode step.

**Fuzz:** `aw/test_m5.py` drives `step`/`apply` with random serial streams (comp_serial timing, repairs, attempts ending inside the first line, random answer registers, all 3 modes) and checks against a python reference (`m5div.bias_ref` + `ref_pm`).
- Result: **M5 FUZZ PASS**: 3405 biased steps, 0 mismatches.
- Mutation checks: dropping the dup normalization, the canon map, the tau gate or the line re-arm each make it FAIL. Dropping s5_pm passes, because s5_m is initialized from pm, so pm is redundant there.

### 04:15Z: switched to the orchestrator's fixed tree, re-ran the ceiling on the fixed rows
- My copy now carries the orchestrator's 2-line `_wpr_ar` fix (`.view(1,-1)`, `[0,:B]`). `stk/fastroll.py` and `stk/aw_hook.py` are re-copied from `gsm_stack/implants/aw/dev_tree` (fixed tree). My own fastroll exclusion is dropped.
- `comp_serial.base.py` is now the fixed `aw/stk/comp_serial.py`. `m5_comp_serial.diff` contains only the 7 added M5 lines.
- Base is the orchestrator's fixed-tree rung-1f rows, `aw/out/r1f_{dev,dev2}`: dev 88.79 (seeds 7/11/13/17 = 88.67/89.50/88.33/88.67), dev2 85.92. These were run with one process per set and seeds in order. My runs use the same structure, so seed 7 is first in the process and the gate-off RNG streams are identical.
- Ceiling re-run on the fixed rows: `python3 ana1.py 'gsm_stack/implants/aw/out/r1f_{st}' 100`, output in `ana1_r1f.md`. Each distinctness policy is compared against random 8-of-32 for the same set. Bold marks the best policy per column.

| policy | dev vote Δ | dev2 vote Δ |
|---|---|---|
| L1-distinct | −0.21 | +0.27 |
| W8-distinct | −0.12 | **+0.62** |
| EQ-distinct | +0.08 | +0.19 |
| EQ: avoid plurality group if mean lg < 1 | **+0.14** | +0.11 |

  Oracle@8 rises by at most +0.67 (dev, EQ: avoid plurality group if lg < 1) and +0.73 (dev2, W8-distinct). Answer-distinct (the bound) gives vote −2.7/−2.6 and oracle@8 +2.1/+2.6. The conclusion is unchanged: the ceiling is within noise.
- Job **j23368** (`job_m5_a`), submitted 04:43Z: `cd implants/m5div/aw && <queue-submit> $PWD 2400 job_m5_a -- env OFF=1 SEEDS=7,11 bash m5job.sh 3/24/0 30/24/0 3/24/1 3/8/2/2.0`. It runs:
  - the node fuzz;
  - the gate-off identity check, `off_dev2` vs `r1f_dev2` seed 7;
  - 4 configs × {dev, dev2} × seeds 7,11.

### 05:05Z: j23368 results (seeds 7, 11; base = fixed-tree r1f rows, same seeds)
- Node fuzz: M5 FUZZ PASS.
- **Off-gate identity:** the patched stack with M5DIV unset (`off_dev2`, dev2, seed 7) matches `aw/out/r1f_dev2` seed 7 token for token on **300/300 rows** (acc 86.33, tok/q 1267.0).
- Analysis: `python3 ana2.py gsm_stack/implants/aw/dev_tree/out/r1f aw/out/a_m5_3_24_0 aw/out/a_m5_30_24_0 aw/out/a_m5_3_24_1 aw/out/a_m5_3_8_2_2.0` writes `ana2_s7_11.md`. The dirs were renamed with an `a_` prefix after the job.

| config (beta/P/mode/tau) | set | vote | oracle@8 | per-att | distinct L1/8 | distinct ans/8 | tok/q | fixes/breaks |
|---|---|---|---|---|---|---|---|---|
| base r1f | dev | 89.08 | 93.58 | 81.00 | 4.00 | 1.79 | 1235 | |
| 3/24/0 (soft, all earlier attempts) | dev | 88.58 | 93.83 | 80.64 | 5.94 | 1.84 | 1221 | 17/23 |
| 30/24/0 (hard) | dev | 88.50 | 94.50 | 79.97 | 7.89 | 1.88 | 1238 | 21/28 |
| 3/24/1 (only if earlier answers unanimous) | dev | 89.17 | 93.92 | 80.34 | 5.31 | 1.84 | 1226 | 18/17 |
| 3/8/2/2.0 (plurality group with lg < 2, first 8 tokens) | dev | 88.67 | 94.25 | 80.67 | 4.86 | 1.83 | 1235 | 18/23 |
| base r1f | dev2 | 86.33 | 94.50 | 79.21 | 4.21 | 1.91 | 1251 | |
| 3/24/0 | dev2 | 85.83 | 93.17 | 78.83 | 6.12 | 1.94 | 1255 | 14/17 |
| 30/24/0 | dev2 | 86.50 | 93.83 | 78.38 | 7.89 | 1.96 | 1274 | 13/12 |
| 3/24/1 | dev2 | 86.00 | 93.67 | 78.88 | 5.53 | 1.90 | 1248 | 6/8 |
| 3/8/2/2.0 | dev2 | 87.67 | 94.00 | 79.52 | 5.09 | 1.92 | 1260 | 17/9 |

- **The implant works mechanically.** With beta=30, almost all 8 first lines are distinct (7.89/8). The number of distinct answers barely moves (1.79 → 1.88).
- **There is no consistent vote gain.** Hard repulsion follows the ceiling's prediction: oracle@8 rises (dev +0.9), while per-attempt accuracy and the vote fall. Only 3/8/2/2.0 on dev2 shows a gain (+1.33, net +8), and on dev it is −0.42.
- Job **j23375** (`job_m5_b`): `<queue-submit> $PWD 2400 job_m5_b -- env SEEDS=7,11,13,17 bash m5job.sh 3/8/2/2.0 3/24/1`. It runs 4 seeds in one process per set, like r1f. Seeds 7 and 11 should reproduce the `a_` rows, which doubles as a determinism check.

### 05:45Z: j23375, 4 seeds (7/11/13/17, one process per set, as in r1f) → CLEAN NEGATIVE
- **Determinism:** seeds 7 and 11 reproduce the j23368 `a_` rows token for token (600/600 and 300/300 for both configs and both sets).
- Analysis: `python3 ana2.py gsm_stack/implants/aw/dev_tree/out/r1f aw/out/m5_3_8_2_2.0 aw/out/m5_3_24_1 --seeds=7,11,13,17` writes `ana2_s4.md`.

| config | set | vote (Δ) | oracle@8 | per-att | distinct L1/8 | distinct ans/8 | tok/q | fixes/breaks (net) |
|---|---|---|---|---|---|---|---|---|
| base r1f | dev | 88.79 | 93.75 | 80.89 | 4.03 | 1.80 | 1229 | |
| 3/8/2/2.0 | dev | 88.29 (−0.50) | 94.08 | 80.51 | 4.84 | 1.83 | 1233 | 38/50 (−12) |
| 3/24/1 | dev | 88.88 (+0.08) | 93.96 | 80.45 | 5.31 | 1.83 | 1230 | 38/36 (+2) |
| base r1f | dev2 | 85.92 | 93.75 | 79.23 | 4.16 | 1.88 | 1252 | |
| 3/8/2/2.0 | dev2 | 86.42 (+0.50) | 93.67 | 79.12 | 5.08 | 1.91 | 1254 | 26/20 (+6) |
| 3/24/1 | dev2 | 85.67 (−0.25) | 93.67 | 78.73 | 5.54 | 1.91 | 1249 | 14/17 (−3) |

### Conclusion
**First-step diversity does not help the vote.**
- Step 1 (CPU) gives the reason: two attempts with different first lines still reach the same answer 70% of the time, and the fork comes later.
- An ideal first-step-distinct resampler moves the vote by only about ±0.6 (fixed rows) and costs per-attempt accuracy.
- The all-weights implant does what it is built to do. Distinct first lines per 8 attempts rise from 4.0 to 5.3–7.9, but distinct answers rise only from 1.80 to 1.83–1.88.
- Every config lands within noise of base on the vote, which is about ±0.5, with dev and dev2 disagreeing in sign. Oracle@8 rises by at most +0.9 (hard repulsion, 2 seeds) while per-attempt accuracy falls by 0.4–1.0.
- tok/q is unchanged (±20).
- Lane closed. Code: `aw/stk/m5div.py`; hooks in `aw/m5_comp_serial.diff`; fuzz: `aw/test_m5.py`; analysis: `ana1.py`, `ana2.py`, `m5lib.py`; rows: `aw/out/`.
