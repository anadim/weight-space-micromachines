# Reading, planning and bookkeeping implants (gsm_push, 2026-11-20)

> **Update:** the user now requires single-call, forward-only, compiled implants. The restart cascade (§4) no longer qualifies. See **Part 2 (§7-8)** for the compiled sanity implant, the forget-mask variant, and the failure-by-failure debugging.

**Question.** About 70% of the RL+calc host's GSM8K failures are reading errors with exact arithmetic: wrong plan, dropped
quantity, reversed relation, units. Can a generic implant that acts on reading (re-read the question, a quantity ledger,
a units table, a plan-first step, a goal reminder) fix some of them on the frozen host?

**Short answer.**
- **No unconditional reading implant is a real win.** Re-reading the question, "We need to find ...", listing
  quantities, or restating the goal at `####` all change about 15% of answers on every row (40-55 fixes and 30-55 breaks
  per 600). On dev and dev2 some variants looked like +2.5 to +5 pp. On a third, never-used slice of split B (dev3, 562
  items) every one of them **lost** 1.8-2.5 pp. A single T = 1 sample of the unmodified host moves accuracy by the same
  amount (+2.8 / +2.3 / −2.8 on dev / dev2 / dev3). These are reshuffles, not gains. At T = 1 (4 samples per item, 1,462
  items) re-reading is worth +0.8 pp, not significant. On the base host it hurts (−0.8 to −4 pp).
- **The quantity ledger does not work.** A question number that never appears in the answer flags 5% of rows, at 33-35%
  precision (base rate of wrong answers 18-25%). Nudging the host with the unused clause fixes 0 rows on the RL+calc host
  (0 fixes / 0 breaks on dev, 1 / 1 on dev2), because the host re-derives the same answer. Restarting on this trigger
  gains 0 to +1 pp, not reproducible across slices. Drop.
- **The units table has nothing to act on.** The unit the question asks for differs from the unit of the last computed
  value (same dimension) in 6 of 806 wrong answers and 4 of 3,694 right ones (split-A dumps). The upper bound is about
  0.1 pp. Dropped offline, no GPU run.
- **One thing works, and it is a reading action, not a reading trigger: a reworded restart on the answer-sanity trigger.**
  When the final answer is missing, negative or non-integer (the sanity agent's exact trigger), restart the answer
  greedily with "Answer: We need to find" (plan-first), then, if still flagged, with the re-read prompt, then one T = 1
  sample. Real decoding, RL+calc host (s2), 1,462 split-B items: **77.50 → 80.03 (+2.53 pp), 37 fixes, 0 breaks**, 5.2% of
  rows fire, +12.7% tokens. The sanity agent's action (up to 3 T = 1 restarts), run in the same harness, gives +1.98 (29
  fixes). Head to head the reworded cascade is ahead 16 to 8 rows (p = 0.15). On the base host the gap is large: +3.22 vs
  +1.11 (26 to 7 rows, p < 0.01). Keep, as the *action* of the sanity implant.

## 1. Setup

- **Host.** RL+calc `a2_u300_s2` (sha256 `ee82ee1e…`, copied from A100 to `H100:<work>/gsm_push/reading/ckpt_a2s2`).
  Same 8-shot prefix, "=" calculator implant on, greedy, 320-token budget, `common.score` scoring. Second checks:
  RL+calc seed 1 (`a2_u300_s1`, same recipe) and the base host u0 + calculator (`--host_ckpt base`).
- **Data.** GSM8K train split B (md5 % 10 in {6, 7}; never in the RL pool), minus the 8 shots: 1,462 items.
  `random.Random(0).sample(B, 900)`: first 600 = **dev**, last 300 = **dev2** (`reading/dev_split.json`). The other 562
  split-B items = **dev3**. I added dev3 after dev2 because dev and dev2 were used to pick variants. It is the honest
  confirm set. GSM8K test was not touched. GSM-Hard-like check: 300 scaled split-B train items.
- **Harness.** `reading/rd_eval.py` is exec'd inside `dig_san/tw_dump_lean.py --eval_ckpt` (the sanity agent's lean
  `train_wiring` copy), so it uses the RL engine with its CUDA-graph decode and the exact calculator implant.
  - *Prompt variants*: a fixed write at the start of the answer (or in the question block) is one constant token
    sequence on every row, so it is implemented as that sequence in the prompt. It is exactly what a step-0 write
    queue would produce.
  - *Resume variants (nudges)*: pass 1 = off; on trigger rows the context becomes question + the host's answer up to its
    stop token + the nudge. The host continues greedily, and the answer is the first `####` after the nudge.
  - *Restart variants (cascade)*: on trigger rows, generate again from the question with the next prompt. The old attempt
    leaves the context, as in the sanity agent's restart.
  - Analysis: `reading/analyze.py` (fixes and breaks vs off, McNemar p) and `reading/cascade.py` (restart cascades
    simulated from per-variant runs). The cascades were then re-run for real (`casc_*` variants). Real and simulated
    agree within 1 row per slice, except where T = 1 seeds differ.
- **Compute.** H100, one process per job at `--mem_frac 0.09-0.15` (7-12 GB), my own watchdog (kill mine if GPU free
  < 3 GB). Never touched other jobs. Two of my jobs hit their own cap (resume variants with 1,000-token contexts at
  0.09) and were re-run at 0.15. Disk: 1.5 GB (checkpoint copy); pod free ≥ 15 GB throughout.

Baselines (greedy, off): s2 dev **74.83**, dev2 **77.00**, dev3 **80.60** (all 1,462: 77.50). s1: 75.50 / 79.67.
Base host: 63.17 / 63.67. The 6 pp spread between slices is itself a warning about noise at n = 300-600.

## 2. Results per idea (RL+calc host s2; Δ in pp vs off; fixes / breaks)

| idea | variant | dev (600) | dev2 (300) | dev3 (562) | fire | tokens | verdict |
|---|---|---|---|---|---|---|---|
| control: one T=1 sample | `T1s0` | +2.83 (43/26) | +2.33 (22/15) | **−2.85** (27/43) | all | +0% | the noise floor |
| 1. re-read (RE2), query only | "Read the question again: q" | +1.67 (48/38) | +3.33 (25/15) | **−1.96** (30/41) | all | −5% answer, +q prefill | drop |
| 1. re-read, shots also in RE2 form | `re2shots` | +2.50 (51/36) | +5.33 (25/9) | **−2.49** (30/44) | all | −10% | drop |
| 1. same, "Question again:" | `re2nshots` | +3.33 (56/36) | +4.67 (25/11) | **−1.78** (35/45) | all | −10% | drop |
| 1. re-read 3× | `re3shots` / `re3q` | +1.00 / −1.17 | +6.33 / +1.33 | – | all | −12% | drop |
| 1. restate the question inside the answer | `re2a` | −33 (s1) | −38 (s1) | – | all | – | breaks format, drop |
| 1. restate only the asked sentence | `re2ask` | +0.83 | +3.67 | – | all | | drop (same churn) |
| 4. plan-first "Answer: We need to find" | `plan1` | +3.00 (49/31) | +5.00 (33/18) | **−2.31** (42/55) | all | +11% | drop as unconditional |
| 4. plan list "the quantities given and what is asked" | `plan2` | −1.00 | −1.67 | – | all | +3% | drop (s1: −3.5 / −6.7) |
| 4+1. RE2 shots + plan-first | `re2q:re2plan` | +3.00 | +3.67 | – | all | +2% | drop (same churn) |
| 2. ledger nudge, unused clause quoted | `nudge_unused_clause` | +0.00 (0/0) | +0.00 (1/1) | – | 5.5% / 5.0% | +13% | drop |
| 2. ledger nudge, unused number ≥ 3 | `nudge_unused3_number` | +0.00 (0/0) | +0.00 (0/0) | – | 4.0% / 4.3% | +7.5% | drop |
| 2. ledger → restart (re-read prompt) | `unused->re2q` | +0.67 (6/2) | +0.67 (2/0) | – | 5.5% | +6% | drop (dev3: +0.36 with re2shots) |
| 5. goal reminder at `####` ("The question asks: …") | `nudge_all_goal` | +0.17 (3/2) | +0.00 (1/1) | – | 100% | **+118%** | drop |
| 5. "answer ≠ last computed value" → rethink | `nudge_notlast_rethink` | +0.17 | +0.00 | – | 1.5% | +3% | drop |
| 3. units table | offline only | – | – | – | 0.2% of rows | – | drop (≤ 0.1 pp bound) |
| **sanity trigger → reworded restart cascade** | `casc_plan1-re2q-T1s0` | **+2.83** (17/0) | **+2.33** (7/0) | **+2.32** (13/0) | 5.2% of rows | +12.7% | **keep** |
| same trigger, sanity agent's action (≤ 3 T=1 restarts) | `casc_T1s0-T1s1-T1s2` | +2.33 | +2.67 | +1.25 | 5.2% | +14.7% | reference |
| same trigger, one plan-first restart only | `casc_plan1` | +1.67 | +2.33 | +2.14 | 5.2% | +7.6% | cheapest good option |

Seed-1 RL+calc host (dev / dev2):
- re2shots +3.00 / +1.33; re2q +2.17 / +0.67; plan1 +0.17 / +2.67; plan2 −3.50 / −6.67.
- Ledger nudge +0.50 / 0.00.
- Cascade +2.33 / +1.00, vs T = 1 ×3 at +1.50 / +1.67.

Base host (dev / dev2):
- re2q −1.83 / −4.00; re2shots −0.83 / −3.67; plan1 −5.17 / −1.33.
- **Cascade +2.67 / +4.33**, vs T = 1 ×3 at +1.33 / +0.67.

T = 1 check of re-reading (4 samples per item, mean accuracy, s2):
- off 76.92 / 76.75 / 77.05; re2shots 77.25 / 79.33 / 77.49.
- Pooled over 1,462 items: **+0.83 pp**.
- Greedy pooled over 1,462 items: re2shots +1.1 pp, 106 fixes vs 89 breaks, p ≈ 0.25.

Scaled (GSM-Hard-like) split-B 300: re2q 60.67 vs off 60.00.

## 3. Why the reading implants fail on this host

- **Every rewording is a coin flip on about 15% of the items.** The RL host's greedy answer is fragile. Any change of
  context (a T = 1 draw, one extra line, the question twice) flips 13-17% of answers, in both directions in roughly equal
  numbers. A real "reading" gain would show up as fixes ≫ breaks on fresh data. Not one variant has that on dev3.
  Lesson for the loop: at n = 600, differences of ±3 pp between whole-prompt variants are within the spread of one
  T = 1 sample. A variant picked on dev + dev2 needs a third slice (here dev3 reversed every prompt "win").
- **Nudges after the answer do nothing: the host re-derives its own answer.** Examples: after "Wait, I did not use this
  part of the question: 'She has 6 classes per day'", the host writes "Answer: The first class has 20 students. …" again
  and commits the same 126. This is the stable-misconception effect (failures §2d; sanity §1, where 41 of 75 rethinks
  repeated the same impossible answer). A continuation keeps the old attempt in context, and the host copies it.
- **The exact triggers for reading errors are weak.** Unused number 33-35% precision, entity never named 24%, unit
  mismatch 60% precision but 0.7% recall, answer ≠ last computed value 55% precision but 1-2% recall (split-A dumps,
  `reading/trig_offline.py`, `units_offline.py`). As the failure analysis said (§4.4), a dropped *entity* or *event* is
  not a missing printed number.
- **What does transfer is a *different reading* after a *reliable* trigger.** On rows the sanity trigger flags (98-99%
  of them wrong), any rewording only has upside. A deterministic reworded restart fixes more than a random T = 1 restart,
  because it changes how the question is read, not only the sampling noise. Of the 24 cascade fixes on s2 dev + dev2,
  17 came from the plan-first restart alone, 4 from the re-read restart, and 3 from the final T = 1 step. On the base
  host the gap over T = 1 is 3× (26 vs 7 rows), probably because its T = 1 samples are much worse than its greedy
  (failures §3a: 37.6 vs 65.8 on train).

## 4. The kept implant: reworded restart on the sanity trigger

- **f(x).** Trigger (the sanity agent's exact `####`-DFA): the host is about to end its answer and the committed number is
  missing, negative or non-integer. Action: rewind the KV window to the question, as `SanEngine` restart does, and
  regenerate greedily, with a fixed token sequence written at the start of the new answer:
  1. ` We need to find` (plan-first: the host must name the target before it computes);
  2. if still flagged: the question again ("Read the question again: q"), written by an exact copy of the question span;
  3. if still flagged: one T = 1 sample of the plain prompt.
  Generic meaning: *when an answer fails an exact plausibility check, re-solve from a different reading of the problem,
  not from the same reading with noise.* It helps wherever a cheap verifier can reject answers (unit or type checks,
  integer-only domains, constraint checks), not only on GSM8K.
- **Cost.** It fires on 5.2% of rows (base host 8.3%). +12.7% tokens overall (+17% s1, +20% base), about the same as 3
  T = 1 restarts. Each restart is ≤ 320 tokens. The re-read restart needs the question copied once, about 70 prefill tokens.
- **Silent rows are untouched.** Rows that do not fire keep their off answer (0 breaks on 1,462 s2 rows; 2 breaks on
  900 base-host rows).
- **Numbers (real decoding).**

  | host | items | off | cascade | fixes / breaks | T = 1 ×3 restart | cascade vs T=1 ×3 (rows) |
  |---|---|---|---|---|---|---|
  | RL+calc s2 | 1,462 (dev+dev2+dev3) | 77.50 | **80.03 (+2.53)** | 37 / 0 | 79.48 (+1.98) | 16 vs 8, p = 0.15 |
  | RL+calc s1 | 900 | 76.89 | 78.78 (+1.89) | 17 / 0 | 78.44 (+1.55) | 8 vs 5, p = 0.58 |
  | base u0 + calc | 900 | 63.33 | **66.56 (+3.22)** | 31 / 2 | 64.44 (+1.11) | 26 vs 7, p < 0.01 |

- **How to stack.** It replaces the *action* of the answer-sanity implant (same trigger; restart instead of T = 1
  restart). It adds nothing on rows that the trigger does not flag. The expected increment over the sanity agent's best
  restart is about +0.5 pp on the RL+calc host and about +2 pp on the base host.
  - It is compatible with calculator-v2: a different blind spot, independent rows.
  - It is compatible with a selection/verifier stage, which should run after it.
  - For a GSM-Hard eval, use it with the missing-only class (`SAN_CLASSES=3`), because GSM-Hard golds are often
    non-integer.
  - To implement it in `SanEngine`: the restart already rewinds `slot` to `s_qlen` and draws the first token from
    `s_z0`. For the reworded restart, queue the fixed token sequence (as in phrase mode) right after the rewind, and run
    greedy instead of `samp`. Steps 1-2 need no second prefix cache.
- **Caveats.**
  - The step order (plan1 first) was chosen on dev + dev2 among about 10 restart prompts. On dev3, which was not used
    for that choice, plan1 alone gives +2.14 and the full cascade +2.32, vs +1.25 for T = 1 ×3.
  - The advantage over T = 1 restarts on the RL+calc host is not significant on its own (p = 0.15).
  - T = 1 seeds differ between simulated and real runs.

## 5. Keep / drop

- **Keep:** reworded restart cascade as the sanity trigger's action (§4).
- **Drop:**
  - unconditional re-read (all forms);
  - unconditional plan-first / plan list;
  - restating the question inside the answer;
  - quantity-ledger nudge and ledger restart;
  - goal reminder at `####`;
  - "answer ≠ last value" rethink;
  - units table (offline: no mass);
  - entity ledger (offline: 24% precision);
  - latent write of "what is asked": not attempted. Its text version (`re2ask`, goal reminder) showed only churn, so a
    fitted residual write had no signal to aim at in this time box.

## 6. Files

- Code: `gsm_push/reading/`
  - `rd_eval.py`: pod hook with all variants, incl. `casc_*`, `nudge_*`, `*@T1g4`, `SHOTFMT:PROMPT`.
  - `rd_trig.py`: exact triggers and nudges.
  - `run_rd.sh`: job and watchdog.
  - `analyze.py`, `cascade.py`.
  - `trig_offline.py`, `units_offline.py`: offline trigger precision on split-A dumps.
  - `dev_split.json`.
- Outputs: `gsm_push/reading/out/<job>/<variant>.jsonl`, also on `H100:<work>/gsm_push/reading/out/`.
  - Per-row fields: `ok`, `pred`, `ntok`, `fire`, `path`, and text with ⟦calc⟧ and 【RESTART】 marks.
  - Jobs: `s2_*` / `c_s2*` = main host; `s1_*` / `c_s1` = seed 1; `base_A` / `c_base` = base host; `*dev3*` = held-out slice.
- Checkpoint copy: `H100:<work>/gsm_push/reading/ckpt_a2s2/` (1.5 GB; delete when the push is over).

---

# Part 2 (after the scope change): single-call, compiled, forward-only implants

**New rule from the user.** Every implant must work in a single generation call. It must be compiled and forward-only:
no rewind and no restart. The reworded-restart cascade of §4 restarts, so it no longer qualifies; its numbers stay only
as a reference.

**Short answer.**
- **I built a compiled answer-sanity implant and proved it exact.**
  - 300,000 fuzz strings: the token-level circuit matches the character automaton at every token boundary. Its
    "certain impossible" flags agree with the scorer's regex for every continuation tried, and every regex-impossible
    answer is caught.
  - Silent rows are bit-identical to the plain engine: 1,462 / 1,462 with the implant off, and every non-firing row
    with it on.
- **Appending a phrase alone gives little, as expected.** The best phrasing is typed: 77.50 → 78.11 (+0.62, 9 fixes /
  0 breaks) on all of split B. "We need to find" (+0.34) does not reproduce its restart success forward-only. The
  host re-derives the same impossible answer from the attempt that is still in its context.
- **A forget mask keeps most of the restart gain in one call.** On a fire, the failed attempt is masked out of attention.
  This uses the same latch registers and a masked copy of the attention kernel; the KV slot still moves forward.
  The writer then forces " We need to find".
  - RL+calc s2: 77.50 → **79.21 (+1.71), 25 fixes / 0 breaks**, +7% tokens. dev / dev2 / dev3: +1.17 / +3.00 / +1.60.
  - RL+calc s1: +1.67 / +1.33 / +2.31 (27 / 0). The plain typed phrase on s1 gives +0.67 / +1.00 / 0.00.
  - The restart cascade gave +2.53 and a single plan-first restart 29 fixes, so the forget mask keeps about 70% of the
    restart gain without a second call.
  - **Caveat:** it changes attention, not only the residual. Functionally it is a restart done in place by masking. I
    flag it so the coordinator can decide whether it counts under the "no rewind" rule.
- **Failure-first debugging** (§8): 42 dev600 failures read and diagnosed one by one.
  - Only about 25% of them have an exact, small, side-attachable fix:
    - impossible final answer: 9;
    - calculator coverage or format (calc-v2's job): 4;
    - non-integer count intermediate: 2;
    - zero answer: 1.
  - The rest are plan, world-knowledge and relation errors that no cheap trigger separates from correct reasoning.
    Measured trigger precisions are below.

## 7. The compiled sanity implant (`reading/comp_sanity.py`, `comp_eval.py`)

**Circuit.** Runs once per decode step, reading the INPUT token (the token the host wrote last), with fixed tensors only:
- **reader:** `CLS[tok]` is a gather of a fixed row. It maps the token to one of 26 token classes; tokens in the same
  class act identically on the sanity agent's 34-state `#### number` automaton.
- **DFA:** a one-hot state register `s' = s @ TC[CLS]`, with a fixed 0/1 transition matrix per class (26 × 34 × 34).
- **certainty rows `CN`, `CF`:** computed by exhaustive reachability on the character automaton. From these states,
  *every* continuation makes the scorer's first `#### number` negative (6 states) or non-integer (3 states). In
  practice the implant fires at the first nonzero digit after `-`, or the first nonzero digit after `.`.
- **gates:** ReLU on 0/1 wires: `fire = relu(c + idle + under + room + on − 4)`.
  - idle: the pointer register is at 0.
  - under: fewer than 2 fires so far.
  - room: the position counter is below the per-row budget.
- **writer:** a pointer register over the phrase rows, one bank per class. On fire `p' = START_bank`; otherwise
  `p' = p @ SHIFT`. The write is `G · p'[1:] @ DIR`, added to the residual before the final norm.
  - `DIR[k]` is a unit-RMS direction that makes phrase token k the argmax of the readout. It is solved from `E[t]/lnf`
    and checked at setup.
  - G = 3e4. Every written span equals the phrase exactly: 0 mismatches over all runs (`spans_not_exact_phrase = 0`).
- **reset:** on fire the DFA returns to its start state, so the next `####` after the phrase is read fresh.
- **forget-mask variant:** on fire, a latch stores `hi = slot + 1` (lo = the question's end, already a register). The
  suffix attention uses a masked copy of `hostkern.attn_suffix_kernel` in which keys in [lo, hi) get −inf. With
  hi = −1 (never fired) the kernel does the same math as the plain kernel, which is why silent rows are bit-identical.
- **Disclosed non-weight pieces:**
  - the decode loop and its per-row token budget (+320 tokens per fire, as in the sanity engine);
  - the registers are stored as indices between steps (argmax of the one-hot) so the engine can compact rows;
  - the forget mask is an attention-mask write, not a residual write.
- **Not covered:** a *missing* answer (8 of 76 flags on split B). Detecting it needs the output stop token. The stack
  agent's stop-token hook covers it.

**Exactness** (`fuzz300k_v2.log`, 300,000 strings, half of them built around `####` lines with signs, commas and decimals):
- token-level state mismatches vs the character automaton: **0**;
- unsound fires (a continuation's regex class differs from the fired class): **0** over 56,580 fires × 3 suffixes;
- incomplete (regex says negative or non-integer, the circuit never fired): **0**;
- zero class: 9,081 fires, all at a regex match whose first digit is 0.

**Results**, RL+calc s2, greedy, all of split B. Δ vs off (74.83 / 77.00 / 80.60); fixes / breaks. Every non-firing
row is bit-identical to off.

| variant | dev (600) | dev2 (300) | dev3 (562) | all 1,462 | fixes / breaks | p | rows fired | tokens |
|---|---|---|---|---|---|---|---|---|
| phrase "Hmm, wait. Let me rethink this." | +0.67 | 0.00 | +0.36 | +0.41 | 6 / 0 | 0.03 | 4.4% | +16% |
| phrase typed ("can't be negative" / "should be a whole number" + "We need to find") | +1.00 | +0.67 | +0.18 | **+0.62** | 9 / 0 | 0.004 | 4.4% | +13% |
| phrase "Wait, that can't be right. We need to find" | +0.67 | 0.00 | +0.18 | +0.34 | 5 / 0 | 0.06 | 4.4% | +18% |
| phrase "Hmm, wait. Let me read the question again. We need to find" | 0.00 | +0.33 | +0.18 | +0.14 | 2 / 0 | 0.5 | 4.4% | +3% |
| **forget mask + " We need to find"** | **+1.17** | **+3.00** | **+1.60** | **+1.71** | **25 / 0** | < 0.001 | 4.4% | +7% |
| + zero class (fires on `#### 0`) | +1.17 | +3.00 | +1.60 | +1.71 | 25 / 0 | | 4.9% | +8% |
| reference: restart cascade (§4, not allowed now) | +2.83 | +2.33 | +2.32 | +2.53 | 37 / 0 | | 5.2% | +13% |

- On the coordinator's common dev600 / dev2 (Random(0).shuffle of split B, reconstructed in `reading/common_split.json`):
  - off 80.33 / 77.00;
  - typed phrase 80.67 / 77.33;
  - forget mask **82.50 / 78.67** (13 / 0 and 5 / 0).
- Seed-1 RL+calc host:
  - typed +0.67 / +1.00 / 0.00 (7 / 0);
  - forget mask (+zero) **+1.67 / +1.33 / +2.31** (27 / 0).
- Why the phrase is weak: after "We need to find", the host writes "out how much time it actually took … Let's recheck
  the calculations" and copies its old lines. With the typed phrase, 17 of 65 fired rows commit the same number again,
  and most of the rest commit a different impossible one. The stack agent saw the same on test (62 of 80 fired rows
  still impossible). Masking the old attempt removes what it copies from.
- **Zero class: drop.** On split B, 7 wrong greedy answers are `#### 0` and no right one is; 1 of 7,473 train golds
  is 0. Firing on them fixed 0 rows here, and also 0 in the stack agent's hook.

**Recommendation.**
- **Keep the compiled trigger and writer.** They are exact, silent, single-call, and fully in tensors. Use the typed
  phrase if only residual writes are allowed (+0.6).
- **The forget mask is worth +1.7 and is the only forward-only form that keeps most of the restart gain.** Its
  admissibility (it is an attention-mask write) is the coordinator's call.
- For stacking with the stop-token hook (which covers missing answers): use one trigger, and route the suffix attention
  through `attn_suffix_mask_kernel` with `s_fr = slot + 1` on fire.

## 8. Failure-first debugging: 42 dev600 failures, one by one

Common dev600, RL+calc s2, greedy (`reading/debug/fail42_dump.txt`; 118 dev600 failures, the first 42 not read before).
For each failure:
- **first wrong step**: what went wrong;
- **side fix**: what small attached thing could have prevented it, and whether it is exact (an exact check) or soft (needs judgment).

| # | idx | first wrong step | side fix |
|---|---|---|---|
| F0 | 1470 | computes 5 × 30 = 150, then adds only Tue + Fri (35 + 55); answer 1.5 h | **sanity** (non-integer); dangling 150 |
| F1 | 262 | "space for 4 more" read as 12 + 4 (relation reversed) → 1.5 | **sanity** |
| F2 | 3995 | "6 square inches" per tile taken literally (gold: 6-inch tiles), wording ambiguous | none |
| F3 | 1951 | counts day 2 twice in the day count (plan) | none |
| F4 | 6620 | computes the $2,000 total, then uses the weekly $500 per person (wrong earlier number) | dangling-result ledger (soft) |
| F5 | 2549 | forgets 4 shoes per horse (world knowledge); 28 kg computed and unused | none |
| F6 | 5608 | ignores the 18 left-over melons (stops a step early) | unused-number ledger (soft) |
| F7 | 89 | "There are 5 Fridays in a week" | fact table (soft: see below) |
| F8 | 6899 | sets sum = 131 and product = 131 separately; writes "inconsistency" itself | none (self-doubt text) |
| F9 | 2980 | "two for every 10" read as 20 / 10 = 2 per customer | none |
| F10 | 6959 | transfer bookkeeping; reaches −10 dolls mid-way, answers 10 | negative-intermediate trigger (soft, 50% precision) |
| F11 | 554 | drops "50% play it" | unused-number ledger (soft) |
| F12 | 3645 | garbled gas cost line ("0.8 x $3.75 = $3."), 56.3 | **sanity** |
| F13 | 7408 | average per minute instead of per hour | **sanity** (0.0667) |
| F14 | 239 | host-written 7 + 20 + 20 + 20 = 77 (should be 67) | **calc-v2** (4-operand chain); sanity |
| F15 | 4621 | "6 − 3 = 3 x 8 = 24 8-string guitars", then 24 × 8 | **calc format** (chained `=`; calc-v2) |
| F16 | 1227 | invents 10 crayons per box; the 85 is never used | unused-number ledger (soft) |
| F17 | 5781 | baguette not × 4 | **sanity** (73.5) |
| F18 | 1872 | stops before × 20 birds | unused-number ledger (soft) |
| F19 | 3346 | takes 1/4 of the remainder instead of 1/4 of X (relation) | none |
| F20 | 6938 | pools all car types, 20 / 6 = 3.33 stations, then floors | **non-integer count intermediate** (exact, 83-93% precision) |
| F21 | 6002 | drops the rent sharing | none |
| F22 | 3274 | drops the third week of eating (event dropped) | none |
| F23 | 6282 | ignores the 3 in/day drainage | unused-number ledger (soft) |
| F24 | 5440 | per-week saving vs a two-week lunch (rate period) | **sanity** (67.5) |
| F25 | 3406 | couple room = 1 person (world knowledge) | none |
| F26 | 7400 | treats 28 · 7/4 as 1/7 of the value, then × 7 | none |
| F27 | 3022 | misreads the loan; host-written comma division | **sanity**; calc-v2 |
| F28 | 7228 | uses Matt's 2nd-year 75 as his 1st year (wrong reference) | **sanity** (412.5) |
| F29 | 3749 | calculator returns 21 for "42 / 6 x $3.00"; the host multiplies by 3 again | **calc format** (calc-v2) |
| F30 | 518 | "if Merill has 45" (relation); 22.5 marbles | **non-integer count intermediate** |
| F31 | 5333 | counts all 100 slices plus 150 pieces (double count) | none |
| F32 | 4569 | buys feed for 12 months, not the 4 left; the computed 8 months is unused | dangling-result ledger (soft) |
| F33 | 4504 | drive / cool-down cycle structure | none |
| F34 | 5545 | multiplies the fern by 4 pots, then the total by 4 again | none |
| F35 | 1836 | uses Caleb's original 11, not 22 | none |
| F36 | 5947 | brother born 1970 − 20 (direction) | none |
| F37 | 2692 | host-written 90 + 98 + 92 + 94 = 384 (should be 374) | **calc-v2** |
| F38 | 2796 | uses Julian's 48 boys for Boyd; 64 computed and unused | dangling-result ledger (soft) |
| F39 | 2966 | doses × 12 "periods" (double count) | none |
| F40 | 986 | merges both swimmers, difference 0 | zero-answer class (exact) |
| F41 | 1091 | ignores ½ bottle per day | none |

**Patterns** (42 failures, primary fix):

| pattern | n | exact trigger? | trigger precision (split B greedy / split A greedy + 8 samples) | outcome |
|---|---|---|---|---|
| impossible final answer | 9 | yes | 100% / 100% | the compiled implant (§7) |
| calculator coverage / format | 4 | yes | – | calc-v2 agent's scope |
| non-integer count intermediate | 2 | yes (`= x.y <count noun>`) | 83% / 93%, but only 4 / 14 beyond sanity | not built: ≤ 0.2 pp upper bound |
| zero answer | 1 | yes | 100% / 100% | built; 0 fixes (also 0 in the stack agent's hook): drop |
| unused question number | 5 | yes | 33-35% | nudges 0 fixes, restarts ~0 (§2): drop |
| dangling computed result | 3 (+3 overlapping) | yes | 27% beyond sanity | too weak: drop (offline) |
| negative intermediate | 1 | yes | 50% beyond sanity | drop (offline) |
| world-knowledge fact | 3 | fact table | 4 of 12 "wrong" stated facts are on wrong answers; the rest are problem-specific ("5 days in each week" of work) | a table would break more than it fixes: drop (offline) |
| plan / relation / double count / dropped event | 14 | no | – | no small exact fix; needs a verifier or more RL |

Scripts:
- `reading/debug/trig2_offline.py`: dangling, non-integer count, negative intermediate and zero triggers.
- `reading/debug/facts_offline.py`: the world-knowledge fact table.

The verdict of the debugging pass: about a quarter of the remaining failures can be caught by a small exact attachment.
The largest of these, the impossible final answer, is now compiled and silent. The rest is reading the problem, and
nothing exact separates a wrong plan from a right one.

## 9. Weight-space forget mask (replaces the masked kernel)

**The user's condition:** the forget mask must "somehow happen in weight space". So I replaced the kernel's key-range
mask with fixed extra attention dimensions that read an attempt register. Code: `reading/comp_sanity.py`
(`wbias` path, `attn_suffix_bias_kernel`), and `reading/forget_ws.py`, a drop-in `wrap_stack(cls)` for the stack
agent's `stack3_hook.py`.

**Construction.**
- **Attempt register.** `a` is the number of fires so far (`s_nf`), a one-hot over E = 4 (attempts 0-2, padded to a
  power of 2).
- **Side coordinates.** Two one-hots live in coordinates the host never reads or writes; this is equivalent to appending
  them to the residual with zero rows and columns in every host matrix.
  - The key-attempt label of the token at this position: the attempt *before* a fire at this step, so the token at a
    fire slot belongs to the abandoned attempt.
  - The current attempt.
  - Prompt and question positions carry zeros: they are "shared".
- **Extra head dimensions** in every layer and head. They are not rotated by RoPE.
  - `k_extra = W_K · onehot(label)` with `W_K = I₄`.
  - `q_extra = M · W_Q · onehot(a)` with `W_Q = −(J₄ − I₄)`.
  - Score of key j: `q·k_j/√d + q_extra·k_extra(j)`. The extra term is −M for keys of an abandoned attempt and 0 for
    keys of the current attempt or shared keys.
  - The extra term is computed as a separate 4-dim dot product and added to the score, so with no fire it is an exact
    +0.0.
- There is no key-range mask in the kernel and no KV deletion. The attention kernel is the plain one plus 4 more
  dimensions in its dot product.
- **In the stack hook** (`forget_ws.py`) the label is computed from the stack agent's own registers as
  `a − [j < s_fr]`. This is exact: `s_fr` is the forget end of the latest fire, and labels only ever increase.
- **Cost.**
  - Parameters: `W_K` and `W_Q` are 4 × 4 each. If materialised per layer and per head (28 × (16 q + 8 kv) heads),
    that is 10,752 numbers, about 0.002% of the host's 596M. They are identical in every layer and head, so one copy
    is used.
  - Memory: 4 floats per cached position (one side key cache serves all layers, since the side coordinates are
    unchanged through the network), plus 4 per row.
  - Wall time (1,462 items, one H100, B = 32):
    - off: 47.8 s;
    - masked kernel: 72.8 s;
    - weight-space: 73.7 s, +1.2% over the masked kernel.
    - The +52% over off comes from the long tail of fired rows (+7.5% tokens), not from the extra dims.

**Verification.**

| check | result |
|---|---|
| greedy tokens, weight-space vs masked kernel, all 1,462 split-B rows (M = 50) | **1462 / 1462 identical** (fired rows 65 / 65) |
| same at M = 30 and M = 1e4 | 1462 / 1462 identical, both |
| silent rows vs the plain host (weight-space mode on, implant not fired) | **1397 / 1397 identical** |
| largest attention weight on an abandoned key (upper bound: the softmax numerator relative to the running max, atomic max over all steps, layers and heads) | **9.7e-15** at M = 50; 4.7e-6 at M = 30; 0 at M = 1e4 (exp underflows) |
| written phrase spans equal to the phrase | 95 / 95 |
| inside the stack agent's `stack3_hook.py` (calc v2 fwd + compiled sanity + stop-token missing trigger), weights vs kernel | dev600 **600 / 600** identical (27 / 27 fired rows); dev2 **300 / 300** (12 / 12) |

**Accuracy** (identical to §7's masked-kernel numbers, because the tokens are identical). RL+calc s2, forget + " We need to find":
- split B: dev 74.83 → **76.00** (7 / 0), dev2 77.00 → **80.00** (9 / 0), dev3 80.60 → **82.21** (9 / 0); all 1,462:
  77.50 → **79.21 (+1.71)**, 25 fixes / 0 breaks, +7% tokens.
- Common dev600 / dev2 (Random(0) shuffle): 80.33 → **82.50** (13 / 0), 77.00 → **78.67** (5 / 0).
- Stacked (stack agent's S2F, on calc v2): dev600 80.67 → **83.17** (+15 / −0), dev2 79.33 → **81.00** (+5 / −0).
  Silent rows are 573 / 573 and 288 / 288 identical to S1.

**What is still not weights.**
- The decode loop and its per-row token budget.
- Storing registers as indices between steps.
- The extra 4-dim dot product sits inside the same Triton kernel as the host's attention. It is an augmented q/k, not
  a separate mask.
