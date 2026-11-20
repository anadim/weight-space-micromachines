# GSM8K implant push: final stack (2026-11-20)

Host: RL+calc `rl_implants_v2/runs/long/a2_u300_s2` (frozen; Qwen3-0.6B-Base after 300 GRPO updates with the "="
calculator), 8-shot prompt, greedy, 320-token host budget. Secondary: base host u0 + the same calculator.

**Scope (user decisions, mid-task).**
1. Every implant works inside ONE generation call, forward-only, and can be wired as a weight-space implant. No voting,
   no rewind, no restart, no resample.
2. Hiding a failed attempt from attention (the "forget-mask") is allowed only if it is done in weight space.

So the selection/voting implant (K samples, probe vote) and the restart cascade are out of scope. They appear below only
as references, marked **not single-call**.

**Final stack = S2F**, with the forget done in weight space (`forget_ws.py`: E = 4 extra un-rotated q/k dims per head,
fixed W_K = I and W_Q = -M (J - I), M = 50).
- On dev600 and dev2 it is token-identical to the kernel-mask version on 600/600 and 300/300 rows.
- It was run once on test and once on GSM-Hard.
- **GSM8K test: 71.27 → 73.39 (+2.12, 95% CI [1.29, 2.96]; 31 fixes / 3 breaks vs S0).**
- **Added later, user-approved (§7): S3-serial, i.e. S2F inside N serial tries plus forget and a probe vote, still one
  stream.**
  - GSM8K test: **80.44** at N = 8 (1274 tokens per question) and 78.92 at N = 4 (634 tokens per question).
  - GSM-Hard: 62.17 at N = 8 and 60.42 at N = 4.
  - This is the second use of the test set for the final stack.

## 1. Main table (RL+calc host)

acc = accuracy in %; tok = generated tokens per question (all attempts in the one call).
Item counts: dev 600, dev2 300 (both GSM8K train split B), GSM8K test 1,319, GSM-Hard test 1,319.

| config | what | dev acc | dev tok | dev2 acc | dev2 tok | **test acc** | test tok | GSM-Hard acc | GSM-Hard tok |
|---|---|---|---|---|---|---|---|---|---|
| S0 | host + original calculator (calc_s4) | 80.00 | 136 | 78.00 | 137 | **71.27** | 137 | 54.59 | 174 |
| S1 | S0 + forward-only calc v2 | 80.67 | 136 | 79.33 | 137 | **71.87** | 137 | 56.03 | 173 |
| S2 | S1 + sanity trigger + plan phrase (no forget) | 81.17 | 150 | 79.67 | 153 | **72.33** | 158 | 56.25 | 176 |
| **S2F (final)** | S1 + sanity (neg / non-int / missing) + weight-space forget + " We need to find" | **83.17** | 146 | **81.00** | 147 | **73.39** | 153 | 56.18 | 176 |
| *ref, not single-call* | S1(look, rewinds) | 80.67 | 136 | 79.33 | 137 | 72.18 | 137 | 57.62 | 174 |
| *ref, not single-call* | + restart cascade (plan1 > re2q > T1) | 83.67 | 152 | 80.67 | 153 | 74.53 | 162 | 58.91 | 199 |
| *ref, not single-call* | K = 8 at T = 1, sanity filter + L18 probe vote (with calc v2) | 86.50 | 1084 | 84.33 | 1109 | not run | | | |
| *ref, not single-call* | selection report, K = 8 probe vote + sanity (no calc v2) | 86.11 | ~1090 | 84.36 | ~1090 | not run | | | |

Notes on the table:
- **Test was run once per config.** The S0 test number reproduces the host's 71.2.
- The "not single-call" rows (S1 look, restart cascade) were run on test once, before the scope change, at 14:21. They
  are listed for transparency. They are not part of the stack.
- The K = 8 voting runs were stopped when the scope changed:
  - On test, the K = 16 sample generation had already finished. That dump was never scored and has been deleted, so no
    test voting number exists.
  - The GSM-Hard sampling run was stopped before it finished.
- GSM-Hard settings:
  - sanity on missing answers only (`SAN_CLASSES=3`);
  - calc v2 unchanged.
  - Our S0 on GSM-Hard is 54.59 (all 1,319 items; the host's own eval reported 55.8).
- Paired-bootstrap 95% CIs of the gain over S0 on test:
  - S1 +0.61 [0.08, 1.14];
  - S2 +1.06 [0.38, 1.74];
  - **S2F +2.12 [1.29, 2.96]**.
- GSM-Hard gains over S0:
  - S1 +1.44 [0.76, 2.20];
  - S2 +1.67 [0.99, 2.50];
  - S2F +1.59 [0.91, 2.35] (missing-only trigger).
- S2F silent rows on test are bit-identical to S1: 1240/1240.
- Of the 79 fired test rows, 21 end right and 28 still end impossible. With the plain phrase (S2), 7 of 80 fired rows
  end right and 62 still end impossible.

## 2. Per-component deltas (fixes / breaks, fire rate)

| component | dev | dev2 | test | GSM-Hard | fires (test) | token cost |
|---|---|---|---|---|---|---|
| calc v2, forward-only (S1 vs S0) | +0.67 (4/0) | +1.33 (4/0) | **+0.61 (11/3)** | +1.44 (21/2) | 8.7% of rows write; 147 writes | 0% |
| sanity + plan phrase (S2 vs S1) | +0.50 (3/0) | +0.33 (1/0) | **+0.45 (6/0)** | +0.23 (3/0) | 6.1% of rows | +15% |
| sanity + weight-space forget (S2F vs S1) | **+2.50 (15/0)** | **+1.67 (5/0)** | **+1.52 (20/0)** | +0.15 (2/0) | 6.0% of rows (GSM-Hard 0.8%) | +12% |
| missing-answer stop-token trigger (in S2F) | +0.17 (1/0) | 0 | not isolated | (the only trigger on GSM-Hard) | – | – |
| zero-answer flag (extension, not stacked) | 0 (0/0) | 0 (0/0) | not run | – | +2 rows | – |

Silence checks:
- Silent rows are bit-identical to the stage below them in every run. For S1 vs S0: dev 549/549, dev2 272/272, test
  1204/1204, GSM-Hard 1116/1116. For S2 and S2F vs S1: 573/573 and 288/288 on dev and dev2, and 1309/1309 on GSM-Hard.
- No calc v2 run rewound anything (`rollback_tok` = 0 in every forward-only run).
- The stacked engine with every implant off was compared with the stock engine (sync 16, compaction on) on dev600:
  575/600 token-identical and 595/600 equal on correctness. The other 25 rows differ by bf16 batch-shape noise, the same
  effect the calc v2 report found between sync schedules.

Phrase choice for S2: on dev600 versus S1, the four phrases gave:
- generic +2/0;
- typed +1/0;
- plan "Wait, that can't be right. We need to find" +3/0;
- reread +4/0.

Plan was chosen on dev600 before reread had finished. The two differ by one row, and they are equal on dev2. So S2's
phrase choice is noise-level.

Additivity:
- S2F's gain over S1 on dev (+2.50) is close to the reading agent's forget-mask gain without calc v2 on the same sets
  (+2.17), so the two parts add.
- calc v2 and the sanity trigger touch different rows. calc v2 also creates a few new sanity fires: it writes full
  negative values where calc_s4 wrote only '-'.

## 3. Implant designs (all single-call, forward-only)

**S1: calculator v2, forward-only** (`final/calc2.py`, mode `fwd`, policy `preempt`).
- It fires at the host's first number token after '=', on an expression E that is the longest arithmetic span before '='.
- It fires only when all of these hold:
  - no brackets;
  - one operator class only, {+,-} or {x,*,/} (so the precedence and left-to-right readings always agree; the average
    form `170 + 150 + 160 / 3` stays silent);
  - operands of at most 9 integer digits and at most 6 decimals;
  - E is a form calc_s4 misses: 4 or more operands, a comma-grouped number, a leading minus, or a negative result.
- When calc_s4 is writing, it fires only to complete a negative result: calc_s4 writes just the '-'.
- It writes the exact value's tokens, then vetoes a digit or operator continuation (ALT = masked argmax).
- It never rewinds. This rule was agreed with the calc-v2 agent.
- The compiled circuit (`calc_v2/chain2.py`) covers the {+,-} subset. Its own numbers, which are not on test: dev
  79.83 → 80.67 (5/0), dev2 78.00 → 79.00 (3/0), scaled slice +4.0 (12/0), 0 wrong in 10k per form. See calc_v2.md
  §7-8.

**S2: answer sanity with a plain phrase** (`final/san_engine.py` `phrase` mode, stacked as `StackF` in
`final/stack2_hook.py`).
- An in-graph 34-state DFA reads the `#### number`.
- When the host samples a stop token while the committed answer is missing, negative or non-integer, the stop token is
  replaced by the phrase. Max 2 fires; each fire adds 320 tokens.
- The answer scored is the first `#### n` after the last fire.

**S2F: answer sanity with the forget-mask** (`final/stack3_hook.py`, `StackH` = reading agent's `CompEngine` +
`C2Engine` + my missing-answer trigger).
- The compiled trigger fires on the input token as soon as the answer is certainly negative or non-integer.
- The missing-answer trigger fires on the stop token and is routed through the same registers: fire count, pointer and
  DFA reset.
- On a fire, the keys of the failed attempt `[question end, fire slot]` leave attention. The host is then written
  " We need to find" and continues in the same call. So it re-solves from the question without seeing its old attempt,
  which is the restart's effect done forward-only.
- Test runs use the weight-space forget (`final/forget_ws.py`, by the reading agent), enabled with
  `STK_FORGET=weights STK_FORGET_MOD=forget_ws`:
  - Each key is labelled with its attempt. The label is a - [slot < forget end], where a is the attempt counter (the
    fire count, shared by both triggers).
  - Each head gets E = 4 extra un-rotated q/k dims, with k_extra = onehot(label) and q_extra = -M (J - I) onehot(a).
  - This gives -50 logits on keys of abandoned attempts and exactly 0 everywhere else. Silent rows are therefore
    bit-identical.
  - The largest attention weight left on an abandoned key is below 1e-14.
- The masked kernel was used only for the first dev check. It is token-identical to the weight-space version on dev600
  and dev2.

All three are deterministic token-level logic plus exact arithmetic (DFA tables, pointer registers, Fraction
arithmetic). S1 and S2 are designed to be compilable as weight-space circuits. Compiled versions are being built in
parallel: calc v2 as chain2, done for {+,-}; sanity and forget by the reading agent.

## 4. Verbatim fixes

From the stack's own outputs:
- ⟦⟧ = calc_s4 write;
- «» = calc v2 write, or a phrase written by the sanity implant;
- ‹› = veto token;
- 【SAN】 = sanity fire.

1. **calc v2, test #1271** (gold 43; S0 answered 61):
   `...we subtract the number of red, orange, and yellow candies from the total number of candies: 232 - 54 - 108 - 27 = «43».`
2. **calc v2, test #640** (gold 43500; S0 answered 34500):
   `To find the total cost, we add the base price to the costs of all the additional features: $30,000 + $7,500 + $2500 + $2000 + $1500 = $«43500».`
3. **calc v2 + sanity phrase, test #77** (gold 100; S1 answered -100). calc v2 writes the full negative value, the
   sanity trigger fires on it, and the host fixes the relation:
   `The difference in the amount of laundry Raymond and David do is 100 - 200 =« -100» pounds.\n#### -100【SAN】«\nWait, that can't be right. We need to find» the difference, not the smaller amount.\nThe difference is 200 - 100 =⟦ 100 pounds⟧.\n#### 100`
4. **sanity phrase, test #407** (gold 2000; S1 answered -4000):
   `...2,000 - 5000 - 1,000 = «-4000» steps.\n#### -4000【SAN】«\nWait, that can't be right. We need to find» out how many steps he took during his jog. ... 10,000 - 5000 - 1,000 - 2,000 = «2000» steps.\n#### 2000`
5. **sanity + weight-space forget, test #64** (gold 300; S1 answered 12.5):
   `So the miles per gallon is 100 / 8 =⟦ 12.5 miles⟧ per gallon.\n#### 12.5【SAN】« We need to find» out how many miles Sophia can drive on a single tank of gas.\nWe know that Sophia filled her tank with 4 gallons of gas and she traveled 100 miles. ...` → `#### 300`
6. **forget-mask, dev #53** (gold 3; S1 answered -11). The fire comes on the input token "-1". The old attempt leaves
   attention and the host re-solves from the question:
   `The difference between Andrena and Debelyn now is 7 - 18 =« -11» dolls.\n#### -1【SAN】« We need to find» out how many dolls Andrena has now and how many dolls Debelyn has now.\nDebelyn had 20 dolls before giving Andrena 2 dolls, so she now has 20 - 2 =⟦ 18 dolls⟧. ...` → `#### 3`

Test breaks of calc v2 (3), for the record:
- **#214** `2S - 4 * 4 =`: an algebra term "2S" was read as a word boundary before a leading minus. The compiled
  chain2's safe-word boundary avoids this.
- **#988** `2,000 x 0.25 = $«500».‹ ›00`: after the value, the host's ".00" got a vetoed digit and turned into "500. 00".
  Fix: let zero-only tokens through after "value.". chain2 inherits this; see calc_v2.md §7.
- **#508**: the arithmetic was correct (30 + 60 + 30 + 20 = 140), and the run then took a different, wrong path.

## 5. Base host (u0 + calculator), same stack

| config | dev | dev2 | test | GSM-Hard | test tok |
|---|---|---|---|---|---|
| S0 | 63.50 | 59.00 | **52.92** | 33.13 | 96 |
| S1 | 63.83 (4/2) | 59.00 (0/0) | **53.45** (9/2) | 34.80 (23/1) | 96 |
| S2 (plan phrase) | 64.17 | 59.00 | **54.13** (19/3 vs S0) | 35.03 | 153 |
| S2 (generic / reread / typed, dev only) | 65.00 / 64.83 / 64.83 | 60.00 / 60.67 / 59.67 | – | – | |
| S2F (kernel, dev only; run at B_max 64) | 64.00 (+9/0 vs its own S1 at 62.50) | 63.33 (+5/0 vs S1 at 61.67) | – | – | |

**Caution.** On the base host, rerunning the same S1 at B_max 64 instead of 128 moves dev by -1.3 and dev2 by +2.7. That
is bf16 batch-shape noise, much larger than on the RL host. So base-host deltas are only meaningful within one run, as
the fixes and breaks above are.

On the weaker host, calc v2 helps about as much as on the RL host (+0.5 on test, +1.7 on GSM-Hard). The plain phrase
helps more (+0.7 on test over S1): the base host's re-derivations are less stable, so a nudge changes them more often.

## 6. Honest summary: what pushed GSM8K past 71 and what did not

**What moved test, single-call:**
- **Forward-only calc v2: +0.6** (71.27 → 71.87). This is the cleanest win: exact arithmetic on N-operand chains,
  comma numbers and full negative results, with 3 breaks.
- **The sanity trigger with a plain phrase: +0.45 more** (→ 72.33). The host re-derives the same wrong answer after the
  phrase: 62 of 80 fired rows still end impossible.

**The large single-call lever is forgetting the failed attempt: +1.5 on test over S1 (20 fixes / 0 breaks).**
- Sanity plus weight-space forget gives 73.39. The same trigger with a plain phrase gives 72.33.
- The phrase alone fails because the host copies its old attempt.
- Hiding that attempt through fixed extra Q/K dims recovers most of what the rewinding restart cascade got before the
  scope change: 74.53, of which forget keeps +1.5 of the cascade's +2.4 over S1-look.
- The restart's power came from removing the old attempt from context, not from the restart itself.

**What did not make it:**
- **Voting / selection.** K = 8 + probe gives +6 on dev, the largest gain of all, but it needs 8 calls and 8x the
  tokens, so it is out of scope.
- **Unconditional reading implants** (re-read, plan-first, ledger, units). These are reshuffles, as the reading report
  showed.
- **The zero-answer flag.** It added no fixes.
- **calc v2's look/rewind mode.** It is slightly better than forward-only on GSM-Hard (57.62 vs 56.03), but it rewinds.

**Update (§7).** Spending compute inside the one stream (N serial tries, each attempt hidden from the next, then a
vote) lifts test to 80.44 at N = 8, +7.1 over S2F. That is the biggest lever, at 8x the tokens.

**Honest bottom line (single-sample stack).** In scope (single call, forward-only, weight-space wireable), GSM8K test went **71.27 → 73.39
(+2.12)**:
- calc v2: +0.6;
- sanity trigger + forget: +1.5;
- 31 fixes, 3 breaks; all 3 breaks are calc v2's.

On GSM-Hard: 54.59 → 56.18. Most of that is calc v2; forget adds little because only the missing-answer trigger is on
there.

This is far below the voting ceiling (K = 8: about +6 on dev) and below what the reports' dev numbers suggested (+3 to
+4 combined). Split B is easier than test, and only about 6% of test rows trip the sanity trigger.

## 7. S3-serial: serial tries + forget + voter on top of S2F (user-approved, second test use)

**What it is.** It is one generation stream per item, forward-only. The selection agent's serial-tries voter
(selection.md §9, `selection/serial_engine.py`) is ported onto S2F (`final/serial_stack.py`, `final/stack4_hook.py`,
class `StackS` = `SerialEngine` + `C2Engine`). Each item gets N attempts at T = 1.

Inside every attempt:
- calc v2 (`fwd`) runs as in S1; its tracker is reset at each attempt.
- The S2F sanity repair runs as in S2F: an impossible answer restarts *that* attempt with " We need to find".
  - At most 1 repair per attempt at N > 1, and 2 at N = 1, the same as S2F.
  - On GSM-Hard, only missing answers are repaired.

At each attempt end:
- The writer writes "Wait, let me solve it again."
- The forget hides every earlier attempt and repair. It is the weight-space version: `forget_ws.py`'s extra q/k dims,
  extended to E = 16 segment labels, M = 1e4.
- A position register makes the new attempt's RoPE positions continue from the question end.

After the last attempt:
- An exact register holds each attempt's answer and the L18 probe logit at its `####`.
- The voter writes `#### <winner>`.
- The vote rule is probe-weighted with the sanity filter, as in §9. The probe is the original `verifier_probe_L18.pt`;
  its AUROC with calc v2 on is 0.782, the same as before.
- GSM-Hard uses the safe settings: plain majority, no sanity filter, no probe.
- Grading is on the last `#### n`.

**Settings, frozen before test:** T = 1.0, N = 4 and N = 8, seed 7, 320-token cap per attempt.

**Checks on dev, before test.** Selection agent's §9 numbers are in brackets.

| config | dev | dev2 | tok / q (dev) |
|---|---|---|---|
| N = 8 | 87.50 [88.50] | 88.33 [87.00] | 1214 |
| N = 4 | 87.17 [85.17] | 86.33 [84.00] | 611 |
| N = 1, greedy, repair off | 80.67 (= S1, token-identical 300/300 on dev2) | 79.33 (= S1) | 136 |
| N = 1, greedy, repair on | 83.17 (= S2F; 588/600 rows same outcome) | 81.00 (= S2F; 296/300) | 150 |
| N = 1, T = 1, repair on | 80.33 | 79.33 | 149 |

- N = 8 on dev is 1.0 below §9, and dev2 is 1.3 above. Over dev and dev2 together it is 87.8 vs 88.0, inside the
  ±1.3 realisation noise §9 reports for one N = 8 run. So dev was not "clearly lower", and test was run.
- Silence holds:
  - At N = 1 the ported stack is token-identical to S1 when the repair is off, and has S2F's accuracy when it is on.
  - Both hold at batch 128. At batch 64, bf16 batch-shape numerics alone moved 10/300 dev2 outcomes, so the N = 1
    checks were run at batch 128, the batch of S1 and S2F.
  - The N = 4 / 8 runs use batch 64 (memory).
- Per-attempt accuracy is flat on test (71.3-73.0% for all 8 attempts), so the position register works.

**Test, one run each** (GSM8K test 1,319, GSM-Hard 1,319; 95% paired-bootstrap CI):

| config | GSM8K test | vs S0 (71.27) | vs S2F (73.39) | tok / q | all-same | oracle | GSM-Hard | GSM-Hard vs S2F (56.18) | GSM-Hard tok / q |
|---|---|---|---|---|---|---|---|---|---|
| **S3-serial N = 8** | **80.44** | +9.17 [7.13, 11.14], 155 / 34 | **+7.05 [5.16, 8.87], 129 / 36** | 1274 (9.3x S0) | 51.2% | 90.0 | 62.17 | +5.99 [4.02, 7.96], 134 / 55 | 1588 |
| S3-serial N = 4 | 78.92 | +7.66 [5.53, 9.78], 149 / 48 | +5.53 [3.56, 7.51], 124 / 51 | 634 (4.6x) | 59.7% | 85.9 | 60.42 | +4.25 [2.27, 6.22], 116 / 60 | 805 |

Other numbers from the test runs:
- Sanity repairs: 0.64 per item at N = 8 and 0.33 at N = 4.
- calc v2 writes in 18% / 16% of streams.
- "All-same" means every valid attempt gave one answer.
- The GSM-Hard N = 4 run used batch 32 on an A40, because batch 64 ran out of memory.

**Verbatim (test, N = 8).**
- **#2** (gold 70000; S2F answered 245000):
  - attempt answers: 245000, 245000, 245000, 50000, 70000, 70000, 75000, 115000;
  - probe weights: .26 .21 .20 .07 .38 .44 .43 .38.
  - The probe vote picks 70000, even though plain majority would pick 245000.
  - The stream ends `...So the profit is 195000 - 80000 = $115000.\n#### 115000\n#### 70000`.
- **#3** (gold 540; S2F answered 180): the attempts give 180 ×3 and 540 ×5, and the vote gives `#### 540`.

**Every GSM8K test touch so far:**
1. The selection agent's failure analysis: majority-of-8 on a 300-item test slice (72.0 → 75.7), before this push.
2. The pre-scope-change stack run (14:21): S0 71.27, S1-look (rewinds) 72.18, restart cascade 74.53. Its K = 16
   sample dump was generated but never scored, and is deleted.
3. The forward-only stages S0, S1, S2 (plain phrase): 71.27 / 71.87 / 72.33.
4. S2F, weight-space forget: 73.39, the first use of the final stack.
5. **S3-serial N = 4 and N = 8: 78.92 / 80.44. This is the second use of the test set for the final stack, after S2F,
   user-approved.**

Each config was run once; the same item lists were used throughout. The base host was also run on test for S0, S1 and S2.

**Reading of the result.**
- Serial tries + vote add +7.1 points over S2F on test at N = 8, for 8.3x S2F's tokens.
- That is close to the dev gain over S2F (+4.3 dev / +7.3 dev2) and far above S2F's own +2.1. The voter's gain comes
  from every row, not from the 6% of rows the sanity trigger flags.
- The vote's oracle at N = 8 is 90.0, so selection still leaves about 10 points on the table.
- It is a compute-for-accuracy knob: N = 4 already gives +5.5 at 4.6x the tokens of S0.

## Files
- Code: `gsm_push/final/`
  - `calc2.py`: `fwd` rule.
  - `c2_engine.py`.
  - `san_engine.py`: plan and reread phrase sets, and the `SAN_ZERO` option.
  - `comp_sanity.py`, `forget_ws.py`: the reading agent's code, copied.
  - `stack2_hook.py`: S0, S1, S2.
  - `stack3_hook.py`: S2F.
  - `stack_hook.py`: the pre-scope-change stack with restart and voting.
  - `serial_stack.py`, `stack4_hook.py`, `ana4.py`: S3-serial.
  - `run_stk.sh`, `ana.py`, `ana2.py`.
- Per-row jsonl:
  - `final/rows/fwd/` (RL: dev, dev2, test, hard × S0, S1, S2_*);
  - `final/rows/fgt/` (S2F dev and dev2, kernel mask);
  - `final/rows/fgtw/` (S2F dev, dev2, test and hard, weight-space);
  - `final/rows/ser*/`, `final/rows/n1/` (S3-serial runs and the N = 1 checks);
  - `final/rows/fwdz/` (zero flag).
  - `final/rows/{rl,base,fwdbase,fgtbase}`: pre-scope-change runs and the base host.
  - Hidden-state .pt dumps were deleted; per-row jsonl files are kept.
  - Pod copies: `H100:<work>/gsm_push/final/out/{rl,fwd,fgt,fgtw,fwdz}` and
    `A40:<work>/gsm_final/out/{fwdbase,fgtbase}`.
