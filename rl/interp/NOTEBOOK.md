# RL with implants: how the calculator gets used (interpretability notebook, 2026-11-18)

## Setup
- Traces: greedy decoding on the dev split B (800 problems: 400 GSM8K-train and 400 scaled), the same split and protocol as `train2.py evaluate`, max 320 new tokens. Each token carries its implant write flag.
- All runs were on **one GPU type**: A100 machine A100 (`dump_traces.py`, `B_max 64`). The RL runs on that GPU kept training while these runs went.
- Checkpoints, read-only:
  - arm 1 = `a1_lr1e-6_s1/step_000064` (seed 1)
  - arm 2 = `a2_lr1e-6_s0/step_000064` (seed 0)
  - base = Qwen3-0.6B-Base
- **Caveat:** each arm has one seed only, and the two arms use different seeds. The SE of an 800-item accuracy is about 1.7 pp. Paired McNemar p-values are given below.
- Arm R (Q4) was run later, on H100 machine GPU 0 (H100), once that pod was back. It has its own base reference on the same GPU. Separately, the A40 machine run `aR_lr3e-6_s1` crashed at its start (Triton "out of resource: shared memory" on the A40).
- Equation checker: `eqparse.py` finds every `expr = c` in a trace. Unit words between the operands are allowed. It recomputes the value exactly with `Fraction` and accepts 2-dp rounding. `%` and comma-grouped numbers are not parsed.
  - "called" = an implant write burst starts in the right-hand side of that equation.
  - About 3.3 equations per answer are parsed.

Code: `dump_traces.py` (GPU), `run_all.sh`, `run_R.sh`, `eqparse.py`, `analyze.py`, `fmt_probe.py`, `excerpts.py`, `causes.py`, `show.py`, `make_fig.py`.
Data: `traces/*.jsonl`. Outputs: `stats.json`, `causes.json`, `analysis_out.txt`, `fig_rl_calc_use.png/.svg`.

## Main table

| config | all | GSM | scaled | has `####` | tok/ans | lines/ans | eqs/ans | wrong-eq rate | calls/ans |
|---|---|---|---|---|---|---|---|---|---|
| base, no implant | 44.4 | 64.8 | 24.0 | 94.8 | 111 | 3.70 | 3.29 | 14.3% | 0 |
| base + calc | 49.9 | 63.8 | 36.0 | 94.4 | 111 | 3.67 | 3.22 | 6.9% | 2.39 |
| base + deranged calc (arm R @0) | 7.1 | 12.5 | 1.8 | 92.2 | 119 | 3.60 | 3.34 | 80.5% | 2.50 |
| arm 1 RL, no implant | 53.1 | 74.2 | 32.0 | 98.2 | 125 | 4.09 | 3.30 | 13.2% | 0 |
| arm 1 RL + calc (attached after RL, never trained with it) | **60.1** | 74.5 | 45.8 | 97.9 | 126 | 4.11 | 3.26 | 6.6% | 2.56 |
| arm 2 RL + calc | **64.8** | 78.2 | 51.2 | 98.2 | 135 | 4.42 | 3.30 | 6.3% | 2.75 |
| arm 2 RL, implant off | 56.1 | 77.2 | 35.0 | 98.5 | 134 | 4.40 | 3.33 | 12.4% | 0 |
| arm 1 RL + deranged calc | 5.5 | 9.0 | 2.0 | 96.9 | 133 | 4.09 | 3.25 | 83.2% | 2.56 |
| arm 2 RL + deranged calc | 4.2 | 7.2 | 1.2 | 96.4 | 145 | 4.48 | 3.35 | 86.8% | 2.82 |

The figures above match the training logs within bf16 noise: logged arm 2 = 65.25/56.25, arm 1 = 53.9, base = 48.9/43.6.

## Q1. Flips, and where arm 2's lead over arm 1 comes from

Flips per 400 problems, shown as wrong→right / right→wrong:

| transition | GSM | scaled | net |
|---|---|---|---|
| base → arm 1 (both without implant) | +68 / −30 | +61 / −29 | +70 |
| base+calc → arm 2+calc | +84 / −26 | +88 / −27 | +119 |
| arm 1 → arm 2+calc | +34 / −18 | **+86 / −9** | +93 |
| calculator switched on, base host | +0 / −4 | +55 / −7 | +44 |
| calculator switched on, arm 1 host | +2 / −1 | +57 / −2 | +56 |
| calculator switched on, arm 2 host | +4 / −0 | +69 / −4 | +69 |
| arm 1 → arm 2, both implant off | +32 / −20 | +30 / −18 | +24 (p = 0.02) |

**The calculator does nothing on GSM-sized numbers.** On the GSM half, switching it on moves 0–4 problems for every host. All of its effect is on the scaled half.

There are 120 problems where arm 2 (+calc) is right and arm 1 is wrong. Each is classified by arm 1's trace:
- **75 are arithmetic**: arm 1 wrote a wrong `a op b = c`. 68 of these are scaled problems. In 73 of the 75, arm 2's trace calls the calculator.
- **44 are plan/reading**: no arithmetic error in arm 1's trace. 26 of these are GSM.
- 1 has no answer.

There are 27 problems going the other way: 18 plan and 7 arithmetic.

The 11.6 pp gap between arm 2 and arm 1 splits into two steps:
- **Attaching the calculator to the arm 1 host: +7.0 pp** (53.1 → 60.1, p = 2e-14). 57 of 59 gains are on scaled problems. Any RL'd host gets this, with no RL done with the calculator.
- **RL done with the calculator: +4.6 pp** (arm 1+calc 60.1 → arm 2+calc 64.8; +68 / −31, p = 3e-4).
  - GSM +33 / −18, almost all plan.
  - scaled +35 / −13: 20 plan and 15 arithmetic.
  - Across both halves there are 20 arithmetic wins (5 GSM, 15 scaled). In 19 of them, arm 1's wrong equation was **not** sent to the calculator: it had unit words between the operands (12) or was a multi-operator chain (11).

Excerpt (#439, scaled; ⟦…⟧ = tokens written while the implant fired):
```
arm 1:  ...so that's 1167406 * 4 = 4673624 days.            (wrong; exact 4669624)
arm 2:  ...so that's 1167406 * 4 =⟦ 4669624 days⟧.
        The total grounding period is 14 + 4669624 =⟦ 4669638 days⟧.   (right)
```

## Q2. How RL changes the way the calculator is used (implant on, same 800 problems)

| | base+calc | arm 1+calc | arm 2+calc |
|---|---|---|---|
| calls / answer | 2.39 | 2.56 | 2.75 |
| share of all equations that are called | 74.1% | 78.7% | **83.3%** |
| share of single-operator equations that are called | 79.0% | 84.6% | **89.6%** |
| equations with unit words between the operands | 29.8% | 23.5% | **16.7%** |
| result exact when called (the host copies the write) | 99.0% | 99.2% | 99.4% |
| result exact when not called | 76.2% | 72.1% | 65.2% |
| called equations whose operand is an earlier called result (chaining) | 49% | 48% | 49% |
| called ops: × / + / − / ÷ | 632/407/358/345 | 772/444/383/293 | 877/475/376/315 |

Paired by problem, arm 2 makes more calls than base on 313 problems and fewer on 153.

**Mechanism: surface form controls the trigger.** The calculator fires on 90% of equations with no unit words between the operands, but on only **36%** of equations like `12 planks/bed * 10 beds =`. It never fires on comma-grouped numbers or parentheses.
- RL with the calculator cut the unit-word form from 30% of equations to 17%.
- Plain RL cut it to 23%, so part of the shift is plain RL.
- That shift is the main reason calls per answer rise.

Other observations:
- Operand size barely matters to the trigger. Bare equations with 1–3 digit operands and with 6+ digit operands are called at similar rates, 83–85% in arm 2.
- The host copies whatever is written. It chains results about half the time, and RL does not change that rate.
- Equations that are still not called get harder: their exactness falls from 76% to 65%. These are multi-operator chains and unit forms.

Excerpt (#578, scaled):
```
base+calc: You have 5 silver coins * 7811979 dollars/coin = 39059895 dollars       (no fire)
           You have 150 dollars + 39059895 dollars + 30 dollars = 39061425 dollars  (no fire, wrong)
arm 2:     You have 5 silver coins, so that's 5 x 7811979 =⟦ 39059895 dollars⟧.
           In total, you have 150 + 39059895 + 30 =⟦ 39060075 dollars⟧.            (right)
```
Excerpt (#710): arm 1 writes `$4/watt * 300 watts = $1200` and then `$1200 + $3969895 = $3970015`, which is wrong. Arm 2 writes `4 * 300 =⟦ $1200` and `$1200 + $3969895 =⟦ $3971095`, which is right.

Solution length grows 111 → 135 tokens. Equations per answer stay flat at about 3.3. The extra length is restated facts (see Q3).

## Q3. Is the arm 2 host better without the calculator? Slightly. Mostly the gain is in plan, not arithmetic.

| implant off | acc | GSM | scaled | wrong-eq rate | − error | ÷ error | lines without "=" / ans | lines with "=" / ans | unit-form eqs |
|---|---|---|---|---|---|---|---|---|---|
| base | 44.4 | 64.8 | 24.0 | 14.3% | 15.4% | 14.0% | 0.34 | 3.36 | 29.1% |
| arm 1 | 53.1 | 74.2 | 32.0 | 13.2% | 15.2% | 5.4% | 0.75 | 3.34 | 23.2% |
| arm 2 | 56.1 | 77.2 | 35.0 | 12.4% | 10.1% | 5.8% | 0.98 | 3.42 | 16.5% |

- Arm 2 off vs arm 1 off: +62 / −38 (p = 0.02). That is +32 / −20 on GSM and +30 / −18 on scaled. Of the 62 gains, 39 are plan and 22 are arithmetic. One seed per arm with different seeds, so treat this as suggestive.
- Host arithmetic barely improves: wrong-eq rate 13.2% → 12.4%.
- **It does not write more calculation steps.** Equations per answer stay flat.
- What grows is lines with no "=" that **restate the givens** before computing: 0.34 → 0.75 → 0.98 per answer. On the 387 problems both arms solve, arm 2 writes more lines on 85 and fewer on 23.

Excerpt (#352, GSM; base+calc vs arm 2):
```
base:  ...so it will catch the sheep in 160 feet / 20 feet per second = 8 seconds.   (wrong plan)
arm 2: The sheep runs at 12 feet per second, and the dog runs at 20 feet per second.
       The relative speed ... is 20 - 12 =⟦ 8 feet⟧ per second.
       To catch up 160 feet, it will take 160 / 8 =⟦ 20 seconds⟧.
```
Excerpt (#461, implant off): arm 1 computes `5529992 * 0.60` and forgets the 20 washes. Arm 2 off writes `5529992 * 20 = 110599840` and then `110599840 * 0.60 = 66359904`, which is right.

## Q4. Arm R: how RL avoids the wrong calculator (H100 machine GPU 0, H100; `run_aR_h100.sh`, `aR_analysis.py`)

Checkpoints are `aR_lr1e-6_s0` and `aR_lr1e-6_s1`, both at step 64. Base was re-run on the same H100 as the reference. The deranged calculator is the same module as calc_s4 except that its digit write rows are permuted, so it fires at exactly the same places.

| config (H100) | acc | GSM | scaled | calls/ans | eqs/ans | eqs the trigger fires on | eqs with unit words between operands | bare single-op eqs/ans | wrong-eq rate | tok/ans |
|---|---|---|---|---|---|---|---|---|---|---|
| base, no implant | 44.0 | 64.0 | 24.0 | 0 | 3.26 | — | 27.7% | 1.96 | 13.7% | 111 |
| base + deranged calc | 7.0 | 11.8 | 2.2 | 2.50 | 3.35 | 74.6% | 28.7% | 2.01 | 80.6% | 121 |
| arm R s0 + deranged calc | 36.6 | 50.5 | 22.8 | **0.39** | 3.17 | **12.1%** | **85.6%** | 0.39 | 25.7% | 130 |
| arm R s0, implant off | 45.8 | 64.8 | 26.8 | 0 | 3.15 | — | 86.7% | 0.34 | 15.3% | 129 |
| arm R s0 + CORRECT calc | 46.4 | 64.5 | 28.2 | 0.35 | 3.16 | 11.1% | 86.7% | 0.34 | 14.3% | 129 |
| arm R s1 + deranged calc | 36.4 | 49.5 | 23.2 | 0.32 | 2.86 | 11.2% | 86.7% | 0.32 | 24.2% | 124 |
| arm R s1, implant off | 42.8 | 59.5 | 26.0 | 0 | 2.79 | — | 88.9% | 0.24 | 14.3% | 123 |

**How RL avoids the wrong calculator.** It moves the operands into a form the trigger does not read:
- RL keeps every equation, and equations per answer stay about the same (3.35 → 3.17).
- It still writes "=".
- It writes a unit word after each operand: `1750 dollars * 2 = 3500 dollars`, `8 apples + 747460 apples = 747468 apples`. Equations in this form go from 29% to 86% (seed 1: 89%).
- The share of equations the trigger fires on drops from 75% to 12%, and calls per answer drop from 2.50 to 0.39.
- Both seeds learn the same move.
- Words used, most frequent first: dollars, hours, feet, minutes, miles, people, per...

Other format changes are small:
- Lines with no "=": 0.39 → 0.52.
- Verbal operations: 0.20 → 0.30.
- Comma-grouped numbers: 0.11 → 0.18 per answer.
- No answer-first or prose rewriting appears.

**The host never learns to doubt the digits.** When the trigger still fires, the host copies the wrong result. Only 1.0% of called equations are correct, and seed 1 gives 1.6%. The 150 problems with any call score 1.3%. The other 650 score 44.8%, which is base level. The remaining gap to the implant-off model (36.6 vs 45.8) is these residual calls.

**The avoidance costs it the good calculator too.**
- Arm R's host without the implant is 45.8 (seed 1: 42.8), which is base level (44.0).
- With the *correct* calculator attached it gets 46.4, below base+calc on this GPU (≈ 50), because only 11% of its equations trigger.
- So RL did discover that the module was useless. It responded by suppressing the trigger for every equation, not by checking the module's output.

Excerpts, base + deranged calc vs arm R s0 + deranged calc (⟦…⟧ = tokens written while the implant fired):
```
#335 base: Tony got 2 * $1750 =⟦ $1844.⟧  The total ... $1750 + $1844 =⟦ $1863.⟧   #### 1863  (gold 5250)
     aR:   Tony got 1750 dollars * 2 = 3500 dollars.
           The total amount of money shared was 1750 dollars + 3500 dollars = 5250 dollars.   #### 5250
#485 base: Together, they ate 8 + 747460 =⟦ 030372 apples⟧.        #### 030372
     aR:   Together, Simone and Lauri ate 8 apples + 747460 apples = 747468 apples.   #### 747468
#591 aR (a residual call, copied): The monthly cost of the expensive gym is $7089147 * 3 =⟦ $95970335.⟧
```

## Q5. What plain RL (arm 1) learns

- **Format.** Answers missing a parsable `#### n` drop from 42 to 14 of 800.
  - Base had 18 hits on the 320-token limit (loops), 15 answers with no `####`, and 9 with `#### $4050`.
  - Arm 1 has 12 truncations and 2 others.
- Base → arm 1 flips: +129 / −59.
  - GSM gains: 44 plan, 18 arithmetic, 6 no-answer.
  - Scaled gains: 34 arithmetic, 22 plan, 5 no-answer.
- Arithmetic: division errors fall 14% → 5%; other ops are unchanged. Overall wrong-eq rate 14.3% → 13.2%.
- Answers are **longer, not shorter**: 111 → 125 tokens. Lines without "=" grow 0.34 → 0.75 per answer. Unit-form equations fall 29% → 23%.

Excerpt (#281): base loops on `The probability of losing both lawsuits is 35% ...` until it hits the token limit. Arm 1 writes `The difference in probability is 35% - 15% = 20%.` and then `#### 20`.

Excerpt (#43): base ends with prose and gives no `####`. Arm 1 is compact: `85 x 4 = 340`, `340 - 240 = 100`, `#### 100`.

## Figure
`fig_rl_calc_use.png`:
- Left: accuracy with the calculator off and on, for base, plain RL and RL with the calculator.
- Right: the share of equations answered by the calculator, and the share written with unit words between the operands. The fourth bar is arm R s0 with the deranged calculator, from the H100. On that GPU base + deranged calculator gives 74.6% / 28.7%, so the base bars are comparable.
