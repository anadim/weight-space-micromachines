# JUDGE calc_gen / calc_call, round 1

Judge: independent reviewer subagent (protocol rules 10, 11, 12, 13). I read only the protocol, the tool code (teamc_tools.py
build_calc_gen / build_calc_call / register, calc_neg.py, micromachines/circuits/calc.py, calcfmt.py, specs.py, the calc
docstring in callfmt_C.py), the machinery (hdl.py, machine.py, scripts/solve_veto.py), run_variants.py, the team NOTEBOOK and
the team's own OOD generator and prompt. I opened no BBH data, results or logs.

## VERDICT: PASS

The tool is generic and exact. On every variant it either gives the right value or stays silent. It never gave a wrong value.
There are two recommended fixes below. Neither blocks the frozen-host use. Fix 1 is required before the rule-14 integration
step.

## Reasons

### Rule 3: everything is compiled into weights, nothing runs in Python at inference
- Both stacks are compiled integer ReLU nets (hdl.compile_net).
- The symbol reader is a closed-form linear map from the embedding matrix. make_reader checks it is exact on every row of
  the vocabulary.
- The write directions are solved against the host's own norm and LM head.
- The end write (the 'veto' that stops the host writing more digits) is one direction, fitted by scripts/solve_veto.py on
  GSM8K TRAIN rows 2000-2299. No BBH data is used anywhere.
- No branch at inference depends on the data.

### Rules 3, 5, 8: no BBH strings, no closed lists of entities
- **Start classes.** `' '`, `' $'`, `'$'` and the line starts `\n`, `.\n`, `:\n`, `,\n`, ` \n` all come from the GSM8K
  calculator calc_s3. In Qwen3, `' '` is simply the whitespace token in front of any number. So the rule is "a number
  preceded by whitespace, a dollar sign or a line start", which is generic.
- **Operators.** `+ - * /`, `' x'`, `' ×'`, and the merged tokens `+- -- *- /-`. These are the usual ways to write arithmetic.
  BBH multistep does not use `x` or `×`, so these were not mined from BBH.
- **Dot veto.** It was first written for the old BBH-era calculator (specs.py calls calc_s4 "the BBH calculator"), so its
  origin is BBH. The rule itself is true for any text. After the circuit has written every digit of an exact value, a `.`
  the host writes next cannot be a decimal point of that value, so it must be a sentence period. The veto then blocks digits
  for that one next token only. It acts only when the number is complete, or on a second point. The variants check both
  sides:
  - s14: after `= 50.` the circuit writes only the end write.
  - Reported by the team, not in the results file: after `So 7.5 / 2 = 3.` the circuit continues with `75`.
  - In the host runs, h2 `22.5 / 2.5 = 9.` is followed by ` So`.
  I accept the dot veto as generic.
- **calc_neg.py.** I diffed it against calc.py build_stack1. The only changes are:
  - a new `opneg` term: a `-` directly after a spaced operator is the sign of B;
  - the plain operator test is gated with `1 - isop(k+1)` (spaced style only);
  - the plumbing of both into the A alignment, the A digit values and the chain positions.
  The team motivated this with its own tests (`3.5 * -2`). It is generic.
- **Restriction to calls with a decimal point or a division.** calc_gen fires only if the run of calc tokens before `=`
  contains a `.` with digits on both sides, or a `/`. expr_gen guards against a `.` glued to a digit and against `/`. The
  restriction is sound:
  - Integer-only calls: s1 and s2 are silent.
  - Decimal or division calls: o2 and o3 show expr_gen stays silent.
  - Degenerate case `Sizes 2.5 3 * 4 =` (s15 / o1): only calc_gen fires (12). The answer is correct, and the two tools do
    not collide.
  - The team reports BOTH=0 over 5 x 400 random expressions (test_tools.py).

### Rule 12: works outside the benchmark's style
- BBH multistep_arithmetic_two is integer-only, so calc_gen does almost nothing on its own benchmark. All evidence of use
  comes from other settings: money, unit conversion, ratios, code-like decimals, and the shopping prompt in host runs.
- In host runs it fired correctly on 3 of 5 decimal-price word problems (h1, h2, h5). The host kept writing normally after
  the end write (e.g. ` dollars`).
- Plain accuracy was already 5/5 on these easy questions, so the runs show correct use, not an accuracy gain.

### Rule 13: what f computes, stated without the benchmark
- **calc_gen.** f(text) = the exact value of `a op b`, or of a chain of two operations with normal precedence:
  - A and B are signed decimals, up to 18 integer digits and 6 decimals;
  - op is `+ - * /` (`x` and `×` also mean times);
  - division is truncated to 6 decimals;
  - it fires at the `=` the writer puts after the expression, and is silent on anything it cannot certify.
- **calc_call.** The same f for `<tool_call>calc A op B</tool_call>`.
- **USEFUL-GENERALLY: YES.** Three settings:
  1. Money: prices with cents, totals, change, tips, tax, splitting a bill (a2, a4, h1, h5).
  2. Unit conversions and rates: cm per inch, km per liter, hourly pay, cutting lengths (a5, a12, h2).
  3. Exact decimal ratios and averages in science, engineering or code, where float error matters (`0.1 + 0.2 = 0.3`, a9;
     `-7.5 / -2.5`, a13).

## Rule 11: variants and results

Setup:
- The team ran the variants on the GPU machine with the deployed config: veto `runs/veto_qwen3.pt`, the model writes the space
  after `=`, the same as run_gen.py.
- Raw results: `bbh_generic/teamC/judge/results_calc_gen_1.json`.
- The variants file is `judge/variants_calc_gen_1.json`. My generator did not produce any of them, and they use other
  wording, numbers, notation and an irrelevant sentence (h4).
- `<END>` in the results marks the end write.

### Tool alone, calc_gen: should fire

| id | text (end) | wrote | result |
|---|---|---|---|
| a1 | The bill is 12.5 * 4 = | 50 | OK |
| a2 | ...six slices cost $3.75 * 6 = | 22.5 | OK |
| a3 | Pi is roughly 22 / 7 = | 3.142857 | OK |
| a4 | ...among 8 friends gives 100 / 8 = | 12.5 | OK |
| a5 | ...12 inches is 12 x 2.54 = | 30.48 | OK |
| a6 | The floor measures 3.5 × 4.2 = | 14.7 | OK |
| a7 | ...for 4 hours: 4 * -1.5 = | -6 | OK |
| a8 | Balance after the fee:\n-2.5 + 0.75 = | -1.75 | OK |
| a9 | In exact decimal arithmetic 0.1 + 0.2 = | 0.3 | OK |
| a10 | She paid 2.5 * 4 + 3 = | 13 | OK |
| a11 | What is left: 100 - 4 * 12.75 = | 49 | OK |
| a12 | The car does 450 / 37.5 = | 12 | OK |
| a13 | Ratio:\n-7.5 / -2.5 = | 3 | OK |
| a14 | ...three bags weigh 0.45*3= | 1.35 | OK |
| a15 | It costs 4.5. Then 2.5 * 4 = | 10 | OK |
| a16 | `7.5 / 2 = 3.` (text STARTS with the number) | silent | MISS, safe (see below) |

### Tool alone, calc_gen: must be silent, or must not give a wrong value

| id | text | wrote | result |
|---|---|---|---|
| s1 | Three rows of four chairs: 3 * 4 = | silent | OK (integer: expr_gen owns it) |
| s2 | Total: $12 * 3 = | silent | OK |
| s3 | Dividing by zero: 5.5 / 0 = | silent | OK |
| s4 | Three steps: 1.5 + 2 + 3 + 4 = | silent | OK |
| s5 | With x unknown, 2.5 * x = | silent | OK |
| s6 | The mean is (7.5 + 8.25) / 2 = | silent | OK (did not answer 4.125) |
| s7 | The sum is 1,250.5 + 3 = | silent | OK (did not answer 253.5) |
| s8 | assert 0.1 + 0.2 == | silent | OK |
| s9 | The value is -2.5 * 4 = | silent | OK, safe (did not answer 10) |
| s10 | 3.1415926 * 2 = (7 decimals) | silent | OK |
| s11 | 192.168.1.1 + 1 = | silent | OK |
| s12 | 4.5 ÷ 1.5 = | silent | OK, safe |
| s13 | 12.5% * 8 = | silent | OK |
| s14 | The bill is 12.5 * 4 = 50. | end write only, no digits | OK (dot veto) |
| s15 | Sizes 2.5 3 * 4 = | 12 | OK (only calc_gen fires) |
| o1 | expr_gen on the same text | silent | OK, no collision |
| o2 | expr_gen on 12.5 * 4 = | silent | OK |
| o3 | expr_gen on 22 / 7 = | silent | OK |

### Tool alone, calc_call

| id | call | wrote | result |
|---|---|---|---|
| c1 | calc 12.5 * 4 | 50 | OK |
| c2 | calc 22 / 7 | 3.142857 | OK |
| c3 | calc 7 * 6 | 42 | OK |
| c4 | calc -1.25 + 0.5 | -0.75 | OK |
| c5 | calc 3.5*-2 | -7 | OK |
| c6 | calc 2.5 × 4 | 10 | OK |
| c7 | calc 100 - 4 * 12.75 | 49 | OK |
| c8 | prose before the call, then calc 2.75 * 18 | 49.5 | OK |
| c9 | calc 1 / 0 | empty response | OK |
| c10 | calc x * 2 | empty response | OK |
| c11 | calc 1 + 2 + 3 + 4 | empty response | OK |
| c12 | calc $4.50 * 3 | NO response | safe, see fix 1 |
| c13 | expr 2.5 * 4 (wrong name) | nothing | OK |
| c14 | calc (2 + 3) * 1.5 | NO response | safe, see fix 1 |

### Host + tools (team's shopping prompt, frozen Qwen3-0.6B-Base, greedy)

`[...]` marks steps where an implant was writing.

| id | question (short) | target | plain | with tools | where the tool fired |
|---|---|---|---|---|---|
| h1 | 8 bottles at 1.75 dollars | 14 | 14 | 14 | `8 * 1.75 = [1][4][ dollars]` (calc_gen) |
| h2 | 22.5 m ribbon cut into 2.5 m pieces | 9 | 9 | 9 | `22.5 / 2.5 = [9][.][ So]` (calc_gen, then dot veto) |
| h3 | 45 L at 1.6, pay 100 | 28 | 28 | 28 | the host wrote `45 liters * 1.6 dollars per liter`, which is not a call; expr_gen did `100 - 72` |
| h4 | 12.5/hour for 36 h, minus 150, irrelevant sentence | 300 | 300 | 300 | the host put units inside the expression; nothing fired |
| h5 | 5 packs at 4.2, pay 50 | 29 | 29 | 29 | `5 * 4.2 = [2][1][ dollars]` (calc_gen), then expr_gen `50 - 21` |

### Summary
- Tool alone: 46 of 48 scored correct.
- The 2 misses (a16, s9) were both silent, which is the safe outcome. **There were 0 wrong values.**
- Host runs: 5/5 with tools, the same as plain.
- calc_gen fired in 3 of 5 host runs and was exact each time. calc_gen and expr_gen worked together inside one answer (h5).
- The team also reports, from runs I did not see:
  - BOTH=0 and calc_gen silent on every integer-only expression (test_tools.py, 5 seeds x 400 random expressions);
  - 0 bad results over about 1,000 random call-format calls, about 25% of them random decimal calc calls (test_calls.py).

## Limitations (all fail safe, none is test fitting)
- **No start class at the start of the text.** An expression that opens the context has no token before A, so it is
  silent (a16). The team reports that with any left context (`So 7.5 / 2 = 3.`) the dot logic gives `75`. In real use the
  host always writes text first.
- **Negative first operand after a word** (`is -2.5 * 4 =`, s9) is silent. A negative first operand works at a line start,
  after whitespace, or in the call format.
- **Coverage gaps, all silent:** parentheses, `÷`, `%`, thousands separators, more than 2 operations, more than 6 decimals.
  When the host writes long expressions or units inside the expression (h3, h4), neither tool fires. This is a coverage
  limit, not a correctness problem.

## Required / recommended fixes
1. **calc_call, required before the rule-14 integration step.**
   - Problem: a call with a token outside the grammar (`$`, `(`) gets NO `<tool_response>`. A fine-tuned host would then
     write the response itself, possibly a made-up value.
   - Rule 14 says an empty RESULT means an error.
   - Fix: delimit the call by the envelope tokens (`<tool_call>` ... `</tool_call>` after the name `calc`) instead of by the
     run of calc tokens, so that every call named `calc` gets at least `<tool_response></tool_response>`.
   - The current behavior is now documented correctly in callfmt_C.py. That was the docstring inconsistency I flagged in
     round 1, and the team has fixed it.
2. **Recommended, not blocking.**
   - Add a start-of-text / BOS start class so that an expression at the very start of a context can fire (a16).
   - Make run_variants.py host mode take `--veto`, so that judge host runs use the deployed config by default. The team ran
     this round with the veto by hand.
3. **Reporting (rule 12).**
   - Add a calc_gen out-of-style row (e.g. the h1-h5 decimal shopping questions, or a generated decimal-price set) next to
     DEV/FRESH in the NOTEBOOK.
   - Without it, the notebook has no evidence that calc_gen is ever used. It is effectively a passenger on the integer-only
     BBH task.
