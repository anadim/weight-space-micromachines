# MATH-500 failure analysis for generic weight-space implants (2026-11-17)

Analysis only. No implants were built. All model compute ran on one GPU.

## Setup
- Models: Qwen3-0.6B-Base and Qwen3-1.7B-Base, frozen, bf16. Minerva 4-shot prompt (gen.py, the standard Minerva MATH prompt).
  Greedy, max_new_tokens 1024, batched with left padding and an attention mask, stop string "\nProblem:" (gen.py).
- Grading (grade.py): last \boxed{} with nested braces, else the last "The final answer is $...$". The answer counts as correct
  if the Hendrycks-style normalized strings match OR math_verify says the two are equal.
  Note: "The final answer is east." (no $) is NOT extracted, so it counts as a format failure (0.6B ids 97 and 255).

## Accuracy
| | Qwen3-0.6B-Base | Qwen3-1.7B-Base |
|---|---|---|
| overall | 151/500 = 30.2% | 203/500 = 40.6% |
| Algebra | 50/124 = 40.3% | 80/124 = 64.5% |
| Counting & Probability | 8/38 = 21.1% | 11/38 = 28.9% |
| Geometry | 13/41 = 31.7% | 18/41 = 43.9% |
| Intermediate Algebra | 13/97 = 13.4% | 22/97 = 22.7% |
| Number Theory | 17/62 = 27.4% | 20/62 = 32.3% |
| Prealgebra | 40/82 = 48.8% | 41/82 = 50.0% |
| Precalculus | 10/56 = 17.9% | 11/56 = 19.6% |
| Level 1 | 27/43 = 62.8% | 32/43 = 74.4% |
| Level 2 | 46/90 = 51.1% | 57/90 = 63.3% |
| Level 3 | 43/105 = 41.0% | 48/105 = 45.7% |
| Level 4 | 25/128 = 19.5% | 45/128 = 35.2% |
| Level 5 | 10/134 = 7.5% | 21/134 = 15.7% |
| no answer extracted (failures) | 68 | 62 |
| hit 1024 tokens (failures) | 64 | 54 |
| exact repetition loop (failures) | 48 | 34 |

Overlap: 113 problems are solved by both models, 38 by the 0.6B only, and 90 by the 1.7B only. Greedy outcomes are fragile.

## Read-through (by hand)
I read 120 stratified 0.6B failures and 40 1.7B failures in full (read_0.6b.txt, read_1.7b.txt; labels in annotations.py).
For each one I recorded the FIRST error, every error class present, and which single tool would fix it ALONE
(strict: the rest of the solution would then have been right).

| class | 0.6B first (n=120) | 0.6B any | 1.7B first (n=40) | 1.7B any |
|---|---|---|---|---|
| wrong method / plan (METH) | 35 | 59 | 9 | 16 |
| hallucinated step, fact or fake code output (HALL) | 17 | 32 | 8 | 12 |
| symbolic algebra slip: expand/factor/simplify/substitute (ALG) | 18 | 25 | 8 | 8 |
| slip while solving an equation/system/congruence (SOLVE) | 7 | 11 | 4 | 6 |
| plain-number arithmetic slip (ARITH) | 10 | 21 | 4 | 7 |
| fraction / radical / power slip (FRAC) | 7 | 8 | 1 | 1 |
| modular arithmetic slip (MOD) | 3 | 3 | 2 | 2 |
| base conversion slip (BASE) | 1 | 3 | 0 | 0 |
| wrong formula recalled (FORM) | 10 | 13 | 1 | 3 |
| misread question (MISR) | 7 | 8 | 1 | 1 |
| miscount in enumeration (COUNT) | 0 | 1 | 2 | 2 |
| repetition loop / never answers (LOOP) | 3 | 18 | 0 | 5 |
| answer format / not boxed (FMT) | 2 | 8 | 0 | 1 |

Fix ALONE by hand (strict, '?' cases counted):
- exact calculator (LaTeX numbers, fractions, radicals, binomials, %, mod): 0.6B 16/120 [2,29,57,116,144,223,241,261,268,270,283,310,311,325,417,451];
  1.7B 5/40 [35,38,97,111,144]
- equation / system solver: 0.6B 9/120 [59,114,231,258,290,346,424,456,464]; 1.7B 6/40 [7,81,284,297,316,354]
- expand/simplify/factor (CAS): 0.6B 7/120 [84,88,157,159,252,347,417]; 1.7B 3/40 [36,84,241]
- base conversion: 0.6B 2 [30,70]; modular power: 0.6B 1 [415]; loop breaker: 0.6B 2? [220,311]
- answer checker (plug the answer back in): 0.6B 1 [290]; 1.7B 3 [201,316,342]
- format only (right answer, not boxed): 0.6B 2 [97,255]
About 2/3 of failures (METH+HALL+FORM+MISR first) have no exact-tool fix: the plan or the fact is wrong.

## Oracle injection (measured upper bounds)
Code: oracle_lib.py (fire-point detection), oracle_run.py (cut, inject, continue). Python only builds the oracle text.
It is a measurement, not an implant. The frozen model continues greedily after each injection (budget max(384, 1024 - prefix)).
The loop repeats up to R rounds: after each continuation the next fire point is searched only in the new text.
- Every '='-chain inside $..$, $$..$$, \[..\], align is parsed with latex2sympy (lowercase off; (a)(b) and 2(3) get explicit \cdot).
- calc (num): a closed numeric claim A = B (fractions, radicals, powers, binomials, factorials, %) where B != A exactly.
  The oracle replaces B with the exact value of A (inside \boxed{} if the model had boxed it). R=4.
- algebra (sym): a claimed rewrite A = B with the same free symbols, A or B having something to expand/factor/cancel,
  equal syntactic degree and at least one shared term, and A - B != 0. The oracle replaces B with expand/factor/cancel/apart(A). R=4.
- solver (solve): a 1-variable equation (polynomial deg <= 4, or with radicals) or a linear system that the model then
  answers with a value that is not a root. The oracle injects ' \implies x = <exact roots>' right after the equation. R=4.
- loop: a word block written 3 times in a row. At the start of the 3rd copy the oracle injects
  "Wait, I am repeating myself. Let me stop and try a different approach." (loop) or "...Let me stop here and give my final answer." (loopcommit).
- check (oracle verifier, uses the gold answer): before "Final Answer:" it injects "...It checks out." when the answer is right,
  or "...It does not check out, so the answer above is wrong. Let me redo the problem carefully." when it is wrong. R=4 (1.7B: R=2).
- rethink: unconditional "Hmm, wait. Let me rethink this." before "Final Answer:" (no oracle). R=1.
- all: loop breaker + solver + calc + algebra, whichever fires first, R=6.
- NOOP controls: the same cut points with nothing inserted. They measure re-tokenization / batch noise.
  Noise is real: a few fragile items (0.6B ids 122,258,265,270,310,346,453) flip to correct with NO injection.

Accuracy after injection (n=500). 'noise-adj' removes items that the NOOP control at the same cut points also flips.

| implant (oracle) | model | fires on fail/right | fixed | broke | acc | noise-adj fixed | noise-adj broke | net adj | fixed ids |
|---|---|---|---|---|---|---|---|---|---|
| calc: LaTeX exact calculator | 0.6b | 57/0 | 17 | 0 | 151->168 (30.2->33.6%) | 13 | 0 | +13 | [2, 33, 58, 92, 144, 181, 192, 241, 254, 325, 331, 369, 453] |
| calc: LaTeX exact calculator | 1.7b | 41/1 | 8 | 0 | 203->211 (40.6->42.2%) | 5 | 0 | +5 | [33, 92, 143, 158, 203] |
| equation/system solver | 0.6b | 29/2 | 9 | 1 | 151->159 (30.2->31.8%) | 7 | 0 | +7 | [114, 199, 231, 322, 344, 424, 464] broke [7] |
| equation/system solver | 1.7b | 17/4 | 3 | 1 | 203->205 (40.6->41.0%) | 3 | 1 | +2 | [212, 284, 354] broke [23] |
| algebra: expand/factor/simplify | 0.6b | 11/1 | 2 | 1 | 151->152 (30.2->30.4%) | 2 | 1 | +1 | [139, 456] broke [376] |
| algebra: expand/factor/simplify | 1.7b | 10/4 | 3 | 2 | 203->204 (40.6->40.8%) | 3 | 1 | +2 | [63, 66, 394] broke [376, 429] |
| combined: loop+solver+calc+algebra | 0.6b | 134/3 | 27 | 2 | 151->176 (30.2->35.2%) | 21 | 2 | +19 | [2, 33, 58, 92, 114, 139, 144, 181, 192, 231, 241, 254, 322, 325, 331, 344, 369, 390, 417, 424, 464] broke [7, 376] |
| combined: loop+solver+calc+algebra | 1.7b | 98/9 | 15 | 3 | 203->215 (40.6->43.0%) | 13 | 2 | +11 | [33, 63, 66, 92, 143, 158, 172, 203, 212, 284, 354, 390, 394] broke [23, 376, 429] |
| loop breaker -> different approach | 0.6b | 48/0 | 0 | 0 | 151->151 (30.2->30.2%) | n/a | n/a | n/a | [] |
| loop breaker -> different approach | 1.7b | 34/0 | 0 | 0 | 203->203 (40.6->40.6%) | n/a | n/a | n/a | [] |
| loop breaker -> commit to answer | 0.6b | 48/0 | 5 | 0 | 151->156 (30.2->31.2%) | n/a | n/a | n/a | [69, 74, 119, 147, 267] |
| loop breaker -> commit to answer | 1.7b | 34/0 | 2 | 0 | 203->205 (40.6->41.0%) | n/a | n/a | n/a | [267, 351] |
| ORACLE verifier (uses gold) + redo | 0.6b | 287/151 | 14 | 1 | 151->164 (30.2->32.8%) | 14 | 1 | +13 | [26, 28, 57, 93, 122, 131, 192, 247, 322, 344, 347, 442, 461, 477] broke [227] |
| ORACLE verifier (uses gold) + redo | 1.7b | 241/203 | 4 | 0 | 203->207 (40.6->41.4%) | n/a | n/a | n/a | [28, 144, 321, 476] |
| unconditional rethink (no oracle) | 0.6b | 287/151 | 21 | 12 | 151->160 (30.2->32.0%) | 21 | 12 | +9 | [23, 35, 59, 107, 114, 122, 125, 172, 192, 258, 261, 313, 322, 344, 353, 368, 424, 441, 442, 484, 489] broke [31, 38, 55, 73, 127, 227, 256, 370, 389, 437, 449, 455] |
| unconditional rethink (no oracle) | 1.7b | 241/203 | 11 | 18 | 203->196 (40.6->39.2%) | n/a | n/a | n/a | [28, 66, 113, 157, 172, 176, 207, 254, 439, 451, 488] broke [16, 32, 51, 65, 70, 87, 122, 125, 128, 169, 192, 216, 225, 341, 348, 456, 461, 496] |


### What the oracle shows
- The calculator is the largest single tool. Nearly all of its fixes are LaTeX fraction, radical or power expressions;
  among the 17 0.6B fixes, 15 are \frac/\sqrt/^ expressions, 1 is \binom and 1 is a plain integer.
  A plain `a*b=` reader would get about 1-2 of them.
- Splice matters. A first calculator run inserted the bare value plus a space ("= \frac{14}{3} ") where the model had planned `\boxed{...}`.
  The model then kept extending the expression ("\frac{14}{3} + \frac{5}{3} = \boxed{19/3}") and ignored the value: 8 fixes instead of 17 (0.6B).
  An implant must write a CLOSED value in the model's own wrapper (or end the statement) or the model treats it as a partial result.
- Solver: of the 29 0.6B failures where it fired, 9 had the right equation and botched the solve (all 9 fixed, 7 net of noise).
  2 had the right setup with later errors (170, 380), 1 had an arithmetic slip inside the setup (354), and about 17 had a WRONG
  equation or garbage setup (23,36,99,101,172,173,184,202,225,249,297,382,394,409,436,473,484), where a solver cannot help.
  1.7B: 3 botched solves fixed (212,284,354). In 290 the solver gave r=4 but the model talked itself back to 7/2. The other 12 were wrong setups.
- Algebra (expand/factor/simplify): small (+1 each). Every break was a detector misfire: a real equation such as
  sqrt((a-10)^2+(b+10)^2) = sqrt(a^2+b^2) (id 376), or a step that uses a constraint from context (429), read as a false identity.
  The hand read finds more expand slips (0.6B 84,88,157,159,252,347), but they are written in prose ("Expanding, we get ...")
  with no `A =` chain, so a passive trigger cannot see them. A CAS needs an explicit, fine-tuned call.
- Loops: "try a different approach" fixes 0/48 (0.6B) and 0/34 (1.7B). The model re-enters a loop (only 1/48 even answers).
  "give my final answer" fixes 5/48, but these are mostly bare guesses or memorized answers (147, 267 come from nowhere; 74 is an unsimplified
  expression that math_verify accepts as equal to cot x). Treat this as reward-hacking-prone.
- Verifier: even a PERFECT correctness signal + "redo carefully" only gets +13 (0.6B, 4 rounds) and +4 (1.7B, 2 rounds).
  The model usually cannot redo a problem correctly once told it is wrong. The unconditional rethink line gets 0.6B +21/-12. Its breaks come from
  doubt loops ("I made a mistake" repeated), which a loop-commit could catch.
- Hand estimate vs oracle: in the 120 read 0.6B failures I judged 31 fixable by calc/solver/CAS. The passive '=' oracle realizes 13 of them.
  The others have no '=' fire point: "is equal to \boxed{10}", |M ∩ S| variables, prose expansions, the reversed "55 = 9·6+3", \text in the chain.
  So a model-CALLED tool (fine-tuned to emit the call) could plausibly reach about 2x the passive numbers.

## Ranked shortlist of generic implants (by measured gain; 0.6B noise-adjusted net, 1.7B raw)
1. LaTeX-aware exact calculator. f(expr) = the exact value of a closed numeric expression (integers, decimals, %, \frac,
   \sqrt / n-th roots with simplified surds, integer powers, n!, \binom), written in simplest exact form.
   Outside MATH: invoices and spreadsheets in text ("15\% of 240 ="), physics/chemistry notes and lab reports, unit and size
   arithmetic in docs/code comments (2^{20}\cdot 4 bytes), recipe/dose scaling.
   Call: passive trigger on '=' after a closed numeric expression inside math mode (no fine-tuning if the reader handles LaTeX),
   or a fine-tuned <tool_call>calc ...</tool_call> to reach prose computations ("which is equal to").
   Build: flat + - * / ^ on rationals is moderate; nested \frac{}{}/\sqrt{} parsing and surd simplification in weights are hard.
   Measured: 0.6B +17 raw / +13 adjusted, 0 broken [2,33,58,92,144,181,192,241,254,325,331,369,453 (+122,265,270,310 also flip under no-op)];
   1.7B +8 raw / +5 adjusted, 0 broken [33,92,143,158,203 (+57,172,220 also flip under no-op)].
2. Equation / small-system solver. f(equations) = exact solution set of a 1-variable polynomial equation (deg <= 2 exact with surds,
   deg 3-4 via rational roots + numeric), a simple radical equation, or a linear system up to 3x3.
   Outside MATH: physics/engineering word problems (kinematics, circuits), finance (break-even, loan rate for fixed payment),
   mixtures/rates in business text, solving for a parameter in code or config.
   Call: needs fine-tuning to emit <tool_call>solve EQS ; VARS</tool_call> after writing the equation (variable choice cannot be inferred passively).
   Build: fixed-size Gaussian elimination and the quadratic formula are feasible; surd simplification and deg 3-4 are harder.
   Measured: 0.6B +9 raw / +7 adjusted, 0 net broken [114,199,231,322,344,424,464]; 1.7B +3/-1 [212,284,354; broke 23].
   Ceiling: about 60% of fired failures have the wrong equation.
3. Verifier / answer checker. f(problem constraints, candidate) = check result. The buildable part is plug-back
   (substitute the candidate into the equation/conditions; domain checks such as zero denominators and an even root of a negative).
   Outside MATH: unit tests for code, checking extracted values against a spec, constraint checks in scheduling.
   Measured ONLY as an oracle (it uses the gold answer, so it is an upper bound, not a tool): 0.6B +14/-1, 1.7B +4/0.
   The realistic plug-back subset is smaller (0.6B 290, 1.7B 201, 316, 342 by reading).
   Unconditional "Hmm, wait. Let me rethink this." (no oracle): 0.6B +21/-12 = +9 net; 1.7B: +11/-18 = -7 net (it HURTS the stronger model).
4. Loop handler. f(text) = detect that a block was written twice and stop. "Different approach": 0 (both models).
   "Commit to an answer": 0.6B +5 (guess-like); 1.7B +2 (267, 351). Useful only to stop wasted tokens, or combined with rethink to cut doubt loops.
5. Expand / factor / simplify (CAS). f(expr) = canonical expanded/factored/cancelled/partial-fraction form.
   Outside MATH: simplifying formulas in physics/engineering, symbolic derivations in papers, code simplification of expressions.
   Measured: 0.6B +2/-1, 1.7B +3/-2 (breaks are misfires on equations). Hand reading suggests about 6-7/120 if explicitly called (needs fine-tuning).
6. Small number-theory/base tools (base conversion, modular power/inverse, gcd/lcm/factorization): by hand, base conversion fixes
   0.6B 30, 70 (and helps 71, 403); modpow 415; most NT failures are method errors. No oracle run (rarely written as '=' chains).
Combined calc + solver + algebra + loop break (R=6): 0.6B 151 -> 176 (+27/-2 raw, +21/-2 adjusted, 35.2%);
1.7B 203 -> 215 (+15/-3 raw, +13/-2 adjusted, 43.0%).

## Reward-hacking risks (MATH-style only)
- Auto-boxing / "The final answer is $...$" formatting fixes 0.6B 97 and 255 (answer right, not boxed). Pure benchmark format.
- Loop-commit gains are mostly guesses/memorized answers (147, 267), and one is grader leniency (74, an unsimplified expression accepted).
- The checker and rethink triggers fire before "Final Answer:", which is Minerva-prompt-specific. A generic trigger must be "about to state
  the final answer" in any style.
- Answer-form converters (decimal -> common fraction for "express as a common fraction", "express in base 8") match MATH instructions;
  base conversion itself is generic, the formatting step is not.
- The oracle verifier uses the gold answer. It is an upper bound only, never an implant.

## Noise and caveats
- Greedy re-generation from a cut point is not bit-identical: re-tokenization and batch padding flip a few fragile items.
  NOOP controls: solver 0.6B +2/-2, 1.7B 0/0; calc 0.6B +4/0, 1.7B +4/0; algebra 0.6B 0/0, 1.7B 0/-2; combined 0.6B +7/0, 1.7B +3/-2. The adjusted columns subtract these.
- Fire-point detection uses heuristics. Misfires are the source of every algebra break and of solver id 7 (a proportionality check).
- Three 1.7B runs went out of memory once when too many processes shared the GPU; they were re-run with at most 2 processes at a time.

## Files
gen.py, grade.py, autopass.py (generate, grade, automatic first pass), dump.py (stratified failure dumps for reading; ids in
results/read_*.ids.json), annotations.py (hand labels), oracle_lib.py, oracle_run.py, run_seq.sh, final_table.py.
Released result files are reduced (results/graded_*.json, results/runs/or_<model>_<kind>.json: per-problem correctness and
fire flags, no problem text and no generated text). Superseded runs (calculator without the closed-value splice, an earlier
parser) are not included.
