# Team C notebook: multistep_arithmetic_two, boolean_expressions, object_counting

Binding protocol: ../PROTOCOL.md (rules 1-14). No BBH test data, no old per-question outputs were
opened. Only the three official CoT prompt files were downloaded (curl of `cot-prompts/<task>.txt` at the pinned commit,
into `cot/`). Host: frozen Qwen3-0.6B-Base, greedy, same loop as `scripts/run_bbh.py` (`run_gen.py`).
Runs: one H100 (GPU 0), from a copy of `bbh_generic/teamC/`.

## Files
- `gen_tasks.py` generators (dev seed 0 x30, fresh test seed 1 x100; no train split: no fine-tuning in the frozen path)
- `teamc_tools.py` the tools (compiled ReLU circuits + solved readers / write directions; `attach_gen`)
- `run_gen.py` harness; `launch.sh`; `test_tools.py` machine-alone exactness tests

## Rule 13: what each tool computes, and is it generally useful? (stated before building)
- **expr_gen**: f(text) = the exact value of an integer arithmetic expression written in text (+, -, *, x, ×, unary minus,
  parentheses), at the moment the writer puts `=` after it. USEFUL-GENERALLY: yes. Settings: (1) word problems / GSM-style
  multi-step arithmetic where the model writes `3 * (12 + 5) =`; (2) totals of shopping lists, invoices, budgets, scores
  (`4 + 2 + 7 + 1 =`); (3) checking integer expressions in code or spreadsheets (`(a+b)*c` with numbers substituted).
- **calc_gen**: f = exact a op b (and two-op chains) on signed decimals with + - * /. Yes: prices/discounts, unit
  conversion, averages. Restricted to calls with a decimal point or a division so it never overlaps expr_gen.
- **bool_gen**: f(text) = truth value of a propositional formula over True/False with not/and/or/parentheses (Python
  precedence), at `=`. Yes: (1) evaluating `if` conditions when reasoning about code, (2) truth tables / logic exercises,
  (3) feature-flag / rule checks ("enabled = not maintenance and (beta or admin)" with values substituted).
- **counting**: no separate tool. Counting items in a list = summing the quantities the model writes, which is expr_gen's
  job. A dedicated "count objects of category C" tool would need world knowledge (is a fig a fruit?), which is not a
  mechanical function; see the object_counting section for whether expr_gen helps at all.

## Baselines (plain host, same harness)
| task | DEV (n=30) | FRESH TEST (n=100) |
|---|---:|---:|
| multistep_arithmetic_two | 66.7 | 59.0 |
| boolean_expressions | 90.0 | 77.0 |
| object_counting | 43.3 | 31.0 |

## Log

### L1. Generators (before any tool work)
- First object_counting dev set had distractor items in EVERY question; the plain host then scored 0/30 (it lists every
  item, never filters by category; looked at 3 dev outputs). The public task description has both plain counting
  ("how many objects") and category counting, so the generator now makes 1/3 of questions distractor-free, 1/6 "how many
  items/things" (everything counts), the rest with 1-5 (dev) / 1-6 (test) distractors. Dev and test regenerated; plain
  baselines re-run. Not fitted to any tool (no tool existed yet).

### L2. Audit of the old tools (code read, no BBH data)
- `expr_n` (BBH multistep): only 2-4 SINGLE-DIGIT numbers, brackets only as one outer pair. That is BBH's number range,
  not a generic evaluator -> replaced by a new generic evaluator `expr_gen`.
- `sum_fn`: generic sums, but its start-token list contained ' A', ' B', 'A', 'B' (the variable names of the BBH
  multistep exemplars). Removed: `expr_gen` uses a rule instead of a start list (any token may precede the expression
  except a written-down guard list of things that can be part of an operand: letters, '.', ',', '/', '^', '%', '**',
  unicode minus/dashes, ...). '.' and ',' only guard when glued to a digit ('3.5', '1,000'), so prose like
  'more, 5 + 3 =' or 'is 5. 3 + 4 =' works.
- `boolnat`: the expression had to start right after '=', <= 16 words, and merged bracket tokens ('((', '))', '(not',
  ')=') made it silent. Replaced by a scan (pushdown) evaluator `bool_gen` that reads every vocabulary token of the form
  [' '] ')'* '('* [True|False|true|false|not|and|or].
- `calc_s4`: generic (a op b on signed decimals, two-op chains). Kept as `calc_gen` with one restriction (only calls with
  a decimal point or a division, so it never overlaps expr_gen) and one generalization found by our own tests: a
  negative second operand written with its own sign token after a spaced operator ('3.5 * -2') made it silent;
  `calc_neg.py` is calc.py's stack 1 with that case added.
- The old ". \n" end (sum_fn/expr_n 'OE' = '.\n') and the boolnat '.' after the value were fixes found on BBH failures.
  We started without them (expr_gen ends with the solved veto = "no more digits, host continues") and re-tested them on
  DEV only (L4).

### L3. expr_gen design and bugs
- Stack machine over the token span before '=': registers A (sum of finished terms), R (running product of the current
  term, including its sign and the bracket's base coefficient B), stack frames (A, B) for up to 4 open brackets. A '('
  pushes and starts the inner sum with B = R (so the bracket's value arrives already multiplied); ')' pops with
  R := A + R. Every product is (register) x (one digit), done digit by digit: R = Rs * (number read so far). So any
  integer sizes work, all exact; |values| <= 1e12 else silent. Runs in float64 (21.7k ReLU units, depth ~1000 layers).
- Bug 1: hdl.compile_net stores weights in float32; the gate constant 1e14 was rounded -> garbage values. Fixed with a
  float64 compile (`compile_net64`).
- Bug 2: digit extraction multiplied 10^k by a difference of two large ReLU wires (inexact in float64). Fixed by passing
  each digit flag through one more ReLU (a clean 0/1 wire).
- Machine-alone tests (`test_tools.py expr`, random expressions with 1-6 digit numbers, nesting <= 3, unary minus,
  x/×, spacing variants, 11 different left contexts): seeds 0/1/2: 0 WRONG, ~92% exact, the rest silent (expression
  longer than the 72-token window, 'x' glued to a digit, the token '(-(' which is not in the grammar). Must-be-silent
  list (decimals, '1,000', variables, '**', '/', '^', unbalanced, single number) all silent.

### L4. DEV runs (frozen host + official 3-shot prompt + our dev questions)
| setting | multistep | boolean | counting |
|---|---:|---:|---:|
| plain | 66.7 | 90.0 | 43.3 |
| expr_gen, end = veto (host continues) | 63.3 | | 50.0 |
| expr_gen, end = '.' | 93.3 | | |
| expr_gen, end = '.\n' | **100.0** | | 50.0 |
| bool_gen, value only | | 96.7 | |
| bool_gen, value then '.' | | **100.0** | |
- DEV failure that motivated '.\n' (veto variant): `A = (-8 * -2 - 3 - -6) =[ 19] - 3 - (-6) =[ 22] - (-6) =[ 28]`:
  the host treats the tool's value as the first piece of its own step-by-step rewrite and copies the rest of the
  expression after it, so the tool's value is used as a partial result (wrong). Ending the value with '.\n' tells the
  host "this computation is finished" (same idea as a closing </tool_response>). '.' alone fails differently on DEV: the
  host continues '[10][.]5' as a decimal. Generic reading: an end-of-result marker that cannot be confused with a
  continuation of the number. Kept '.\n'.
- DEV failure that motivated bool '.': `Z = not not True and not False and True =[ True] and False and True`: same
  mechanism (host continues the expression after the tool's value). Kept the '.' after the value.
- Counting: the tool fixes the host's addition errors (the host's own sum was wrong in 8/30 dev questions), but most
  counting errors are the host listing items of the wrong category (e.g. animals and tools listed as vegetables). No
  mechanical tool can fix category knowledge, so no counting-specific tool: counting = expr_gen on the sum.

### L5. bool_gen rewrite (scan) and tests
- Scan/pushdown evaluator over the token span before '=' (0/1 registers: A = OR of finished and-terms, R = AND of the
  current term times the bracket's base B, pending-not parity, per-bracket not parity; frames for 4 levels). 16.2k units.
- Machine-alone tests (`test_tools.py bool`, random formulas, depth <= 3, spaced and unspaced brackets, lowercase literals,
  8 left contexts): seeds 0-3: 0 WRONG; everything with an operator exact; single literals and over-long formulas silent.

### L6. Genericity rules added from our own tests (not from DEV failures)
- Separation rule (expr_gen, bool_gen): the expression must be separated from what precedes it: its first token starts
  with a space, or the token before is a separator ('=', ':', newline tokens), or it is the start of the text. Found by
  the random test `A = ((-560*233307)-587616)x747525+158=` (token ')x' glued) and the cases 'sqrt(4+5)=', 'f(3) + 2 =',
  'abc2 + 3 =': all now silent instead of evaluating a suffix.
- Start of text counts as a separator (symbol ANY = 1 on every real token, 0 in the padding before the first token).

### L7. Call-format versions (protocol rule 14)
- `expr_call`, `bool_call`, `calc_call`: same circuits, envelope `<tool_call>NAME ARGS</tool_call>` -> the tool writes
  `<tool_response>RESULT</tool_response>`; RESULT empty for a call it cannot evaluate. Grammar: `callfmt_C.py`.
- The four envelope tokens share ONE embedding row in the base checkpoint (cos = 1), so no reader can tell them apart.
  teamA/ft.py gives them distinct random rows before fine-tuning; `prep_call_tokens` does the same (same seed and order)
  for testing on the un-fine-tuned host.
- calc_call: calc_s4 had no symbol slot for '<tool_call>'; the input symbol 'OE' (never read by calc's circuits) lists
  '<tool_call>' as an alternative token, so the call is delimited by the envelope (judge calc round 1: every calc call must
  get a response).
- `test_calls.py` (random calls: integer expressions, formulas, decimal calc calls incl. negative operands and division,
  plus edge cases and other tools' calls): 0 bad over 4 seeds after the calc_neg fix.

### L8. Out-of-style settings (rule 12): own 3-shot prompts, generated questions (seed 3, n = 60 each), frozen host
| setting | plain | + tools | (lenient answer extraction: plain / tools) |
|---|---:|---:|---:|
| num_shop: shopping word problems (prices x quantities, change), expr_gen/calc_gen in prose | 73.3 | 75.0 | 90.0 / 95.0 |
| num_list: inventory list with 2-digit quantities, total count, expr_gen | 33.3 | **100.0** | 33.3 / 100.0 |
| bool_code: code condition with variable values given, bool_gen | 80.0 | **98.3** | 81.7 / 98.3 |
- Lenient extraction (`rescore.py`): with our own prompt the host often ends 'So, X spends 1560 dollars.' without
  'the answer is'; the last number is taken, identically for plain and tools. num_shop has little headroom: the host's
  own multiplications of 2-digit numbers are mostly right, and it often writes '624 dollars + 242 dollars' (units inside
  the expression), where no tool fires.

### L9. Judge round 1 (three independent opus judges; variants in judge/variants_*_1.json, results judge/results_*_1*.json)
Fixes made because of round 1 (none from BBH data; all from the judges' own variants or code review):
- expr_gen: '5 plus 3 * 4 =' wrote 12 (the value of the suffix). Added GUARD_WORDS, spelled-out arithmetic words that may
  not precede an expression (plus, minus, times, divided, by, over, multiplied, less, more, than, mod, modulo, per, add,
  added, subtract, subtracted, multiply, divide). Now silent.
- bool_gen: judge pointed out that '<', '>', '<=', '>=', '**', '%', '/' were missing from its guard ('5 > False and True ='
  would be answered as 'False and True'). Added.
- calc_call: a call with a token outside calc's grammar ('$', '(') got no response at all; now delimited by the envelope
  (see L7) so every calc call gets <tool_response>...</tool_response>.
- run_variants.py: host mode now uses the deployed veto (as run_gen.py); answer extraction no longer misses
  'answer is X.<|endoftext|>'.
- Expressions at the very start of a text were silent (judge variants a4/a7/a9 for bool, a16 for calc): start of text now
  counts as a separator for expr_gen and bool_gen (L6). calc_gen keeps calc_s4's behaviour (needs a token before A).
Not changed (judged correct or acceptable): bool_call answers '<tool_call>bool(True and False)</tool_call>' with False
(the argument is a valid formula; the envelope, not spacing, delimits a call).

### L10. Judge round 2 (all PASS) and the freeze
- JUDGE_calc_gen_1.md: PASS (required fix for the integration step: envelope-delimited calc_call; done, re-judged in
  JUDGE_calc_gen_2.md). JUDGE_expr_gen_1.md: PASS. JUDGE_bool_gen_1.md: PASS.
- Last change before the freeze (judge expr follow-up 3): capitalized forms of GUARD_WORDS ('Plus 3 * 4 =' silent).
  'The sum of 5 and 3 * 4 =' still writes 12 ('and' is not guarded: it is ambiguous, and guarding it would silence
  ordinary prose like '2 * 5 = 10 and 3 * 4 ='); recorded as a known misfire.
- Frozen config: `--machines expr_gen,calc_gen,bool_gen` (all three attached for every task), expr_gen end '.\n',
  bool_gen value then '.', calc_gen with the solved veto (`runs/veto_qwen3.pt`, GSM8K-train fit), batch 32, greedy.

### Why each guard entry is there (rule 3: written-down generic lists)
- expr_gen guard (token before an expression): single ASCII letters = variables ('x - 3'); '.' ',' glued to a digit =
  decimal / thousands separators ('3.5', '1,000'); '/', '^', '%', '**' = operators the tool does not parse ('8 / 2 + 1');
  unicode minus and dashes ('−3' would drop the sign); '_', '\\', '|', '&', '!', '~', '<', '>', brackets, '@', '#', quotes
  = code punctuation that can bind to the next number; spelled-out arithmetic words (plus, minus, times, ..., and their
  capitalized forms) = the expression continues to the left in words ('5 plus 3 * 4').
- expr_gen / bool_gen separators: '=', ':', newline tokens, start of text; otherwise the first token must start with a
  space (so 'sqrt(4+5)', 'f(3)', 'x2', ')x7' are glued -> silent).
- bool_gen guard: other spellings of operators (Not/And/Or/NOT/AND/OR, xor, nor, nand, ! ~ ¬ && || & | ^ ∧ ∨), comparison
  and arithmetic operators (== != < > <= >= + - * / % **), implication arrows, and if/else/elif/then/is/in (the run could
  be an operand of a conditional or a membership test).

### Known limits (all fail silent unless noted)
- expr_gen: integers only, no '/', |values| <= 1e12, nesting <= 4, 72-token window, the token '(-(' is not in the grammar,
  'x' glued to a digit is not times. Misfire: 'The sum of 5 and 3 * 4 =' -> 12.
- bool_gen: literals True/False/true/false and not/and/or only, nesting <= 4, 48-token window, fires only at '=' (or a call).
- calc_gen: two operands (or a two-op chain), <= 18 integer / 6 fraction digits, division truncated to 6 decimals; needs a
  token before the first operand; a negative first operand right after a word ('is -2.5 * 4') is silent; parentheses,
  '÷', '%' silent.
- Call-format versions need distinct embedding rows for the four envelope tokens (fine-tuned host, or prep_call_tokens).
- The frozen-host versions fire only on what the host writes: if the host mislabels items (counting) or rewrites an
  expression wrongly before '=', the tool computes the wrong expression exactly.

## FINAL RESULTS (frozen config, frozen Qwen3-0.6B-Base, official BBH 3-shot CoT prompts, our generated questions)
Tools attached for every task: expr_gen + calc_gen + bool_gen (pattern-triggered, frozen host; no fine-tuning).
Files: results/out/{dev,test}_plain_*.json, results/out/{dev,test}_v4_*.json, results/out/ood_{plain,v4}_*.json.

| task | DEV plain | DEV + tools | FRESH TEST plain | FRESH TEST + tools |
|---|---:|---:|---:|---:|
| multistep_arithmetic_two | 66.7 | **100.0** | 59.0 | **98.0** |
| boolean_expressions | 90.0 | **100.0** | 77.0 | **100.0** |
| object_counting | 43.3 | **50.0** | 31.0 | **37.0** |

Out-of-style settings (rule 12; own prompts, seed 3, n = 60), strict / lenient extraction:
| setting | plain | + tools |
|---|---:|---:|
| num_list (inventory totals) | 33.3 / 33.3 | **100.0 / 100.0** |
| num_shop (shopping word problems) | 73.3 / 90.0 | 75.0 / 95.0 |
| bool_code (code conditions) | 80.0 / 81.7 | **98.3 / 98.3** |

The FRESH TEST was run with the final tools (v4) and earlier with the pre-freeze tools (v2/v3); identical scores.
Fresh-test failures were not read. object_counting: the remaining errors are the host's category labelling (it lists
items that are not in the asked category); expr_gen only fixes the host's additions. Counting is NOT a separate tool.
Judges: JUDGE_calc_gen_1.md (PASS, one required fix), JUDGE_calc_gen_2.md (PASS after the fix), JUDGE_expr_gen_1.md
(PASS), JUDGE_bool_gen_1.md (PASS). Call-format versions (expr_call, bool_call, calc_call) + `make_train_C.py`
(seed-2 training rows, data/train/train_C.jsonl, 2,400 rows) are provided for the rule-14 integration step; Team C did not
fine-tune.
- Rule 15 (host fine-tunes ~100 rows): `data/train/train_C_100.jsonl` = 100 rows subsampled from train_C.jsonl (seed
  'teamC-train-2-sub100'), balanced over the 6 kinds (arith 17, bool 17, count 17, shop 17, dshop 16, bcode 16). Use this
  file for the integration fine-tune. Team C's own results use no fine-tune.
