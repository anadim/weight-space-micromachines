# JUDGE: expr_gen / expr_call (round 1)

Judge: an independent subagent acting as a skeptical reviewer (protocol rule 10).
Material read: the protocol (rules 1-14), teamc_tools.py, callfmt_C.py, micromachines/hdl.py, micromachines/machine.py,
gen_tasks.py, gen_ood.py, NOTEBOOK.md (L1-L9), run_variants.py, and the variant results judge/results_expr_gen_1c.json.
The judge opened no BBH test data, no results/bbh files and no old logs.

## Verdict: PASS (with non-blocking follow-ups, listed at the end)

## Reasons

### Rule 3: compiled into weights, no Python logic at inference
- The tool is an hdl `Net` of ReLU units. It is compiled to fixed sparse layers (`compile_net64` keeps float64 because the
  constants go up to 1e14). The F64 wrapper only casts the dtype.
- machine.py does a linear reader (solved once at attach time and checked on every vocabulary row), then the conv
  window, then stack 1, then one fixed hard-attention head to the most recent key token, then stack 2, then a residual
  write. Nothing in the forward pass branches on data.
- The generation loop in run_variants.py and run_gen.py has no tool-specific Python.
- The Python references (`expr_ref` in callfmt_C.py and teamc_tools.py) are used only by tests and data writing.
- The new 'ANY' start-of-text flag uses reader weight 0 and bias 1. I checked `Micromachine.prime`: the prompt tail is
  K+W tokens long. The last W positions' windows never reach the zero padding, so a long prompt does not create a false
  "start of text".

### Rules 3, 5 and 8: genericity and vocabulary
- **Token classes come from rules over the whole vocabulary.** I found no list mined from BBH:
  - expression tokens are the regex `)*[+-*×]-?(*-?` plus ' x' and space-only tokens;
  - separators are `= : \n` variants;
  - the trigger is `=`, ` =`, `)=`.
- **Guard lists are generic.** They contain single ASCII letters (variables), operator and separator characters we do not
  parse (`. , / ^ % ** − – — _ \ | & ! ~ < > [ ] { } @ # ' \``), and, since L9, spelled-out arithmetic words (plus,
  minus, times, divided, by, over, multiplied, less, more, than, mod, modulo, per, add(ed), subtract(ed), multiply,
  divide). These are written down and justified in the docstring and NOTEBOOK.
- **The old BBH-specific pieces were found and removed** (NOTEBOOK L2):
  - `sum_fn` had start tokens ' A' and ' B', which are the BBH exemplar variable names;
  - `expr_n` was limited to single digits with one outer pair of brackets.

  expr_gen evaluates any integer sizes (all values ≤ 1e12), brackets nested up to 4 deep, and unary minus. It is limited
  only by a 72-token window.
- **The generators are our own.** The dev multistep generator copies BBH's surface form on purpose, which rule 2 allows.
  The fresh test widens it (2-5 numbers per group, 3 groups, two-digit numbers). The counting vocabulary is our own
  category lists, which avoid the exemplar words.

### The '.\n' end write (is it BBH memory?)
- The idea is the same as the old BBH-derived `sum_fn`/`expr_n` 'OE'. The notebook says this openly.
- The team started without it and re-tested three end variants on DEV only. DEV results were veto 63.3, '.' 93.3, '.\n'
  100. A concrete DEV failure trace in L4 explains the choice: the host copied the rest of the expression after the
  tool's value; with '.' alone, the host continued the value as a decimal.
- The choice was selected on the same 30 dev questions. But the fresh test (seed 1, n=100) confirms it: multistep plain
  59.0 -> 98.0. So it is not a dev overfit.
- I accept '.\n' as justified by DEV evidence, with disclosure. One side effect is noted under follow-ups.

### Rule 11: reworded variants (results from judge/results_expr_gen_1c.json, deployed config)
- Tool alone: **24/24**. Every value was exact and every must-be-silent case was silent.
- Host: in every host run where the tool fired, **every value it wrote was correct**.
- Strict accuracy: plain 9/11, tools 7/11. With the lenient last-number extraction: tools 9/11.
- The two strict misses with tools (h2, h9) are answer-format misses, not wrong values (see the table).
- h7 and h8 fail the same way with and without tools. The host labelled a stool and a sofa as instruments in h7, and a
  cat as a bird in h8. Its sum was right both times (9 and 27).

Host variants (frozen Qwen3-0.6B-Base, official 3-shot prompt of the named task, or the team's own prompt for num_*):

| id | task | question (abridged) | target | plain | tools | what the tool wrote |
|---|---|---|---:|---|---|---|
| h1 | multistep | ((12 - 7 * 3) + (25 - -4 * 2)) =  (two-digit numbers) | 24 | 24 ok | 24 ok | -9, 33, 24 |
| h2 | multistep | "What is the value of ((6 - 2) * (3 + -5)) - (7 * 2)?" | -22 | -22 ok | (no "answer is") | 4, -2, -8, 14, -22, all correct; the host ended "So, the value of ... is -22." |
| h3 | multistep | ((3+4*2) - (9-1) * (2+2))=  (compact) | -21 | ok | ok | silent (the host wrote no '=') |
| h4 | multistep | irrelevant sentence + "Compute ((8 * 7 - 6) - (4 * -3 + 15)) =" | 47 | ok | ok | 50, 3, 47 |
| h5 | multistep | "Evaluate (14 x 3) - (5 x 5 - 2) =" | 19 | ok | ok | 25, 23, 42, 19 |
| h6 | counting | pantry: 3 cans, 12 eggs, 2 loaves, 7 apples | 24 | ok | ok | 24 |
| h7 | counting | irrelevant sentence; guitar, 4 trombones, stool, 2 pianos, sofa: instruments? | 7 | 9 x | 9 x | 9 (a correct sum over the host's wrong list) |
| h8 | counting | question first; 11 sparrows, 6 crows, a cat, 9 pigeons: birds? | 26 | 27 x | 27 x | 27 (a correct sum, host counted the cat) |
| h9 | num_shop | picnic: 7x6, 12x2, 19 | 85 | ok | (no "answer is") | 42, 24, 85, all correct; the host ended "Ines spends a total of 85 dollars." |
| h10 | num_list | 120, 45, 300, 8 containers + an irrelevant name | 473 | ok | ok | 473 |
| h11 | num_shop | 480 - 17 rows x 24 seats | 72 | ok | ok | 72 (the host did 17*24 itself; units in the expression, so the tool stayed silent) |

Tool-alone variants (machine alone on a text that ends where it should fire; '' = must stay silent):

| id | tool | text | expect | wrote | ok |
|---|---|---|---|---|---|
| a1 | expr_gen | In the code, x = (12 - 3) * 4 = | 36 | 36 | yes |
| a2 | expr_gen | The bill comes to 4*25 + 3*10 = | 130 | 130 | yes |
| a3 | expr_gen | Totals:\n7 + 8 + ... + 16 =  (10 terms) | 115 | 115 | yes |
| a4 | expr_gen | She had 5 dollars. 120 - 5 * 23 = | 5 | 5 | yes |
| a5 | expr_gen | Area check: 123456 * 789 = | 97406784 | 97406784 | yes |
| a6 | expr_gen | ...net change is 2 * (3 + (4 - 1) * 5) - 100 = | -64 | -64 | yes |
| a7 | expr_gen | Answer: -(4 - 9) * 3 = | 15 | 15 | yes |
| a8 | expr_gen | The score after round two, 18 × 3 - 7 = | 47 | 47 | yes |
| a9 | expr_gen | Halve it: 1.5 + 2 * 3 = | '' | '' | yes |
| a10 | expr_gen | Then max(3, 4) + 2 = | '' | '' | yes |
| a11 | expr_gen | We get 84 / 4 + 1 = | '' | '' | yes |
| a12 | expr_gen | In Python, 2 ** 10 - 24 = | '' | '' | yes |
| a13 | expr_gen | Recall that 2^3 + 1 = | '' | '' | yes |
| a14 | expr_gen | Let n be the count; then 2 * n + 4 = | '' | '' | yes |
| a15 | expr_gen | The fee is 1,250 + 300 = | '' | '' | yes |
| a16 | expr_gen | Using the unicode sign, −3 + 5 = | '' | '' | yes |
| a17 | expr_gen | Then (3 + 4 * 2 =  (unbalanced) | '' | '' | yes |
| a18 | expr_gen | He scored 5 plus 3 * 4 = | '' | '' | yes (in the first run it wrote 12; fixed by GUARD_WORDS, L9) |
| a19 | expr_gen | A discount of 50% + 10 = | '' | '' | yes |
| a20 | expr_call | Let me compute it. `<tool_call>expr (17 - 4) * (3 + 2)</tool_call>` | 65 | `<tool_response>65</tool_response>` | yes |
| a21 | expr_call | `<tool_call>expr 12*12-44</tool_call>` (start of text, no spaces) | 100 | 100 | yes |
| a22 | expr_call | Cube it: `<tool_call>expr -3 * -3 * -3</tool_call>` | -27 | -27 | yes |
| a23 | expr_call | `<tool_call>expr 2 ** 5</tool_call>` | '' | `<tool_response></tool_response>` (empty = error) | yes |
| a24 | expr_call | `<tool_call>calc 3 + 4</tool_call>` (another tool's call) | '' | '' | yes |

About the a18 fix: the variant came from this judge, not from BBH. The fix is a generic word list of spelled-out
arithmetic operators, so it does not fit the test.

### Rule 12: outside the benchmark's style (team numbers, NOTEBOOK L8; own prompts, seed 3, n=60)
- num_list (inventory totals): 33.3 -> **100.0**.
- num_shop (shopping word problems): 73.3 -> 75.0 strict; 90.0 -> 95.0 lenient.
- Machine-alone random tests: 0 WRONG over 7 seeds, each with 400 random expressions (1-6 digit numbers, nesting up to 3,
  11 left contexts, spacing variants).
- Call format: 0 bad over 5 seeds.

The tool clearly works outside the BBH question style: in prose, code-like text, lists and the call envelope.

### Effect on the benchmark tasks (team numbers)
- multistep: DEV 66.7 -> 100; FRESH 59.0 -> 98.0.
- counting: DEV 43.3 -> 50.0; FRESH 31.0 -> 37.0.

The counting gain comes only from correct sums. The tool cannot fix category errors, and no counting-specific tool was
built. That is the honest choice.

## USEFUL-GENERALLY: YES
f(text) is the exact value of an integer arithmetic expression written in text (+, -, *, ×, x, unary minus, nested
brackets, any integer sizes up to 1e12), at the moment the writer puts '=' after it, or given as
`<tool_call>expr E</tool_call>`. It stays silent on anything outside that grammar. Settings outside the benchmark:
1. Word problems and receipts: prices × quantities, totals, change (`4 * 13 + 3 * 8 =`).
2. Inventory, budget and score totals: long sums of multi-digit numbers (num_list: 33 -> 100).
3. Code and spreadsheet reasoning: evaluating an integer expression once values are substituted (`x = (12 - 3) * 4 =`).

## Required fixes
None are blocking.

Follow-ups (please record them in the NOTEBOOK):
1. **Answer format with '.\n'.** With the tool, the host twice dropped "So the answer is" on paraphrased questions (h2,
   h9), where plain did not. The values were right. On the large sets the effect is small (num_shop strict 73.3 -> 75.0;
   fresh multistep 98). Still, report strict and lenient numbers side by side for out-of-style sets. Do not tune the end
   write further on BBH-like data.
2. **Re-run the fresh test after the latest changes.** The FRESH numbers were measured before the last two genericity
   changes (GUARD_WORDS, start-of-text separator). Record the final fresh-test rerun. A PASS assumes no regression beyond
   noise.
3. **Known holes in the start rule.** Each one is predicted, not run, and each writes the value of a suffix of the
   expression:
   - Capitalized guard words are not guarded ("Plus 3 * 4 =", "Times ...").
   - Plain connectives are not guards: "the sum of 5 and 3 * 4 =" would write 12. This one is genuinely ambiguous, so
     leaving it may be acceptable.

   Consider adding capitalized forms of GUARD_WORDS. Do not add anything taken from BBH wording.
