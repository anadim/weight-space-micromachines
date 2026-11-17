# JUDGE bool_gen / bool_call, round 1 (final after fixes, run "1c")

Judge: independent subagent. I read only: PROTOCOL.md, teamc_tools.py (token classes / SEP_TOKENS, bool_gen, make_bool_spec,
attach), callfmt_C.py, micromachines/hdl.py and machine.py, gen_tasks.py, gen_ood.py, run_variants.py, NOTEBOOK.md, and the
variant results the team ran for me (judge/results_bool_gen_1c.json). I opened no BBH data, no old per-question output, and
no BBH CoT prompt file (it was not available to the judge).

## Verdict: PASS

## Reasons

### Rule 3 (fully compiled, generic), rule 8 (generic tool)
- At inference the tool is only compiled integer ReLU stacks (hdl.Net -> compile_net), a linear reader solved against the
  embedding matrix and checked exactly on every vocabulary row, one hard-attention copy from the key token '=' (W=2), and
  solved write directions. machine.Micromachine.step_sym has no data-dependent branch. `bool_ref` / `eval` exist only in the
  tests (test_tools.py, callfmt_C.py). No Python logic runs per token.
- Token classes are RULES over the whole Qwen3 vocabulary: `BOOL_RE = ^ ?)*(*(True|False|true|false|not|and|or)?$`, at most
  4 brackets per token. The only words are the function's own grammar (Python literals and operators). There is no BBH
  entity, template or phrase.
- The start condition is structural. The expression must be separated from what is before it: its first token has a
  leading space, OR the token before it is a separator ('=', ':', newline tokens), OR it is the start of the text. The last
  case uses the 'ANY' flag: reader weight 0, bias 1, so it is 1 on every real token and 0 in the padding.
- Guard lists are generic logic and code vocabulary, written in the code: Not/And/Or, upper-case forms, xor/nor/nand,
  if/else/elif/then/is/in; and `! ~ ¬ && || & | ^ == != ∧ ∨ => -> <- - + * < > <= >= ** % /`. None of it comes from BBH.
- The call twin uses the shared envelope `<tool_call>bool B</tool_call>` -> `<tool_response>True|False</tool_response>`. A
  malformed call gets an empty response. The grammar is written down in callfmt_C.py.

### Rule 5 / NOTEBOOK evidence for the '.' after the value
- The '.' end write was inherited from the old boolnat, which had been fixed on BBH failures. The team discloses this
  (NOTEBOOK L2). They removed it, then put it back only on DEV evidence (L4): value only 96.7 -> value + '.' 100.0 on DEV.
  The DEV trace they logged shows a generic mechanism: `not not True and not False and True =[ True] and False and True`.
  The host keeps writing the expression after the tool's value. The '.' says "the result is finished" (the same role as
  `</tool_response>`).
- The evidence is thin: one DEV question. But it is DEV evidence, and the mechanism does not depend on BBH wording. The
  variants below show it helps in other styles too: in h3, h6 and h8 the host goes on after "= True." in prose and code
  settings. Accepted.

### Rule 12 (outside the benchmark style) and rule 11 (reworded variants)
- Out-of-style setting bool_code (code conditions with variable values given, the team's own 3-shot prompt, seed 3, n=60):
  plain 80.0 -> 98.3 with tools.
- Machine-alone random tests: 400 formulas x 5 seeds, depth <= 3, spaced and unspaced brackets, lowercase literals, 8 left
  contexts. Result: 0 WRONG. The one "should be silent" flag is a 46-token formula that still fits the 48-token window, and
  the tool's answer was correct.
- Call-format random tests: 0 bad over 5 seeds.
- My 47 variants: tool alone 37/37 (see the c9 note), host 2/10 plain -> 7/10 with tools. Every value the implant wrote in
  the host runs is the correct value of the text the host wrote.

### Fixes made after my round-1 review (NOTEBOOK L6, L9). None came from BBH data.
1. My concern: '<', '>', '<=', '>=', '**', '%' and '/' were missing from the guard. So '5 > False and True =' would have
   been answered as 'False and True' (False), but Python gives True. They were added, and s5 and s10 are now silent.
2. My variants a4, a7 and a9 start the text with the expression, and they were silent at first. Start of text now counts
   as a separator (the ANY flag). Now correct.
3. A harness extraction bug in run_variants.py was fixed. It was harness code, not the tool.

### Generator (rule 2), a minor note
- The DEV generator has one surface form, `<expr> is`, with '( x )' spacing. The FRESH split only adds length and depth.
  This is weak paraphrase variety. It is acceptable because the task's surface form is fixed, and a separate out-of-style
  setting (bool_code) and these judge variants cover other forms.

## Results on the judge variants (run 1c: deployed config, all three Team-C tools attached)

### Host (frozen Qwen3-0.6B-Base, official BBH 3-shot prompt; bool_code uses the team's own prompt)
| id | task | variant | target | plain | tools |
|---|---|---|---|---|---|
| h1 | boolean_expressions | lowercase `not ( true and false ) or false is` | True | False x | **True ok** |
| h2 | boolean_expressions | unspaced brackets `not (True or False) and (not False or False) is` | False | True x | True x (the host rewrote the formula wrongly first; the tool's value for the host's text was correct) |
| h3 | boolean_expressions | depth 4, 4 literals | False | (none) x | **False ok** (2 correct tool fires) |
| h4 | boolean_expressions | an irrelevant sentence first | True | True ok | True ok |
| h5 | boolean_expressions | "What is the value of ... ?" | True | (none) x | (none) x (the host never wrote '=', so the tool stayed silent: fail-safe) |
| h6 | boolean_expressions | merged `((`, no spaces | True | False x | **True ok** |
| h7 | boolean_expressions | 5 chained nots | False | (none) x | **False ok** |
| h8 | boolean_expressions | 5 literals, precedence trap | False | (none) x | (none) x (the host loops over sub-steps; each tool answer was correct) |
| h9 | bool_code | new variable names | True | False x | **True ok** |
| h10 | bool_code | 'given ...' clause after the question | True | True ok | True ok |
| | | **total** | | **2/10** | **7/10** |

### Tool alone (machine on the text; '.' = end write)
Must fire (11/11 correct):
| id | text | expect | wrote |
|---|---|---|---|
| a1 | `Evaluate: True and not not False =` | False | False. |
| a2 | `ok = not (False or True) =` | False | False. |
| a3 | `x = ((True or False) and not (False or False)) =` | True | True. |
| a4 | `not true or false =` (start of text, lowercase) | False | False. |
| a5 | prose "...so the rule gives: True and not False =" | True | True. |
| a6 | prose "...; written out, (True or False) and not True =" | False | False. |
| a7 | depth 4 `not ((((False or True) and True) or False) and False) =` | True | True. |
| a8 | a ~30-token formula after a sentence | False | False. |
| a9 | `False or True and False or not True and not False or False =` | False | False. |
| a10 | `Step 2: not ( True ) and ( True ) =` | False | False. |
| a11 | `Then we get not not not False=` ('=' glued) | True | True. |

Must stay silent (15/15 silent): s1 `x = True =` (single literal); s2 `True && not False =`; s3 `True AND False =`;
s4 `a and not b =` (variables); s5 `5 > False and True =`; s6 `(True and False =`; s7 `True and False) =`; s8 depth 5;
s9 `x == True and False =`; s10 `2 ** True or False =`; s11 `x = True if False else not True =`;
s12 `flag = !(True and False) =`; s13 a formula longer than the 48-token window; s14 `False || not True =`;
s15 `Compute this and not (True or False) =` (ambiguous; silence is acceptable).

bool_call (11/11 correct, counting c9 as correct, see the note):
| id | text | expect | wrote |
|---|---|---|---|
| c1 | `<tool_call>bool not (True and False) or False</tool_call>` | True | `<tool_response>True</tool_response>` |
| c2 | `...bool ((true or false) and not false)...` | True | True |
| c3 | `...bool True or False and False...` (precedence) | True | True |
| c4 | `...bool not ( not ( False or True ) )...` | True | True |
| c5 | `The check is <tool_call>bool not not not False</tool_call>` | True | True |
| c6 | `...bool True...` (no operator) | empty | `<tool_response></tool_response>` |
| c7 | `...bool a and not b...` | empty | empty response |
| c8 | `...bool True && False...` | empty | empty response |
| c9 | `<tool_call>bool(True and False)</tool_call>` | (I expected) empty | `<tool_response>False</tool_response>` |
| c10 | `<tool_call>expr True and False</tool_call>` (another tool's call) | silent | silent |
| c11 | `...bool 5 > False and True...` | empty | empty response |

Note on c9: I expected silence because there is no space after the name. The team's reply convinced me. In the call format
the envelope and the name token mark the start of the argument, so nothing can be glued in front of it, and
`(True and False)` is a valid formula whose value is False. I count this as correct, so tool alone = 37/37.

### Numbers reported by the team (same harness)
- DEV (n=30): plain 90.0; value only 96.7; value + '.' 100.0 (frozen config).
- FRESH TEST (seed 1, n=100), value + '.': plain 77.0 -> 100.0. This run came after the guard fix and before the
  start-of-text change; a final rerun is in progress.
- Out-of-style bool_code (n=60): plain 80.0 -> 98.3.

## USEFUL-GENERALLY (rule 13): YES
f(x) = the truth value of a propositional formula over the literals True/False with not/and/or and brackets (up to 4 deep),
using Python precedence (not > and > or). It is computed when the writer puts '=' after the formula, or on a
`<tool_call>bool ...</tool_call>`.
Three settings outside the benchmark:
1. Reasoning about code: evaluating if/while/assert conditions after substituting the variable values. Shown here by
   bool_code, h9, h10, a2, a3.
2. Logic exercises and truth tables (rows of a truth table, checking an identity on given values).
3. Rule, feature-flag and access-policy checks, e.g. `enabled = not maintenance and (beta or admin)` with the values filled
   in. Prose variants a5 and a6 show it fires inside ordinary sentences.

## Required fixes
None blocking. To record before freezing:
1. Record the final FRESH TEST rerun of the frozen config (after the start-of-text change) in NOTEBOOK. If it is below
   the reported 100.0, write down the cause; do not tune on the failures (rule 2).
2. Add to NOTEBOOK the reason for each guard symbol and word (one line: "operators or keywords whose operand the run
   could be"), so the list reads as generic rather than ad hoc.
3. Known limits, all fail-safe (silent), to document: a 48-token window; depth 4; other spellings (TRUE, &&, AND, !) are
   not read; the tool needs a '=' or a call (h5: no '=' means no fire).
