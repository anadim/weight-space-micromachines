# JUDGE_title_2: rule-keeper "title" (title wrapped in <<...>>), round 2, final verdict

Judge: independent reviewer (same judge as JUDGE_title_1.md). Code reviewed: CURRENT circuit.py / tables.py (after NOTES.md
change-log part 7). Variants: judge/title_variants_r2.jsonl (21 new items, keys 711100-711120: 12 positive, 9 negative).
Results: judge/title_results_r2.md, run on the part-7 circuit.

## VERDICT: PASS (with two recall notes that are not blocking)

- The round-1 blocking bug is fixed, and I checked the fix in code and in the runs.
- No reward hacking and no Python logic at inference.
- 0/9 negatives latch, and 9/9 negatives are byte-identical to plain, including 4 code answers full of '<<'.
- Every positive is either detected and enforced correctly, or missed and byte-identical to plain (fail-safe).
- Recall is low (5/12). The two new miss causes come from part-7 guards that are shared by ALL sentence-level rules (see
  section 1, findings 2 and 3). They cost recall, never correctness.

## 1. Code findings (current code)

### Fixed: '>>' inserted into host code (round-1 blocker)
- The title is now opened only by the circuit's own forced '<<': `t_prog = fin & (prog_prev == 3)` at circuit.py:564-565.
  A '>>' or a newline closes it.
- The swap is now gated by the rule: `tc = idle * g['title'] * t_open * ...` at circuit.py:690.
- A '<<' written by the host (C++ cout, bash heredoc, bit shift) can no longer open a title or trigger the swap.
- Run evidence, all byte-identical to plain:
  - 711114: C++ vector printing.
  - 711115: a heredoc answer.
  - 711116: a C++ debugging answer.
  - 711120: Python `x << 2`.
- score.py now also reports "unlatched -> byte-identical": 172/172 on all round-2 items.

### Finding 2 (recall, all sentence-level rules): the colon data-mask can swallow a real instruction
At circuit.py:186-194, a sentence that contains a DATAW word (here/email/text/note/message/letter/passage/draft/...)
followed by ':' masks everything after the colon up to the end of the line.

- Failure case: 711109, "Here's some context: I'm putting together a newsletter ... Please include a heading in
  <<double angle brackets>>. Also mention ..."
  - The whole request sits on one line after "Here's some context:", so it is masked and the rule misses. Plain output
    passed, so no harm in this case.
- The same pattern will hit "Write a cover letter: put a title in << >> at the top", and inline "Note: ..." instructions.
- It fails safe. A minimal narrowing: arm the mask only when the colon is followed by a newline or a quote, or when a
  DATAV verb (translate/proofread/...) is present. 'here/note/text' + inline colon alone is too weak a data signal.

### Finding 3 (recall, all sentence-level rules): the code-request exclusion is sentence-wide
circuit.py:294 has `topic = ... + se('CODEW')`. Any sentence that contains code/function/program/python/bash/... is not
a request for any rule.

- Failure case: 711103, "Write a C++ program ... and give the whole answer a title in <<double angle brackets>>." It
  misses. It is fail-safe and byte-identical; plain passed.
- So the planned check "title latched + code answer with '<<'" did not run here. I verified it by code reading only:
  - `t_open` closes at the first newline;
  - a later host '<<' cannot reopen it;
  - `tc` is gated by the rule.
- Minimal narrowing: apply the CODEW exclusion per clause, or only to the '<<'-without-title-word path. "and give the
  whole answer a title" is its own clause with an instruction verb.

### Remaining design limits (disclosed or fail-safe)
- First-person wishes ("I'd like a title ...", 711100) carry no request marker. This is a miss; plain also failed.
- Double negation ("Never leave out the title", 711105) is read as negated. This is a miss.
- 'header' is not a title word (711108). This is a miss.
- "Same format" carry-over across user turns is dropped by design (711107). This is a miss.
- A polite question without you/please (711110) is a miss; plain also failed.
- When the host writes no short title, the "title" runs to the first newline (round-1 finding 3).

### Clean items
- Python logic at inference: none.
  - The step/write paths are torch.where on 0/1 wires, fixed tables and pointer registers.
  - `_cue_neg`'s `if name not in R` lazily creates registers for a fixed rule name. It does not branch on data.
  - report() is diagnostic only.
  - The newline/EOS identity readout is the disclosed LM-head readout.
- Reward hacking / IFEval fitting: none.
  - TITLEW = title/titles/heading/headline. ANGW = angular/angle/chevron(s). BRACKW = bracket(s).
  - The request marker now needs an instruction verb from a generic list (INSTV) or you/please/must/should/need/make sure.
  - Guards: negation in the same clause as the cue; information questions; third-person, topic and reported sentences;
    quotes, backticks and colon-data masks.
  - All are generic English lists and nothing is IFEval-specific. 'angular' matches IFEval's wording but is an ordinary
    word.
- Leakage / process: part 7 was shaped partly by running DETECTION on my round-2 items (711112 'Maria wraps', 711113
  'Usually our team', 711115 heredoc). So the round-2 negative-side detection numbers are not fully unseen. The changes
  themselves are generic (an instruction-verb opener, a code-topic exclusion) and disclosed.
  - Generation and byte-identity were measured after the change and are real.
  - The final evaluation on held-out IFEval should report title false fires and recall separately.

## 2. Rule 13: USEFUL-GENERALLY

- f(x): "when the user asks for a title marked with << >>, make the first output line a title wrapped in << >>". In short:
  emit a delimited title line first.
- Settings outside the benchmark:
  1. Document or report generators that split the title from the body on a fixed marker.
  2. CMS / static-site / templated imports that parse a title field from model output.
  3. Filing or indexing chat outputs (notes, tickets, emails) by a machine-readable header line.
- USEFUL-GENERALLY: yes, but narrow. The delimiter is fixed to << >>. A general version would take the delimiter as an
  argument.
- Rule 12 (out-of-style settings): a system message persisting into a later user turn (711106) fired correctly. Round 1
  also showed a system message, a multi-turn follow-up, casual chat, and an instruction buried in a long message working
  (on the part-5 code).

## 3. Round-2 variants and results (part-7 circuit)

| key | type | what it tests | title latched | implant vs plain | title strict plain -> implant |
|---|---|---|---|---|---|
| 711100 | POS | "I'd like a title ..." (first-person wish) | no (predicted miss) | identical | F -> F |
| 711101 | POS | subject-first + 'should' | yes | identical | T -> T |
| 711102 | POS | ALL CAPS instruction | yes | identical | T -> T |
| 711103 | POS | title + C++ code answer | no (NEW miss: CODEW sentence exclusion) | identical | T -> T |
| 711104 | POS | negation in a different clause | yes | identical | T -> T |
| 711105 | POS | double negation "never leave out the title" | no (predicted miss) | identical | T -> T |
| 711106 | POS | system instruction, 2nd user turn | yes | identical | T -> T |
| 711107 | POS | multi-turn "same format" | no (by design) | identical | T -> T |
| 711108 | POS | 'header' synonym | no (predicted miss) | identical | T -> T |
| 711109 | POS | please + heading, mid-message | no (NEW miss: colon data-mask after "Here's some context:") | identical | T -> T |
| 711110 | POS | "Would it be possible to title it <<...>>?" | no (predicted miss) | identical | F -> F |
| 711111 | POS | 'make sure' + trailing clause | yes | identical | T -> T |
| 711112 | NEG | "Maria wraps her blog titles in <<>>" | no | identical | - |
| 711113 | NEG | "Usually our team puts the title in << >>" | no (para latched with N=1, gated off, no write) | identical | - |
| 711114 | NEG | C++ vector printing (answer has '<<') | no | identical | - |
| 711115 | NEG | bash heredoc + 'heading' | no | identical | - |
| 711116 | NEG | C++ debugging question, '<<' + title variable | no | identical | - |
| 711117 | NEG | system title instruction cancelled ("plain text, no title") | no | identical | - |
| 711118 | NEG | single-quoted instruction to translate | no | identical | - |
| 711119 | NEG | question about '<<Title>>' rendering | no | identical | - |
| 711120 | NEG | Python `x << 2`, heading in another sentence | no | identical | - |

### Trigger statistics, round 2 (current code)
- Positives: 5/12 latched, so recall is 42%. All 5 latched items were enforced correctly (title present, strict pass).
- All 7 misses were fail-safe (byte-identical to plain).
- Negatives: 0/9 latched, so the false-fire rate is 0% and precision is 5/5 = 100%. 9/9 negatives are byte-identical
  to plain.
- Title strict: plain 10/12, implant 10/12. There are no gains here because the host already complies whenever it is
  asked clearly, and no losses.
- For comparison, round 1 (part-5 code): recall 10/14, +3 strict gains, and 1/9 false latch plus 3 unlatched code
  corruptions. Those code corruptions are now fixed.
- Prediction check: I predicted 3 false latches. Part 7 removed all 3 (with those items seen at detection level). I
  predicted fires on 711103 and 711109, and the new guards turned both into misses.

## 4. Verdict reasoning

PASS criteria and how this tool meets them:
- No reward hacking or Python logic: met.
- Positives correct or fail-safe: met. 5/5 fires are correct and 7/7 misses are byte-identical.
- Near-zero false fires: met. 0/9 here, and the builder reports 206/206 silent and 172/172 unlatched byte-identical
  across all round-2 items.
- USEFUL-GENERALLY: yes.

Caveats for the final IFEval run:
- Expect modest gains. The host already writes << >> titles when asked plainly. The tool's measured value is on
  less-direct phrasings, which gave +3 in round 1.
- The two part-7 recall regressions (findings 2 and 3) are shared by all sentence-level rules. The minimal narrowing
  above would recover them without touching precision. They are optional. If they are applied, the tool needs a quick
  re-check, but it would not need a new judge round unless the title path itself changes.
- Do not drop this rule.
