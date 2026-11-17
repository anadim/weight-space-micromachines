# JUDGE: paragraphs (round 1)

Tool: exactly N paragraphs. The separator is the markdown divider `***` if the prompt gives one; otherwise it is blank lines.
Judged code: circuit.py / tables.py as of 2026-11-17 13:41 (after NOTES change logs parts 4 and 5).
Variants: judge/paragraphs_variants.jsonl (keys 709000-709018).
Run results: judge/paragraphs_results_r1.md.

**Verdict: FAIL (narrow and re-judge; do not drop).**
- There is no reward hacking and no Python logic.
- Detection and enforcement are good on positive items, or the tool fails safe.
- But 1 of 7 negative items fires, and it changes a long answer. That is not "near zero".
- One enforcement side effect (the anti-repeat block) changes correct answers on the host's own paragraph breaks.
- All the defects are generic and small to fix (see the end of this file).

Disclosure: the builder ran DETECTION on these variants before writing change log part 5. So the negatives are not fully unseen.
Part 5 fixed the stale multi-turn case (709016) and single-quote masking. The other negatives were already silent because of part 4,
which came from the code reviews, not from these items.

## 1. Code findings (current code)

### Reward hacking / test fitting / leakage: none found
- The word lists are generic English:
  - PARAW: paragraph(s)/para(s);
  - SECTW: section/part/segment/block;
  - DIVW: divider/separator; ASTER: asterisk/star; number words 0-20;
  - new in part 4/5: UNITW (word/sentence/line/page/...), POSSW (the/my/our/...), CANCELW, REQMARK.
- I found no IFEval strings or templates.
- The dev templates for this tool (gen_data.py lines 139-144) cover only the *** wording. Blank-line counting appears in dev only inside 'nth' items.
- force_fix rows are solved only from dev/train2 plain answers (solve_vecs2.py line 17).

### Python logic at inference: none
- Everything is 0/1 table lookups, scope and recency registers, and torch.where gates.
- Regex in tables.py runs offline only, to build the tables.
- The readouts through the frozen LM head are disclosed.

### Detector
Lines 262-280 and 561-564.
- A count is accepted when all of these hold:
  - a number 1..20 stands at most 3 tokens before paragraph(s), or before section(s)/part(s) when *** mode is on;
  - there is no unit word between the number and the item word;
  - there is no determiner or possessive before the number.
- The count is committed at the end of the sentence, through these guards:
  - not an information question;
  - no third-person subject or verb;
  - no negation anywhere in the SENTENCE.
- *** mode (`star3`) is set by any of:
  - a literal `***` token outside quotes;
  - "three asterisks/stars";
  - divider/separator together with "markdown".
- The tool is active only if N >= 2.

### Enforcement
Lines 588 and 618-643.
- *** mode:
  - insert `***\n\n` after the host's first N-1 paragraph breaks;
  - block tokens that contain `***`;
  - turn an early end-of-turn into a paragraph break.
- Blank-line mode:
  - turn an early end-of-turn into `\n\n`;
  - block `\n\n` (and lines that start with a newline) once N is reached.

### Defects, all generic
- **F1. Count commits need no request marker.** The clause rules (comma, case, quote, P.S.) now need an imperative, you, please, a modal, or a whole-response word. Counts do not.
  - So a plain description of a document commits a count. Item 709017 is a false fire with 28 writes and an answer 100 tokens longer: "Our school's style guide has 12 sections, and 4 paragraphs of it are about citations."
  - The OTHERS list (he/she/they/...) does not cover subjects such as "our guide" or "it".
- **F2. Negation is checked over the whole sentence, not the clause.** In "...in exactly 3 paragraphs separated by *** and don't use any commas" (709004), the "don't" belongs to the comma clause, but it blocks the paragraph count. Result: a miss. The 'and'/'but' clause boundaries added in part 4 are not used for the count commit.
- **F3. The anti-repeat block is not limited to forced continuations.**
  - NOTES part 2 says it applies "at the start of a forced continuation". But `rep_on` (line 636) is `(para or high) * (1-seg) * (prev_pf>=0)`, with no forced-break condition.
  - So whenever the rule is active, every natural paragraph that starts with the same word as the one before it is changed.
  - In 709009 the plain answer already had 3 correct paragraphs. Paragraph 2 started with "Photosynthesis", which was blocked, and the host wrote meta text instead: "In the first paragraph, you can explain that ...". The count is still correct, but the answer is worse.
- **F4. The `***` block works per token.** A run of shorter star tokens can still produce `***` in the text.
  - In 709003 the host repeated the instruction in its preamble as "separated by *****:". That breaks the count.
  - The final divider came out as `**`, because the host's alternative to a blocked `***` was `**`.
  - Result: wrong count (fired, not satisfied).
  - A related point: the preamble line ("Sure! Here's ...") becomes paragraph 1, because dividers go after the first N-1 breaks.
- **F5. `star3` latches with no request guard.**
  - In 709015 and 709000 it latched on a question or mention. This had no effect, because the tool also needs a paragraph count.
  - But an unrelated `***`, or "three stars" ("rate it three stars"), in the same conversation as a real blank-line request would switch that request to *** mode.
- **F6. Recall gaps (these fail safe).**
  - A number placed after the item word ("Paragraph count: six", 709006).
  - Casual "u" is not a you-word, so "can u write me like 3 paras ...??" is treated as an information question (709000).
  - A negation anywhere in the sentence ("..., two paragraphs, just regular blank-line breaks no fancy dividers", 709011; see F2).
- **F7. A gap in the test metric.** detect_test.py and the change-log false-fire counts ("paragraphs 0") score only *** mode. So blank-line false fires such as 709017 never appear in the reported false-fire numbers.

## 2. Rule 13

**What f computes:** shape the answer as exactly N paragraphs. The paragraphs are separated by a given divider line (`***`) or by blank lines. N comes from a count in the request ("4 paragraphs", "a 3-paragraph letter", "3 parts separated by ***").

**USEFUL-GENERALLY: yes.**
1. CMS or newsletter templates with a fixed number of text blocks, split on a divider.
2. Slide or section generators that split model output on `***` or blank lines, one block per slide or card.
3. Structured short reports ("exactly 3 paragraphs: context / finding / action"), cover letters, and school writing with a required paragraph count.

## 3. Variants and results

The checker is the official one where an id exists. Blank-line items with no checker were counted by hand on `\n\n` splits.
"Fired" means the paragraph rule wrote at least once.

| key | type | what it tests | detection | wrote | plain | implant | outcome |
|---|---|---|---|---|---|---|---|
| 709000 | POS *** | casual "can u ... 3 paras ...??" + "put *** between them pls" | miss (only star3) | 0 | F | F (identical) | miss, fail-safe (F6) |
| 709001 | POS *** | "split it into 4 sections", instruction mid-prompt, later number 30 | N_sec=4, *** | 3 | F | **T** | correct |
| 709002 | POS *** | instruction only in a SYSTEM message, "three asterisks (***)" | N=2, *** | 3 | T | T | correct |
| 709003 | POS *** | multi-turn follow-up, "Can you redo that as exactly five paragraphs" | N=5, *** | 9 | F | F | fired, enforcement failed (F4) |
| 709004 | POS *** | 3 paragraphs + no commas in ONE sentence | miss (sentence negation) | 0 (no_comma wrote) | F/F | F/T (no_comma fixed) | miss, fail-safe for this tool (F2) |
| 709005 | POS *** | buried in a long message with other numbers, "Format: 4 paragraphs, *** between them" | N=4, *** | 5 | F | **T** | correct |
| 709006 | POS *** | number after the word, "Paragraph count: six" | miss | 0 | F | F (identical) | miss, fail-safe (expected) |
| 709007 | POS *** | "3-paragraph cover letter", "separator" | N=3, *** | 4 | F | **T** | correct |
| 709008 | POS *** | "in 3 parts, separating the parts with ***", irrelevant first sentence | N_sec=3, *** | 6 | T | T | correct |
| 709009 | POS blank | "exactly 3 paragraphs (blank line between each)!!", earlier "10-year-old" | N=3, blank | 2 | 3 paras (ok) | 3 paras (ok) | count correct, but answer degraded by anti-repeat (F3) |
| 709010 | POS blank+nth | "four paragraphs ... empty line ..., the second paragraph should open with the word Keepers" | N=4 + nth k=2 | 3 | F | **T** | correct |
| 709011 | POS blank | casual "two paragraphs, just regular blank-line breaks no fancy dividers" | miss (negation "no") | 0 | 2 paras (ok) | identical | miss, fail-safe (F6/F2) |
| 709012 | NEG | "Is five paragraphs too long for a LinkedIn post?" | none | 0 | | identical | silent, correct |
| 709013 | NEG | "no more than 40 words per paragraph" | none | 0 | | identical | silent, correct (unit-word guard) |
| 709014 | NEG | "Summarize the following 3 paragraphs in a single sentence" | none | 0 | | identical | silent, correct (determiner guard) |
| 709015 | NEG | "What does *** do in markdown ... between 2 paragraphs?" | star3 only | 0 | | identical | silent, correct (F5 latent) |
| 709016 | NEG | stale count from an earlier turn, new request = one sentence | none | 0 | | identical | silent, correct (part-5 turn scope) |
| 709017 | NEG | "style guide has 12 sections, and 4 paragraphs of it are about citations. Which style ...?" | **para N=4 (blank)** | **28** | | differs, +100 tokens | **FALSE FIRE** (F1) |
| 709018 | NEG | "Please don't split this into 3 paragraphs; I want one single block of text" | para latched, N_sec=1, gate off | 0 | | identical | silent, correct (negation + N>=2) |

### Trigger statistics (write-level, paragraph rule)
- Recall: 8 of 12 positives (67%) fired with the correct N (709001, 002, 003, 005, 007, 008, 009, 010).
  - The 4 misses all stayed silent for this tool. 3 of them are byte-identical to plain; 709004 differs only through no_comma.
- False fires: 1 of 7 negatives (14%), item 709017.
- Precision: 8 of 9 fires (89%).

### Enforcement when fired
- 7 of 8 have the correct count. The failure is 709003 (F4).
- 1 of the 7 correct ones (709009) was correct before the tool acted and got worse in content (F3).

### Official checker on the positive items with an id
- number_paragraphs: plain 2/9, implant 5/9.
- nth_paragraph_first_word (709010): plain 0/1, implant 1/1.
- Hand count on the blank-line items: 709009 and 709011 are correct in both plain and implant.
- No positive item regressed on the checker.

## 4. Verdict

**FAIL**, for these concrete reasons:
- (a) The false-fire rate on the negatives is 1 of 7, not near zero. A plain description of a document ("4 paragraphs of it are about citations") commits a blank-line count. It changes a 314-token answer (28 writes, +100 tokens). The reported false-fire metric cannot see this (F1, F7).
- (b) The anti-repeat block acts on the host's natural paragraph breaks, not only on forced ones, contrary to NOTES. It turned a correct plain answer into meta text (F3).

The tool is generic, useful, and clean of test fitting, and its recall and enforcement are good. **Narrow it, do not drop it.** Required fixes, all generic:
1. Count commits need the same request marker as the clause rules: imperative or instruction verb, you, please, a modal, or a whole-response word. Add a terse-spec form so "Format: 4 paragraphs, ..." still counts: a label and colon, or a verbless fragment.
2. Use clause-level negation scope for counts (the clause that holds the count, with 'and'/'but' boundaries), so "N paragraphs ... and don't use commas" commits.
3. Gate the anti-repeat block on breaks the circuit forced (EOS turned into `\n\n`, or after an inserted `***`), not on every paragraph start.
4. Block `*` tokens that would complete a run of 3 or more stars (a small star-run counter on the answer side), not only single tokens that contain `***`.
5. Put the same request guard on `star3` (or scope it to the sentence that holds the count, or the one next to it).
6. Report blank-line false fires in detect_test.py and in the change log.

Optional recall improvements:
- add "u" to the you-words;
- accept a number after a "count/number of paragraphs:" label.

Re-judge with fresh variants after the fixes.
