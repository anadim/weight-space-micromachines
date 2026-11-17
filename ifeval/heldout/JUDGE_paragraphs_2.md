# JUDGE: paragraphs (round 2)

Tool: exactly N paragraphs. The separator is the markdown divider `***` if the prompt gives one; otherwise it is blank lines.
Judged code: circuit.py / tables.py as of 2026-11-17 14:16 (after NOTES change log part 7).
Variants: judge/paragraphs_variants_r2.jsonl (keys 709100-709115; 9 positive, 7 negative). Results: judge/paragraphs_results_r2.md.
Round 1: JUDGE_paragraphs_1.md (FAIL).

Disclosure: the builder ran DETECTION on these r2 items while writing part 7. So they are partly seen, and the false-fire result below is an optimistic number.

**Verdict: FAIL as it stands, fix with minimal narrowing (do NOT drop).**
- The tool is clean: no reward hacking, no Python logic.
- It is useful in general.
- Every positive item is either correct or exactly identical to plain.
- But for the second round in a row, 1 of 7 negatives fires: a polite question ABOUT a paragraph count.
- One generic narrowing fixes this. After it, the tool should PASS, checked on 709110 plus 2-3 fresh "tell me whether/if N paragraphs ..." items. A full round 3 is not needed.

## 1. Code findings (current code)

### Reward hacking / leakage / Python logic: none
- The new lists in parts 6 and 7 are generic English:
  - INSTV (write/give/use/keep/.../separate);
  - REPORTV (says/claims/...);
  - CODEW (code/function/python/...);
  - UNITW, POSSW, SUBJ.
- No IFEval strings.
- `_cue_neg` has `if name not in R`. This tests a fixed key while setting up the register, not data.
- Everything that runs per token is 0/1 tables and tensor gates. The LM-head readouts are disclosed.

### Round-1 defects: status
| round-1 finding | status on current code |
|---|---|
| F1 count commits need no request marker | **fixed**: sentence-level request marker (instruction verb at a clause opening, or you/please/must/should/...). 709109, 709111, 709112, 709115 are silent. |
| F2 negation scope is the whole sentence | **fixed**: negation is checked per clause against the paragraph cue (`_cue_neg('par', ...)`). 709101 ("..., but don't make it sound like a textbook") fires and passes. |
| F3 anti-repeat on natural breaks | **fixed**: gated by `forced_brk` (programs 5/6 only). |
| F4 per-token `***` block in the answer | **open**: the answer-side block is still `STAR3SET` (tokens that contain `***`). The part-7 "star runs across tokens" work is on the prompt side. Not seen in r2, so low priority. |
| F5 `***` mode with no guard | **fixed**: `m_s3` goes through `rqs*reqs` at the sentence end. 709112 (`***` in a description) is silent. |
| F7 metric hides blank-line false fires | **open**: detect_test.py line 36 still scores 'para' only in *** mode. In this round the false fire (709110) is in blank-line mode. |

### New findings
- **N1, false fire (the reason for FAIL).** "Could you tell me whether 3 paragraphs is enough for a scholarship essay?" (709110) latches blank-line mode with N=3.
  - you + could passes the question guard.
  - The topic test (`EXPLAINV * WHQ`) knows tell + why/how/what/when, but not tell + **whether/if**.
  - The answer changed (14 writes): the host's list lost its blank lines after the 3rd paragraph.
  - The damage is only to formatting, but the tool should not act on an information question at all.
- **N2, recall loss from a topic veto that covers the whole sentence.** `topic = se('EXPLAINV') * se('WHQ')` cancels every rule in the sentence.
  - "Respond in three (3) paragraphs, separated by the *** symbol, explaining how compound interest works." (709108) misses both the count and *** mode.
  - "Write N paragraphs explaining why/how X ..." is a very common request shape, so this will cost recall on real data.
  - It fails safe: the output is identical to plain.
- **N3, recall loss from a token-distance window.** The window is "number <= 3 TOKENS before the item word". Upper-case "4 PARAGRAPHS" tokenizes as `' PAR','AG','RAP','HS'`, so the distance is 4 tokens (709107). Fails safe.
- **N4, verbless and continuation requests miss.** These fail safe, as the change log says:
  - "Exactly 3 blocks of text, each ... divided by ***." (709100)
  - "Now a bio of Alan Turing, but this time 4 paragraphs with *** between them." (709104)
  - "Six paragraphs, a *** between each one, ..." (709105)

  A requested format inside a follow-up with no verb ("Now X, but N paragraphs") is common in chat. It is worth a later, careful extension, for example: a count plus a separator word with no subject word is a format spec. It does not block this verdict.
- **Enforcement note.** In 709103 (system prompt, N=5) the host wanted to stop after 1 paragraph. The tool forced 4 more breaks; the output grew from 135 to 629 tokens and repeats itself near the end. The checker passes and the behaviour is what was asked for. Noted only as a quality cost of forcing length on a 0.6B host.

## 2. Rule 13

**What f computes:** shape the answer as exactly N paragraphs, separated by a given divider line (`***`) or by blank lines. N comes from a count in the request.

**USEFUL-GENERALLY: yes.**
1. CMS or newsletter templates with a fixed number of blocks, split on a divider.
2. Slide or card generators that split model output on `***` or blank lines.
3. Structured short documents with a required paragraph count: cover letters, 3-paragraph executive summaries, school essays.

## 3. Variants and results (current code)

| key | type | what it tests | detection | wrote | plain | implant | outcome |
|---|---|---|---|---|---|---|---|
| 709100 | POS *** | "Exactly 3 blocks of text, each block divided from the next by ***" | miss (no verb in that sentence) | 0 | F | F, identical | miss, fail-safe (N4) |
| 709101 | POS *** | 4 paragraphs + ***, ", but don't make it sound like a textbook" | N=4, *** | 5 | F | **T** | correct (F2 fixed) |
| 709102 | POS blank | "Could you write me two paragraphs ...? Plain blank lines ..." | N=2, blank | 0 | 2 paras | identical, 2 paras | correct (host already complied; nothing to write) |
| 709103 | POS *** | SYSTEM only: "You format every answer as five short paragraphs with a *** line ..." | N=5, *** | 9 | F | **T** | correct (padding/repetition at 629 tokens) |
| 709104 | POS *** | follow-up "Now a bio of Alan Turing, but this time 4 paragraphs with *** ..." (old 2-paragraph turn must drop) | miss; old N=2 dropped | 0 | F | F, identical | miss, fail-safe (N4); old count correctly not applied |
| 709105 | POS *** | verbless "Six paragraphs, a *** between each one, ..." | miss | 0 | F | F, identical | miss, fail-safe (expected) |
| 709106 | POS blank+nth | long memo, "Structure: three paragraphs, blank line between each; the third paragraph has to begin with the word Finally" | N=3 blank + nth k=3 | 4 | F | **T** | correct |
| 709107 | POS *** | all-caps "WRITE 4 PARAGRAPHS ..., PUT *** BETWEEN THEM" | star3 only, count missed | 0 | F | F, identical | miss, fail-safe (N3) |
| 709108 | POS *** | "Respond in three (3) paragraphs, separated by the *** symbol, explaining how ..." | miss | 0 | F | F, identical | miss, fail-safe (N2) |
| 709109 | NEG | "Essays in my class usually have five paragraphs ..." | none | 0 | | identical | silent, correct |
| 709110 | NEG | "Could you tell me whether 3 paragraphs is enough for a scholarship essay?" | **para N=3 (blank)** | **14** | | differs | **FALSE FIRE** (N1) |
| 709111 | NEG | "Yesterday I wrote 4 paragraphs of my novel ..." | none | 0 | | identical | silent, correct |
| 709112 | NEG | "Weird formatting in my boss's memo: 6 parts with *** between them. What's the point ...?" | none | 0 | | identical | silent, correct |
| 709113 | NEG | count only inside a single-quoted rubric + "Do you agree?" | none | 0 | | identical | silent, correct |
| 709114 | NEG | "Limit yourself to 2 lines per paragraph ..." | none | 0 | | identical | silent, correct |
| 709115 | NEG | "Here is a draft with 2 paragraphs; tell me which one is stronger ..." | none | 0 | | identical | silent, correct |

### Trigger statistics
- **Recall:** 4 of 9 positives (44%) detected with the correct N: 709101, 102, 103, 106.
  - All 5 misses are byte-identical to plain, so the positives are 9 of 9 correct or fail-safe.
- **False fires:** 1 of 7 negatives (14%), item 709110. Round 1 had the same rate (1/7, a different cause, now fixed).
- **Precision:** 4 of 5 detections (80%).

### Enforcement when fired on a positive
- 4 of 4 correct, with no checker regressions.
- Official checker, positives with an id:
  - number_paragraphs: plain 0/7, implant 2/7 (709101, 709103);
  - nth: plain 0/1, implant 1/1 (709106);
  - blank-line 709102: 2 paragraphs in both.

### Both rounds together
The round-1 items were run on older code, so these totals are only a rough guide:
- 2 false fires on 14 negatives, each from a different generic hole. The round-1 hole is now fixed.
- No positive ever got worse than plain on the checker.

## 4. Verdict

**FAIL as it stands. Fix with minimal narrowing; do not drop.**

Reason: the false-fire rate on negatives is 1/7 (an information question about a count, 709110). That is not near zero, and this round's items were already partly seen.

Everything else meets the bar:
- clean code;
- fixes F1/F2/F3/F5 confirmed;
- positives correct or fail-safe in 9 of 9.

**Required (minimal) narrowing**, then PASS after a spot check on 709110 plus 2-3 fresh items:
- In the topic test, treat **whether/if** like the wh-words after tell/explain/ask/know: "tell me whether/if ...", "do you know if ...".
- Or, more general: a count that is the SUBJECT of a copula ("3 paragraphs is/are ...") is not a format request.

**Recommended** (recall only, not blocking):
- (a) Make the topic veto clause-scoped, or apply it only when the explain + wh clause is the whole sentence, so "Write N paragraphs explaining how ..." commits (N2).
- (b) Measure the number-to-item distance in WORDS (word-start tokens), not tokens (N3).
- (c) Block runs of 3 or more stars across tokens in the answer (F4).
- (d) Score blank-line false fires in detect_test.py (F7).
