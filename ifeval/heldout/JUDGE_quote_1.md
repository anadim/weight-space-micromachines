# JUDGE_quote_1: quote rule (startend:quotation), judge round 1

**Verdict: PASS, provisional.** One generic guard gap must be fixed (see F1), and round 2 must confirm with new unseen variants.

Reviewer: independent judge subagent. Files read: circuit.py, tables.py, gen_data.py (quote templates), NOTES.md (parts 1-5), run.py, detect_test.py.
Results file: judge/quote_results_r1.md. I did not open any data/, fresh, ood or IFEval file.

**Test-hygiene caveat:**
- The builder changed the circuit twice after my preliminary review:
  - Part 4 was driven by the code reviews.
  - Part 5 was driven by running detection on all judges' variants, mine included.
- The results below are therefore on the CURRENT circuit, and my variants are no longer unseen for that circuit.
- They are evidence that the fixes work on these items. They are not clean held-out evidence.
- Under the part-4 circuit, at least two of my negatives would likely have fired: 703016 (single quotes) and 703019 (carry-over from an earlier turn). This is my estimate from the part-5 log; it was not measured.

## 1. Code review of the quote path (current code)

**Detector (circuit.py ~l.255):**
`m_q = QUOTEW · (1-SINGLE) · max(QMARKW, WRAPS) · max(WRAPV, INW) · WHOLE2 · (1 - ENDW·(1-STARTW_sentence)) · (1 - negc)`
- It is latched at the end of the clause through `ce = cb · rqc · reqm`.
- `rqc` blocks a non-request '?' and blocks OTHERS/DESCV words in the clause.
- `reqm` requires a request marker (REQMARK, WHOLE2, an opening negation, or an imperative-position verb).
- Clause boundaries are `, ; . ! ? \n` and also 'and'/'but'.
- Text inside double or single quotes is masked.
- Turn scope: a user-turn latch applies only to the reply to that turn. System latches persist. Cancel cues drop system latches.

**Enforcement:**
- At answer start it forces '"' (program 1).
- When the host picks end-of-turn and its last non-space token does not end in '"', it forces '"' and then EOS (c7).
- The end-phrase program appends '"' before its own EOS.
- No veto vectors are used.

**Reward hacking / test fitting: none found.**
- The word lists are generic English: quotation/quotes/speech marks/inverted commas; wrap/enclose/surround/bookend; entire/whole/full/complete/response/answer/reply/output/text/message.
- There are no IFEval strings or templates in the circuit.
- The drv templates in gen_data.py (drv:122, 2532, 3536) are rewordings, which the author allowed for dev.
- The part-4/5 changes are generic guards, not item-specific patches. Negation, the request marker, the single-quote mask and the turn scope apply to all rules.

**Python logic at inference: none.**
- step/write use only torch where/gather/table lookups.
- The Python `for` loops run over fixed rule-name lists.
- `if fx:` runs only at init.
- Regex is used only in offline table building.
- The three LM-head readouts are disclosed.

**Findings (current code):**
- **F1 (real false fire, 703015): a whole-response word counts as a request marker.**
  - `reqm` includes `cl('WHOLE2')`.
  - So a description such as "My manager wraps her entire reply to clients in quotation marks ..." passes `reqm` by itself, through "entire reply".
  - It also passes `rqc`, because OTHERS has no possessive or object pronouns (her/his/their) and does not recognise noun subjects (my manager / my boss).
  - Fix: in clauses with a possessive third-person determiner (his/her/their) or a non-second-person subject, WHOLE2 should not count as a request marker. Alternatively, require a REQMARK other than WHOLE2 when the clause has no imperative-position verb.
- **F2 (safe miss, 703007): sub-word pieces are probably read as words.**
  - "bookend your reply with speech marks" did not fire. The probable cause is that the continuation piece 'end' of ' book'+'end' gets the single-token ENDW class from the vocabulary table.
  - ENDW without STARTW then vetoes the rule.
  - I could not verify the tokenization on the development machine because transformers is not installed. Please check this.
  - If confirmed, the fix is generic: single-token class flags should apply only to word-initial tokens (WSTART), or only to tokens that are not followed by a continuation piece.
- **F3 (safe miss, 703022): the synonym 'inverted commas' alone no longer fires.**
  - The part-4 rule needs QMARKW (marks/mark/double) or a wrap verb.
  - Clean fix: give the multi-word synonyms 'inverted commas' and 'speech marks' the QMARKW class.
- **F4 (safe misses, 703001 and 703010): clause scope is short.**
  - An instruction split by commas ("Every answer, including this one, must be in double quotation marks ...") has its cues in different clauses.
  - These misses are safe.
- **F5: the closing quote depends on the host ending the turn.** If the host loops to max_new, nothing closes the quote.
- **F6 (disclosed design choice): multi-turn carry-over is not supported.** A user-turn instruction is dropped at the next user turn. This fails safe, as in 703019.

## 2. Rule 13: is f(x) generally useful?

f(x): given a request to quote the whole reply, emit '"' as the first answer token and '"' immediately before end-of-turn. The output is then exactly one string wrapped in double quotes.

**USEFUL-GENERALLY: yes** (narrow, but real). Three settings outside the benchmark:
1. Producing a value that must be a JSON string, a string literal in code, or a CSV-quoted field.
2. Passing the whole reply as a single shell or CLI argument, or a config value.
3. Dialogue, caption and subtitle pipelines that expect one quoted utterance per reply.

## 3. Variants and results

There are 23 items: 14 positives and 9 negatives. File: judge/quote_variants.jsonl, keys 703000-703022.
- "fire" = the 'quote' latch in the circuit report.
- "strict" = the official IFEval quotation checker, plain then implant.
- "same" = the implant output is byte-identical to plain.

| key | type | what it tests | fire | strict plain / implant | same | note |
|---|---|---|---|---|---|---|
| 703000 | POS | system message, "whole replies go inside double quotation marks" | yes | T / T | yes | |
| 703001 | POS | system message split over 3 comma clauses | no | T / T | yes | safe miss (F4); host did it alone |
| 703002 | POS | multi-turn follow-up with history | yes | T / T | yes | |
| 703003 | POS | casual chat, "wrap ur whole msg in quotes lol" | yes | T / T | yes | |
| 703004 | POS | buried mid-way in a long message | yes | T / T | yes | |
| 703005 | POS | two instructions in one sentence (+ <80 words) | yes | T,T / T,T | yes | |
| 703006 | POS | inverted order "In double quotation marks is how your entire answer must appear" | yes | T / T | no | content changed, still correct |
| 703007 | POS | "bookend your reply with speech marks" | no | F / F | yes | safe miss (F2); host also failed |
| 703008 | POS | polite question "Can you make sure the full response is surrounded by quotation marks?" | yes | T / T | yes | |
| 703009 | POS | odd punctuation "whole response -> inside quotation marks, ok??" | yes | T / T | yes | |
| 703010 | POS | instruction interrupted by a parenthetical | no | T / T | yes | safe miss (F4) |
| 703011 | POS | "Don't leave any part of your answer outside double quotation marks" | no | T / T | yes | safe miss |
| 703012 | POS | literal '"' character named instead of the word | no | F / F | yes | safe miss; host also failed |
| 703022 | POS | "the whole reply goes in inverted commas" (+ cross-rule check) | no | T / T | yes | safe miss (F3); no_comma did NOT false-fire (fixed in part 4) |
| 703013 | NEG | explicit negation "don't wrap your answer in quotation marks" | no | - | yes | negation guard works |
| 703014 | NEG | "include two famous quotes ... in your response" | no | - | yes | |
| 703015 | NEG | "My manager wraps her entire reply ... in quotation marks" | **YES** | - | **no** | **FALSE FIRE (F1).** The output changed meaning: plain gave the asked-for advice; implant wrapped a reply that misread the request. |
| 703016 | NEG | single-quoted meta question about the rule | no | - | yes | single-quote mask works |
| 703017 | NEG | rule inside double quotes (reported speech) | no | - | yes | |
| 703018 | NEG | "Put all book titles in quotation marks ..." | no | - | yes | host ran to max length in both runs, identical |
| 703019 | NEG | rule given for the previous turn only | no | - | yes | turn scope works |
| 703020 | NEG | "Quote the whole first sentence of ..." | no | - | yes | |
| 703021 | NEG | information question about the rule | no | - | yes | |

**Trigger statistics (current circuit):**
- Positives: 8/14 fired. **Recall 57%.**
- Negatives: 1/9 fired. **False-fire rate 11%.**
- **Precision 8/9 = 89%.**

**Enforcement:**
- 8/8 fired positives pass the checker.
- 6/6 misses are byte-identical to plain, so they fail safe.
- 8/9 negatives are byte-identical to plain.

**Score:**
- Positives: plain 12/14 strict, implant 12/14 strict. There is no gain on this set: the frozen host already wraps in quotes when asked plainly.
- The two items the host fails (703007, 703012) are both detector misses.
- No positive was lost.

For comparison, my hand-traced prediction on the ORIGINAL circuit was 10/14 fired and 7/9 false fires. The part-4/5 guards removed 6 of those 7 false fires and cost about 2 positives (703007, 703022).

## 4. Verdict: PASS (provisional)

Reasons:
- There is no reward hacking, no IFEval-specific string, no data leakage and no Python logic at inference.
- Every positive is either correct (8/8 fired items pass) or a safe miss (byte-identical to plain).
- Only 1 of 9 negatives that I wrote to break the rule still fires. The builder reports that this is the only quote false fire on all 311 judge items.

Conditions:
1. **Fix F1.** A whole-response word must not count as a request marker in a clause that describes someone else (possessive his/her/their, or a non-second-person noun subject). The false fire changes the meaning of the reply, so it is not harmless.
2. **Check F2** (sub-word piece read as a word). If confirmed, fix it generically, because it affects every rule that uses the single-token tables.
3. **Optional:** F3 (give 'speech marks'/'inverted commas' the marks class) to restore the synonym path.
4. **Round 2 must re-test with fresh variants.** My items informed part 5, so this result is not clean held-out evidence. The false-fire rate on unseen descriptions of someone's habit is the number to watch.

Keep the rule. Narrow only its request-marker condition. Do not drop it.
