# JUDGE_postscript_2: postscript rule-keeper, round 2 (final)

- **Judge:** independent reviewer, round 2.
- **Code reviewed:** the CURRENT circuit.py and tables.py, after NOTES.md change-log parts 6 and 7.
- **Items:** judge/postscript_variants_r2.jsonl (keys 705100-705119). These are fresh items, different from round 1 and from gen_data.py.
- **Results:** judge/postscript_results_r2.md, run on the part-7 code.
- **Contamination disclosure:** the builder ran DETECTION on these items while writing part 7. The numbers below are optimistic for that
  reason. The ps-specific lists did not change in part 7: PSW and PSV are the same as in round 1.
- **Earlier verdict:** see JUDGE_postscript_1.md (PASS, conditional).

**VERDICT: PASS (conditional).** One required fix and two recommended fixes are listed in section 5. This rule must not be dropped. If the
team counts every false latch as a false fire, then the verdict is FAIL with a minimal narrowing: fix item 1 below. Do not drop the rule.

## 1. Function and Rule 13

f(x) = "if the user asks for an after-note, make sure the reply ends with a labelled postscript (P.S., or P.P.S. if the user asks for one);
add nothing if the reply already has one".

**USEFUL-GENERALLY: yes.** Three settings outside the benchmark:
1. Email and cover-letter assistants.
2. Fund-raising and marketing letters, where the P.S. holds the call to action (see item 705104).
3. Cards and notes, where an app sets the P.S. in its system message.

## 2. Code review of the current ps path

- **Python logic at inference: none found.** The path is still tensor gates, fixed tables and `torch.where`. Regex is used only offline in
  `tables.build`, for example the new SENTB/CLAUSEB exception for '.P'. The loops are over static rule names.
- **IFEval strings: none found.** The new lists (INSTV, REPORTV, data masks, topic-sentence words) are generic English words. They are not
  sentences or templates.
- **Detector:**
  - `PSW` (postscript / P.S. / P.P.S.) and `PSV` (add/include/append/attach/put/insert/end/close/conclude/finish/sign/follow/tack/stick)
    must both be in the clause.
  - There must be no clause negation, except "don't forget".
  - There must be no explain/describe verb.
  - The request guards must pass. A request marker is now one of: an instruction verb at the clause opening (INSTV), a modal or you/please
    word anywhere, or a negation at the sentence opening.
  - The rule is committed at the sentence end, so a later '?' that is not a request cancels it.
  - Quoted spans and data spans are masked.
- **Enforcement is unchanged:** at end of turn, if no P.S. has been seen, the circuit forces `sep P .S .` (or `P .P .S .`).

**Remaining defects in the current code:**

1. **'.S' is still a sentence and clause boundary (the main defect).**
   - The tokenizer splits "P.S." as `' P' '.S' '.'`. Part 6 made '.P' a non-boundary but kept '.S' as a boundary on purpose.
   - The part-7 rule "commit at sentence end; a later '?' cancels" therefore commits a P.S. clause at the '.S' token, before the '?' or the
     rest of the sentence is read. The fix that part 7 claims for "Should I include a P.S. ...?" does not work for this rule. 705114 shows
     this.
   - The same boundary makes every verb after the marker invisible ("a P.S. at the end would be great": 705102).
   - Fix (generic): '.S' is not a boundary when the next token is the '.' of the same abbreviation. Then the clause or sentence ends at
     that '.' token (or at the next real boundary).
2. **REPORTV has only third-person and past forms.** "Marketing books **say** you should always add a P.S." latched (705115), because 'say'
   is missing while 'you' and 'should' satisfy REQANY.
   - Fix: add the base forms (say, suggest, recommend, claim, argue) when a non-'you' subject opens the clause.
3. **An adverb before the instruction verb removes the imperative opening.** IMPPRE has please/kindly/also/then/and/just/finally, but not
   always/simply/briefly/never.
   - The system prompt "When you draft letters, always include a P.S. with a call to action" did NOT latch (705104). The 'you' is in the
     previous comma clause, and "always include" does not count as an opening verb.
   - The system rule in 705116, "Always end emails with a P.S.", did not latch either. That negative was silent for this wrong reason, so
     the system-override case was not really tested.
   - This is a recall loss, and it fails safe. "Always add/include/end with X" is the most common system-prompt form, so add frequency
     adverbs (always, simply, briefly, usually) to IMPPRE.
4. **False latches are hidden by the host's own "P.S." text.** Both r2 false latches (705114, 705115) and the r1 false latch (705016) did not
   change the output. In each case the host's answer contained the string "P.S.", which sets `ps_seen`.
   - This self-masking is structural for prompts that mention "P.S.", because the answer usually repeats the abbreviation.
   - It does not protect prompts that only say "postscript", or answers that do not quote the marker. Do not count it as a guard.
5. **Round-1 condition 3 is still open:** a spaced host "p. s." is not detected, so the circuit can force a second P.S. (seen in r1 705001).

## 3. Round-2 variants and results (part-7 code)

### Positives (12)

| key | what it tests | expected | latched | strict ps plain -> implant | identical |
|---|---|---|---|---|---|
| 705100 | verb not in PSV: "Throw in a P.S." | miss | no | T -> T | yes |
| 705101 | "Leave a short P.S. ..." | miss | no | T -> T | yes |
| 705102 | verb after the marker: "a P.S. at the end would be great" | miss | no | T -> T | yes |
| 705103 | "Don't forget to add a P.S." (not a negation) | fire | **yes** | T -> T | yes |
| 705104 | system message "always include a P.S." + multi-turn | fire | **no** (defect 3) | T -> T | yes |
| 705105 | bare "Add a P.P.S." (round-1 defect) | fire | **yes** (ps+pps) | T -> T | yes |
| 705106 | "put a P.S. under the signature" in an instruction list | fire | **yes** | T -> T | yes |
| 705107 | bare "PS" without dots | miss | no | F -> F | yes |
| 705108 | caps + "conclude with a P.S." split at 'and' | fire | **yes** (caps+ps) | T,T -> T,T | yes |
| 705109 | buried in a long casual message, "could you include a postscript" | fire | **yes** | T -> T | yes |
| 705110 | P.S. asked for only in the current follow-up turn | fire | **yes** | T -> T | yes |
| 705111 | both P.S. and P.P.S. requested | fire | **yes** (ps+pps) | T -> T | yes |

- Latch recall: 7/12. It is 7/8 on the items I expected to fire; the one surprise miss is 705104.
- All 5 misses fail safe: the output is byte-identical to plain.
- Strict ps: plain 11/12, implant 11/12. Gained 0, lost 0. The host already complied on these items, so this round could not show a gain
  (round 1 showed +3).

### Negatives (8): the cue words are mentioned, no P.S. is asked for

| key | what it tests | latched | identical |
|---|---|---|---|
| 705112 | "Grandpa would always add a P.S. ..." (noun subject) | no | yes |
| 705113 | "Sometimes copywriters add a P.S. ..." | no | yes |
| 705114 | "Should I include a P.S. in my cover letter, or ...?" | **YES** (defect 1) | yes, masked: the answer contains "P.S." |
| 705115 | "Marketing books say you should always add a P.S. ..." | **YES** (defect 2) | yes, masked: the answer contains "P.S." |
| 705116 | system "Always end emails with a P.S.", user says "skip the P.S." | no (system rule never latched, defect 3) | yes |
| 705117 | instruction inside a single-quoted span to translate | no | yes |
| 705118 | "Tell me when it's appropriate to add a P.S. ..." | no | yes |
| 705119 | "I add a P.S. to almost every email." + an unrelated request | no | yes |

- False latches: 2/8.
- Output-changing false fires: 0/8.

### Trigger precision and recall

| | precision | recall | output changes on negatives | strict losses |
|---|---|---|---|---|
| Round 2 | 7 / (7 + 2) = 0.78 | 7/12 = 0.58 | 0/8 | 0 |
| Rounds 1 + 2 (40 items) | 16/19 = 0.84 | 16/24 = 0.67 | 0/16 | 0 |

Across both rounds, strict ps gained 3 and lost 0. Every miss failed safe.

## 4. My round-2 predictions vs the observed results

- **Fixed by part 7:** I predicted 6 false fires on the part-6 code. Part 7 removed 4 of them: 705112, 705113 and 705118 for the right
  reason (the instruction-verb opening and the topic-sentence guards), and 705116 for the wrong reason (defect 3).
- **Still present:** the 2 remaining false latches (705114, 705115) have one root cause each, and both are generic and small.

## 5. Verdict and conditions

**PASS (conditional).** Reasons:

- No reward hacking, no Python logic at inference, no IFEval strings or templates.
- On 24 positives over two rounds, the rule fires correctly or fails safe (byte-identical output) on every item. It never lowers a score.
- On 16 negatives no output changed. On 3 of them the rule latched, and only the host's own "P.S." text kept the output unchanged. That
  masking is not a designed guard.

**Required before the final IFEval run (minimal narrowing, not a drop):**

1. '.S' must not end the clause or sentence when the next token is the '.' of the abbreviation. Commit the P.S. clause at the following
   real boundary. This makes the part-7 question guard work for "Should I include a P.S. ...?". It also lets a verb after the marker be
   read.

**Recommended:**

2. Add the base forms of the reported verbs (say/suggest/recommend/claim/argue) to REPORTV when the clause subject is not "you".
3. Add frequency adverbs (always, simply, usually, briefly) to IMPPRE. The system-prompt form "Always add a P.S." then latches; this is
   705104 and the untested half of 705116.
4. Detect a spaced host "p. s." / "p. p. s." as a P.S. already present, to avoid a duplicate postscript.

**If the lead's bar is "zero false latches on negatives", not "zero changed outputs":** then this is FAIL. The fix is the minimal
narrowing in item 1 (with item 2), not dropping the rule. The rule is generally useful, and its failures so far all fail safe at the
output level.
