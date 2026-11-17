# JUDGE end_phrase, round 1

Tool: end_phrase. The reply must end with an exact phrase given in the prompt.
- Detector: an end-word (end/finish/conclude/close/last/final/sign off/wrap up/round off), then a quoted span or a span after ':'. The span is kept as pointers.
- Enforcement: at end of turn, copy the span from the prompt by pointer, then end the turn.

Judge: independent reviewer, protocol rules 3, 5, 8, 10-13.
- Variants: `judge/end_phrase_variants.jsonl`, keys 704000-704024.
- Results: `judge/end_phrase_results_r1.md`.
- The run used the circuit after change-log parts 4 and 5 (NOTES.md). Part 5 used detection results on all round-1 judge items, mine included. So these items are no longer fully unseen (see "Contamination" below).

## 1. Code findings (current circuit.py, lines ~292-320 detect/capture, the answer-side suffix matcher, the c6 end-of-turn program; tables.py)

### No reward hacking, no leakage, no Python logic on this path
- Detection and capture are all tensor operations: 0/1 class tables, clause and sentence scope registers, an imperative-position register, and pointer registers (`end_s`, `end_len`, `end_x`) set by `torch.where`.
- Enforcement is one program that the end-of-turn readout starts: separator + span copied by a copy head over `mem` + `end_x` + an optional `"` + EOS. This readout is disclosed. A suffix matcher (case-insensitive, by `CANON`) stops a second append when the host already wrote the phrase.
- The new turn-scope code loops in Python over the fixed rule names (`for k in self.L`). This is structure, not branching on data. Acceptable.

### Word lists are generic English
- `ENDW`, `WHOLE2`, `YOUW`, `WITHW`, `CANCELW`, `NEG` and `OTHERS`/`DESCV` are generic lists.
- `PRE` splits a closing token such as `."` or `?"` into the punctuation and the quote. This works for any phrase.
- The dev phrases (gen_data.py `PHRASES`) are our own.
- The `wrap up` cue and the colon form come from drv:2268 and drv:286 (even-key rewordings, allowed by the author).
- I found no IFEval-specific string or template.

### Guards in the current code (part 4/5)
The detector now arms only if all of these hold:
- the clause has an end-word;
- the clause has a request marker: `you/your`, a whole-response word, or an end-word in imperative position followed by `with`;
- the clause is not negated;
- the sentence is not a non-request question;
- the clause has no third-person subject or third-person verb;
- the clause has no start-word, no each/every word, and the sentence has no paragraph word.

Quoted text is masked, in both double and single quotes. Earlier user-turn latches are dropped at each new user turn; system-message latches persist. The guards are generic. The 'with' requirement for the imperative path is narrow, but it is a real property of the construction ("end with X").

### Remaining weaknesses (each one fails safe; none produced a wrong output)
1. **Bug: multi-piece end-words never count as imperative.**
   - The n-gram reader flags `wrap up` / `sign off` / `round off` on the LAST piece (`' up'`). `imp_end = ENDW * wlead` is read at that piece, and the piece is not at the clause opening.
   - Effect: "Wrap up the list with "X"" (704001) does not arm. It arms only when `your`/a whole-response word is also present, as in the dev template drv:2268 "Please wrap up your answer with", which is why dev did not catch it.
   - Fix: take `wlead` from the first piece of the matched n-gram (one more shifted compare).
2. **A merged opener token is not recognized.**
   - Qwen tokenizes `with "- Helpdesk Bot"` as `' "-'`. `OPENDQ` requires the token to be only a quote, possibly after a bracket or colon, so the capture never opens (704005, system message).
   - This is a generic recall gap for phrases that begin with punctuation (`- Name`, `-- sig`, `"...`).
   - Fix: accept an opener token that has a quote and no letters, and include the trailing punctuation in the span.
3. **The imperative path is strict.**
   - "and this time end with" (704007) does not arm: 'this time' sits before the end-word, so the end-word is not at the clause opening.
   - Colon forms with a modal do not arm, because the end path does not use `REQMARK`: "the last words are: X" (704009), "The final line must be exactly: X" (704011).
   - Casual "end ur msg w/ 'X'" (704013) does not arm: there is no `your` and no `with`.
4. **Cosmetic: the sign-off is duplicated.**
   - When the host's own ending is almost the phrase, the token-level suffix matcher fails and the phrase is appended again. Seen with a trailing emoji ("Go grow something! 🌱"), a curly apostrophe ("Don’t" vs "Don't"), and a line break inside the phrase ("Warm regards,\nUnit 4B").
   - The checker passes, but the reply repeats its sign-off.
   - Fix: normalise ’ to ' in `CANON`, and skip trailing non-letter tokens in the matcher.
5. **Untested false-fire risk for round 2: second-person descriptions with last/final.**
   - Example: "Your last reply said "see you soon" but ...", or "Your final answer was "42"; please double-check it."
   - These have `your` + an end-word + a quote, no negation, no '?', and no third person. By my reading of the code they would arm and force the old text as the ending.
   - Suggested narrowing: when the end-word is the adjective last/final, require `with` or a copula + must/should ("must be"), as for the imperative path.

## 2. Rule 13: is f(x) generally useful?
f(prompt, reply) = if the user (or a system message) asks that the reply end with an exact phrase, copy that phrase verbatim from the prompt and make sure the output ends with it. The phrase is appended at end of turn unless the host already wrote it.

**USEFUL-GENERALLY: yes.** Three settings outside the benchmark:
1. Mandated sign-offs or disclaimers on support and email replies, such as "- Helpdesk Bot" or "This is not legal advice." set in a system prompt.
2. Protocol terminators that a downstream parser waits for, such as "END OF MESSAGE", "Over." or a ticket footer.
3. Template and form generation, where a fixed closing line is required (letters, notices, chat-ops bots that must end with a status tag).

## 3. Variants and results (plain vs implant, same batches; strict = official checker)

| key | type | what it tests | latched (arg) | plain | implant | identical |
|---|---|---|---|---|---|---|
| 704000 | POS | casual, "finish off with", after the request | end ("go grow something") ✓ | F | **T** | no |
| 704001 | POS | middle of prompt + word limit; "Wrap up the list with" | silent (bug 1) | T | T | yes |
| 704002 | POS | single quotes, apostrophe inside the phrase | end ("Don't look behind you.") ✓ | F | **T** | no |
| 704003 | POS | "Open with A and close with B" in one sentence | end ("Warm regards, Unit 4B") ✓ | F | **T** | no |
| 704004 | POS | phrase BEFORE the cue ("Use "X" as the very last line") | silent | F | F | yes |
| 704005 | POS | system message "Always end your replies with "- Helpdesk Bot"" | silent (gap 2) | F | F | yes |
| 704006 | POS | system message "Every reply ... must end with" | silent (each/every) | T | T | yes |
| 704007 | POS | multi-turn follow-up, "and this time end with" | silent (gap 3) | T | T | yes |
| 704008 | POS | standing "sign off with" in an EARLIER user turn | silent (turn scope, by design) | F | F | yes |
| 704009 | POS | buried in a long message, "the last words are: X!" | silent (gap 3) | F | F | yes |
| 704010 | POS | curly quotes, "and conclude with" | end ("Money grows when you wait.") ✓ | F | **T** | no |
| 704011 | POS | colon form, two-sentence phrase | silent (gap 3) | T | T | yes |
| 704012 | POS | synonym only ("at the bottom") | silent | F | F | yes |
| 704013 | POS | "pls end ur msg w/ 'k thx'" | silent (gap 3) | F | F | yes |
| 704014 | POS | multi-turn OVERRIDE of an earlier phrase | end ("Woof, that is all.") ✓, host already complied | T | T | yes |
| 704015 | NEG | grandfather "ends his letters with "..."" (third person) | silent ✓ | - | - | yes |
| 704016 | NEG | "Why do pilots end radio calls with "over"?" | silent ✓ | - | - | yes |
| 704017 | NEG | instruction inside quoted text | silent ✓ | - | - | yes |
| 704018 | NEG | "The final score was "3-1"" | silent ✓ | - | - | yes |
| 704019 | NEG | negated "do not end your reply with "Hope this helps!"" | end silent ✓ (forbid latched, output unchanged) | - | - | yes |
| 704020 | NEG | "My lease ends next month: ..." | silent ✓ | - | - | yes |
| 704021 | NEG | "Finish this sentence ...: ..." | silent ✓ | - | - | yes |
| 704022 | NEG | "closing lines of my cover letter: ..." | silent ✓ | - | - | yes |
| 704023 | NEG | quoted last line mentioned in an earlier turn | silent ✓ | - | - | yes |
| 704024 | NEG | "what does it mean ... end an essay with ..." | silent ✓ | - | - | yes |

### Trigger statistics (end rule)
- **Recall:** 5/15 = 33% of positives latched. Every latched argument is exact (5/5). There were no wrong arguments, so no wrong phrase was ever enforced.
- **Precision:** 5/5 = 100%. False fires on negatives: 0/10.
- **Fail-safe:** all 10 missed positives and all 10 negatives are byte-identical to plain (20/20).
- **Prompt-level strict on positives:** plain 5/15, implant 9/15. That is +4 gained and 0 lost (704000, 704002, 704003, 704010).
- **Side note for the forbidden-words judge:** on 704019 the forbid rule latched on "Hope this helps". That is a reasonable reading of "do not end with X", and the output was unchanged.

### Contamination
- Change-log part 5 was written after running detection on all round-1 judge items, including these. Three of my items match part-5 changes exactly:
  - 704021 → the 'with' requirement;
  - 704023 and 704014 → the turn-scope change;
  - 704002 (single quotes) → the single-quote masking.
- My preliminary predictions for the pre-part-4 code were 8/10 false fires. That is the honest "unseen" number for the old circuit.
- The 0/10 here holds for the current circuit on items it was partly tuned against.
- The changes themselves are generic, not IFEval-specific: request guards, a 'with' requirement, turn scope, quote masking.

## 4. Verdict: PASS (conditional)

**Why PASS:**
- No reward hacking, no IFEval-specific strings, no test leakage, and no Python logic on the inference path.
- On the positives, every item is either correct (5) or fails safe with byte-identical output (10). There are no wrong arguments.
- 0/10 false fires on the negatives, with all outputs byte-identical.
- Net +4 / -0 on the positives.
- The function is generally useful.

**Conditions and recommendations:**
- Recall is low (33%) because the request guards are strict. This costs score but is safe.
- Fix bug 1 (multi-piece end-words are never imperative). It is a real generic bug, and the fix does not need new vocabulary.
- Fix gap 2 (an opener token merged with punctuation).
- Consider using `REQMARK` (must/should/make sure) for the colon form.
- Round 2 must use fresh variants. It must include second-person descriptions with "your last/final ... "X"" (weakness 5), which I expect to false-fire under the current code.
- Do not drop the rule; narrow last/final as described in weakness 5 if round 2 confirms that risk.
