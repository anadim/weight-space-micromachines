# JUDGE two_responses, round 1

Judge: independent reviewer (subagent). Tool: two_responses (two different responses split by `******`).
Code reviewed: the CURRENT circuit.py / tables.py (after NOTES.md change log parts 4 and 5).
Variants: judge/two_responses_variants.jsonl (keys 708000-708019). Results: judge/two_responses_results_r1.md.

## Verdict: FAIL (enforcement). Keep the rule and keep the detector. Fix the enforcement and re-judge. Do not drop or narrow the rule.

- **Detection passes.**
  - Precision 11/11. Recall 11/12, and the one miss fails safe.
  - 0/8 false fires on the negatives.
  - All 9 silent outputs (8 negatives + the missed positive) are byte-identical to plain.
- **Enforcement does not pass.** The official checker passes 11/12 implant outputs, against 3/12 for plain. But only about half of those passes are two real, different responses:
  - 3 passes are hollow: the second part is a copied sign-off line or a repeated end phrase.
  - 1 pass is an accident of the checker: the host wrote a 10-asterisk run, and the circuit counted it as the divider.
  - 1 item regressed: plain was already correct and short, and the implant made the host loop until the 2048-token limit.

  That last failure comes from a generic bug, not a quirk of one prompt. It will happen whenever the second response contains no Latin letters: numbers, code, symbols, non-Latin scripts. A tool meant to be as general as the calculator must not do that.

## 1. Code findings (current code)

### Path

**Detector.** circuit.py lines 257-261.

- `se_six` latches in the sentence when either:
  - a token contains `******` and it is not inside quotes (`STAR6 * (1-inq_prev)`), or
  - an asterisk/star word comes within 3 tokens after the number 6.
- `m_two = se_six * max(SEPV, TWOW) * sentence_end * rqs * (1-negs)`, where:
  - SEPV: separate / divide / between / split.
  - TWOW: two / both / responses / versions / alternatives / answers / options / variants / replies / drafts.
  - `rqs` is the request guard: not an information question, and not a third-person description.
  - `negs` is the sentence-level negation guard.

  The match is read at the sentence end, so word order does not matter.

**Turn scope.** Lines 449-457.

- Latches from earlier user turns are dropped at each new user turn.
- System-message latches persist.
- A cancel cue drops the system latches.

**Enforcement.** write(), lines 621-654.

- At the first end-of-turn with `k6 == 0`, the emitter forces `\n\n ****** \n\n`.
- After the divider, end-of-turn is blocked while `w6 < 5`.
- Tokens containing `***` are blocked after the divider, and also before it while the answer has fewer than 5 words.
- The first word piece of the first response is blocked at the start of the second response.
- Paragraph counting is disabled while this rule is on.

### Reward hacking / test fitting

None found.

- The only benchmark-shaped string is the literal `******`. It is the rule's own delimiter, so it works as an interface, not as IFEval wording.
- The word lists SEPV, TWOW and ASTER are generic English.
- Dev templates are our own wording, plus 3 `drv:` rewordings of even-key sentences, which the author allowed.
- No IFEval data, odd keys or fresh/ood files were opened by me or appear in the path.

### Python logic at inference

None found.

- Every gate is `torch.where` / `max` / `clamp` on 0/1 wires, table lookups, or counters.
- The Python loops (`start_prog`, `pad`, `for k in self.L`) run over static lists.
- Regex runs only at table build time.
- The role-token ids hard-coded in the circuit (151644 `<|im_start|>`, 872 `user`, 8948 `system`, 77091 `assistant`) are chat-template constants, not data branching.
- The LM-head readouts are disclosed.

### Weaknesses (each one seen in the run)

1. **The word counter counts only Latin letters.** This is the main bug.
   - `w6` counts tokens with `LETTERS` (tables.py:167, `[A-Za-z]`).
   - End-of-turn is blocked until `w6 >= 5`, and there is no cap.
   - A second response made of numbers, code, symbols, emoji, or a non-Latin script never reaches 5, so the host is forced to continue until the token limit.
   - Seen in 708009: the plain output was a correct, short two-response answer. The implant output is `7 × 8 = 56` repeated to 2048 tokens.
   - Seen in 708010: the second part is "Purr on. 🐾✨" repeated until 5 letter-words were reached.
   - Fix: count any non-whitespace word start, and cap the block (for example, allow end-of-turn after K tokens of the second part).
2. **The divider counter accepts longer star runs.** `k6` counts any token that contains `******`, so a host-written `**********` (10 stars) counts as the divider. The circuit then blocks further `***` and does not force a real divider.
   - Seen in 708000: the checker passes only because `split('******')` cuts the 10-star run inside the host's first sentence.
   - Fix: count only exact `******` tokens as the divider, and block longer star runs while the rule is on.
3. **The second response can be hollow.** Blocking only the first word piece of the first response stops verbatim copies. But after a forced divider the host often writes only its sign-off line again:
   - 708002 and 708007: "Let me know if you'd like ..."
   - 708010: the end phrase, repeated
   - 708001: the host respelled its blocked first word ("Ahaiku")

   The checker accepts these outputs. They are not two alternatives.
   - Fix (generic): block, at the start of the second part, any line opening of the first response, not only its first word. The last line of the first response is the most important one.
4. **A request sentence that ends in "?" is read as an information question.** In 708011 ("Two different answers, split by ******: what's a fun first-date activity in winter?") the instruction and the question share one sentence, so `nonreq_q` vetoes it.
   - This fails safe: the output was byte-identical to plain.
   - Possible fix: read the request guard at the `:` / clause boundary before the question.
5. **Minor.** A request for N≠2 alternatives with `******` still gets exactly 2. Out of scope, but it should be listed as a known limit.
6. **Part 5 saw these variants.** Change-log part 5 (turn scope, single-quote masking) was informed by running detection on all round-1 judge variants, these included. So the multi-turn negative 708017 is no longer an unseen test. The two-specific detector change (part 4) was made from the code reviews, before the variants were run.

## 2. Rule 13: USEFUL-GENERALLY: yes

What the tool computes: when the user asks for alternatives split by a delimiter, it guarantees the output is exactly two non-empty parts with different openings, separated by that delimiter.

Three settings outside the benchmark:
1. A/B copy variants: two subject lines, two ad taglines, two tones of the same email.
2. Offering the user a choice: two replies, titles or phrasings to pick from.
3. Machine-parsed output: batch generation or an evaluation harness that splits the reply on a fixed delimiter.

The interface should stay "N alternatives + delimiter D". The detector already reads generic words (two / versions / alternatives / separate) plus the literal delimiter.

## 3. Variants and results

Plain/implant columns are the official checker's prompt-strict result. "Wrote" is the number of steps the circuit wrote on.

| key | type | what it tests | latched | wrote | plain | implant | assessment |
|---|---|---|---|---|---|---|---|
| 708000 | POS | casual, abbreviations, literal divider | two | 4 | F | T | **Hollow (checker accident).** The host wrote `**********` (10 stars) in its first sentence; `k6` counted it, so no real divider was forced. Weakness 2. |
| 708001 | POS | word path "split ... six asterisks", no literal | two | 5 | F | T | Real second part, but with an "Ahaiku" spelling artifact. |
| 708002 | POS | "Use six asterisks to separate" (order changed) | two | 3 | F | T | Detected (part 4 fixed the order). Second part is only the copied sign-off line. **Hollow.** |
| 708003 | POS | instruction only in the system message | two | 3 | F | T | **Good.** Two distinct alternatives. |
| 708004 | POS | multi-turn follow-up with history | two | 4 | F | T | **Good.** |
| 708005 | POS | buried in a long message ("Could you ... ?") | two | 3 | T | T | Good. Plain already passed. |
| 708006 | POS | two + lowercase in one sentence | lower, two | 7 | T, T | T, T | Good. The implant blocked the host's `******` at the very start. |
| 708007 | POS | divider is the first token of the message | two | 3 | F | T | Second part is only the copied sign-off line. **Hollow.** |
| 708008 | POS | "divided by 6 stars" | two | 4 | F | T | **Good.** New slogans in the second part. |
| 708009 | POS | short numeric answer | two | 8 | T | T | **Regression.** Plain was correct (42 tokens). The implant loops `7 × 8 = 56` to 2048 tokens. Weakness 1. |
| 708010 | POS | two + end phrase | end, two | 17 | F, T | T, T | Second part is "Purr on. 🐾✨" ×5. **Degenerate.** Weakness 1. |
| 708011 | POS | instruction + question in one "?" sentence | - | 0 | F | F | **Missed, fails safe** (byte-identical to plain). Weakness 4. |
| 708012 | NEG | instruction inside double quotes + information question | - | 0 | - | - | Silent, byte-identical. |
| 708013 | NEG | `******` as a scene break, question | - | 0 | - | - | Silent, byte-identical. |
| 708014 | NEG | `******` as a censored swear word in pasted text | - | 0 | - | - | Silent, byte-identical. |
| 708015 | NEG | `******` as a password mask | - | 0 | - | - | Silent, byte-identical. |
| 708016 | NEG | "please don't split ... with ******" | - | 0 | - | - | Silent, byte-identical. |
| 708017 | NEG | rule asked in an earlier turn only | - | 0 | - | - | Silent, byte-identical. This item was seen by the part-5 change. |
| 708018 | NEG | "split a pizza between six star athletes" | - | 0 | - | - | Silent, byte-identical. |
| 708019 | NEG | "What do six asterisks mean in Markdown?" | - | 0 | - | - | Silent, byte-identical. |

### Trigger metrics

- 12 positives, 8 negatives.
- TP 11, FN 1, FP 0, TN 8.
- **Precision 1.00, recall 0.92, false-fire rate 0/8.**
- Silent outputs byte-identical to plain: 9/9.

### Generation metrics (positives)

- Official checker prompt-strict: plain 3/12, implant 11/12. 8 gained, 0 lost by the checker.
- Two real, different alternatives in the implant output: 6/12 (708001, 003, 004, 005, 006, 008).
- Hollow or accidental passes: 4/12 (708000, 002, 007, 010).
- Runaway regression: 1/12 (708009). Plain was correct; the implant ran to the token limit.
- Missed, fails safe: 1/12 (708011).

## 4. What to change before round 2

1. Count any non-whitespace word start in `w6`, not only Latin letters, and cap the end-of-turn block. This fixes 708009 and 708010 and removes the runaway on numeric, code and non-Latin second parts.
2. Count only exact `******` tokens as the divider. Block longer star runs while the rule is on. This fixes 708000.
3. Make anti-repeat block the openings of the first response's lines, at least its last line, at the start of the second part. This fixes the hollow sign-off copies in 708002, 708007 and 708010.
4. Optional: read the request guard at the clause boundary before a trailing question (708011).

Round 2 should use fresh variants, because part 5 saw these.
