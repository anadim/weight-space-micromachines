# JUDGE end_phrase, round 2 (final verdict on the CURRENT circuit, change log part 7)

Tool: end_phrase. The reply must end with an exact phrase given in the prompt.
- Detector: an end verb (end/finish/conclude/close/sign off/wrap up/round off), or last/final + with/is/be/reads, in a requested clause. Then a quoted span, or a span after ':'.
- Enforcement: at end of turn, copy the span by pointer, then end the turn.

Files:
- Variants: `judge/end_phrase_variants_r2.jsonl` (keys 704100-704120, 12 positive, 9 negative).
- Results: `judge/end_phrase_results_r2.md`.
- Round 1: `JUDGE_end_phrase_1.md`.

## 1. Code findings (current circuit.py end-phrase block + tables ENDV, LASTW, BEW, REQMOD, INSTV, PHRASEW, CANCELW)

### No reward hacking, no leakage, no Python logic at inference
- Detection, capture, the undo and enforcement are all `torch.where` operations on 0/1 wires, pointer registers and fixed tables.
- The word lists are generic English: instruction verbs, modals, copulas, phrase nouns.
- There is no IFEval-specific string or template.
- The only Python loops are over the fixed rule names.

### Status of my round-1 / round-2 findings
| finding | status in part 7 |
|---|---|
| R1 bug 1: a two-word cue ('Wrap up with "X"') never counts as a command | Works in practice: 'wrap' is an instruction verb at the clause opening (`imp_any`), so 704100 now fires. The `imp_end` piece position is still read on ' up'; that is harmless here. |
| R2-2: wrong argument for openers that carry punctuation (' "--', ' "(') | Fixed by reverting to quote-only openers. Phrases that begin with punctuation now miss silently (704101, 704111): fail-safe. |
| R2-3: the question guard fired only on the '?' token, after the capture | Fixed: `undo = end_sent * nonreq_q` drops a phrase captured in a sentence that ends as a non-request question (704114, 704115 silent). |
| R1-5 / R2-4: 'Your last/final ... "X"' arms on your/answer | Fixed: last/final needs with/is/be/reads, and 'you/your' no longer counts as a request marker for this rule (704112, 704113, 704116, 704117 silent). |
| R2-5: 'this time' drops system latches | Fixed: removed from CANCELW. |

### Remaining weaknesses (all fail safe, or untested)
1. **Recall gap: an adverb before the end verb.**
   - "Always end your reply with "X"" misses (704107 here, and round-1 704005), because 'always' is neither an instruction verb nor in IMPPRE. This is a very common system-prompt phrasing.
   - Fix: add frequency/discourse adverbs (always, now, simply, remember to) to IMPPRE.
2. **Recall gap: polite question requests.**
   - "could you finish with "X"?" (704103) misses: REQMOD has no you + can/could/would. The undo is protected by the you+modal exception, but arming is not.
   - Fix: let you + modal count as a request for this rule, the same as the question guard does.
3. **Recall gap: phrases that begin with punctuation** ("-- Name", "(Coach K)", "...and scene") miss silently.
   - A correct fix starts the span at the opener token and strips only the quote character.
4. **Untested false-fire risk.** REQMOD 'need/must' anywhere in the clause + last/final + is/reads + a quote. Example: "I need the final chapter, which is "The Return", summarized" would arm. I rate this low-frequency.
5. **Cosmetic:** colon-form spans keep their leading space (' Pack light, ski hard.'). Enforcement strips it (`fmap=1`), so the output is correct.

## 2. Rule 13
f(prompt, reply) = if the user or system message requests that the reply end with an exact phrase, copy that phrase verbatim from the prompt and make sure the reply ends with it. The phrase is appended at end of turn unless the host already wrote it.

**USEFUL-GENERALLY: yes.** Three settings outside the benchmark:
1. Required sign-offs and disclaimers set in a system prompt (support bots, legal/medical "not advice" lines).
2. Protocol terminators that a downstream parser waits for ("END OF MESSAGE", "Over.", ticket footers).
3. Templates, letters and notices that need a fixed closing line.

## 3. Round-2 variants and results (part-7 code; strict = official checker)

| key | type | what it tests | latched (arg) | plain | implant | identical |
|---|---|---|---|---|---|---|
| 704100 | POS | "Wrap up with "That's the plan, team."" (two-word cue, no you/your) | end ✓ exact | F | **T** | no |
| 704101 | POS | system "Always close your replies with "-- Night Desk"" | silent (opener with '--' rejected; fail-safe) | T | T | yes |
| 704102 | POS | "Instead of a summary, end with "Enjoy the cup."" | end ✓, host already complied | T | T | yes |
| 704103 | POS | "could you finish with "Stay warm!"?" | silent (gap 2) | T | T | yes |
| 704104 | POS | "Close with: Pack light, ski hard." (colon, comma inside) | end ✓, host already complied | T | T | yes |
| 704105 | POS | multi-turn, "Make sure your message ends with "Grocery run at 6."" | end ✓ exact | F | **T** | no |
| 704106 | POS | system rule persisting through history | end ✓, host already complied | T | T | yes |
| 704107 | POS | system "Always end your reply with "Ref: HR-desk"" + 'this time' in the user turn | silent (gap 1, 'Always'; the cancel no longer fires) | T | T | yes |
| 704108 | POS | buried in a long message after "When you are done," | end ✓ exact | F | **T** | no |
| 704109 | POS | "Do not end with "Regards". End with "Talk soon, Priya" instead." | end ✓ (negated one skipped) | F | **T** | no |
| 704110 | POS | "The last line of your answer must be "Questions welcome."" | end ✓, host already complied | T | T | yes |
| 704111 | POS | "... and sign off with "(Coach K)"" | silent (gap 3; fail-safe) | F | F | yes |
| 704112 | NEG | "Your last answer said "the deadline is Friday" ..." | silent ✓ | - | - | yes |
| 704113 | NEG | "Your final answer was "42"; please double-check ..." | silent ✓ | - | - | yes |
| 704114 | NEG | "I noticed you end most of your replies with "Hope this helps!" Why is that?" | silent ✓ | - | - | yes |
| 704115 | NEG | "Why do you always end your messages with "Cheers"?" | silent ✓ | - | - | yes |
| 704116 | NEG | "... you have reached the final level when the screen shows "Victory Road"" | silent ✓ | - | - | yes |
| 704117 | NEG | "The final message in the group chat was "..."" | silent ✓ | - | - | yes |
| 704118 | NEG | "never close the email with "Sent from my phone"" | end silent ✓ (forbid latched, output unchanged) | - | - | yes |
| 704119 | NEG | "Translate this sign into Spanish: "Please end your visit with a smile."" | silent ✓ | - | - | yes |
| 704120 | NEG | earlier-turn end instruction, then an unrelated follow-up | silent ✓ (turn scope) | - | - | yes |

### Trigger statistics (end rule, round 2)
- **Recall:** 8/12 = 67% (round 1 on the part-5 code: 5/15 = 33%).
- **Argument accuracy:** 8/8 exact. There are no wrong arguments. Round 1 predicted wrong arguments on the part-6 code; those can no longer occur.
- **Precision:** 8/8 = 100%. False fires on negatives: 0/9 (round 1: 0/10).
- **Fail-safe:** all 4 missed positives and all 9 negatives are byte-identical to plain.
- **Prompt-level strict on positives:** plain 7/12, implant 11/12, so +4 gained and 0 lost. The only remaining failure (704111) fails identically in plain.
- The builder reports for all 275 round-2 items: 73.09 -> 85.09, 33 gained / 0 lost, silent outputs byte-identical (206/206).

### Contamination (disclosed)
- Part 7 was made after the builder ran DETECTION on these round-2 items. Six of my negatives (704112-704117) were exactly the risks I flagged in the review, and part 7 targets them. So 0/9 is a measurement on seen items.
- The fixes are generic linguistic constraints, not IFEval strings:
  - an instruction verb or modal is required;
  - last/final requires a copula or 'with';
  - the end-of-sentence question undo;
  - quote-only openers.
- My unseen predictions for the part-6 code were 6/9 false fires. Those mechanisms are now closed in code, and I checked this by reading code, not only by the run.
- An unbiased false-fire rate for this rule must come from the builder's fresh/OOD sets, which no judge has seen.

## 4. Verdict: PASS

- No reward hacking, no IFEval-specific material, no test leakage, no Python logic at inference.
- Every positive variant is either enforced correctly or fails safe with byte-identical output (12/12), and no wrong phrase is ever enforced.
- 0 false fires on the negatives in either round, after the part-5/part-7 generic guards.
- Net +4 / -0 on the round-2 positives and +4 / -0 on round 1.
- USEFUL-GENERALLY: yes.

Recommended, optional, recall only (each one is safe to skip):
- (a) add always/now/simply/remember-to as imperative prefixes;
- (b) count you + can/could/would as a request marker for this rule;
- (c) start the span at a punctuation-carrying opener token and strip only the quote.

Re-check the false-fire rate of (b) on the mentions set, because "you ... end ... with" descriptions were the main round-1 risk. Do not drop or narrow the rule further.
