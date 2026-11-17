# JUDGE: repeat_request, round 2 (final verdict on the current code)

- **Tool:** repeat_request. It echoes the user's request verbatim at the start of the reply, then lets the host answer.
- **Code reviewed:** `circuit.py` and `tables.py` of 2026-11-17 14:16 (NOTES.md change log parts 6 and 7).
- **Round-2 items:** `judge/repeat_request_variants_r2.jsonl`, keys 707100-707121 (12 positive, 10 negative).
- **Results:** `judge/repeat_request_results_r2.md`, run on the part-7 code.
- **Round 1:** `JUDGE_repeat_request_1.md` (PASS conditional, on the part-5 code).
- **Seen-data caveat:** the builder ran detection on these r2 items while writing part 7, so they are partly seen. Both false fires below survived that pass anyway.

## 1. Code review (current code)

### Detection (circuit.py about lines 448-464)

The rule fires at a sentence end in a user turn when the sentence contains all of:
- a REPEATV word (repeat / restate / copy / echo / reproduce / recite / reiterate);
- a REQTERM word (request / prompt / question / query / instruction / text / sentence / line / message / task / ask);
- a direction or verbatim word, from either list:
  - VERBATIM (verbatim / exactly / exact / unchanged / as-is / word for word / word-for-word / as it is / as written / without change(s) / without changing). 'change' alone no longer counts.
  - ABOVED (above / previous / preceding / earlier / prior) or BELOWD (below / following / follows / next / underneath / beneath).

It also needs all of:
- the request guards: not an information question, no third-person subject or verb, not a topic, reported or code sentence;
- a sentence-level request marker: an instruction verb opening a clause (INSTV), or you / please / must / ...;
- not in a system turn;
- no negation within 4 tokens before the repeat verb;
- not 'your' + REQTERM;
- first hit in the turn.

Masking: quoted text (double and single quotes), backtick spans and colon-introduced data lines are masked.

### Span and enforcement

These are unchanged since round 1:
- 'above' (default; 'above' wins over 'below'): the span is from the message start to the end of the previous sentence.
- 'below': the span is the next line through the end of the message.
- The span is copied by pointer at the answer start, then `\n\n`.
- Forced-token misses: 0 on every item.

### Python logic / reward hacking

None found:
- All rule logic is `torch.where` / `gather` on 0/1 wires, pointer registers and fixed tables.
- The Python loops run over fixed key lists.
- Regex is used only in the offline table build.
- The token ids used are chat-template structure.
- The word lists are generic English (verbs, nouns, deictics). There are no IFEval sentences or templates.
- The part-7 changes are generic and motivated by judge items, not by test data.

### Defects found in the current code

1. **The exclusion for 'your previous message' does not cover that phrase** (circuit.py line 449). The veto only fires when the token immediately before the REQTERM is a YOUW word, i.e. "your message". In "Repeat your previous message exactly", the word before 'message' is 'previous', so the veto is missed. 'previous' is also in ABOVED, so the rule fires and copies the earlier sentences of the current user message ("Cool."). Confirmed by 707116.
2. **"copy the text below into ..." still fires.** This is a reformat request, not an echo request. 'text' is a REQTERM, 'below' gives the direction, and 'Copy' is an instruction verb. Confirmed by 707115. The implant output lost the bullet list the user asked for; plain made the list correctly.
3. **Host degeneration after the forced copy (707100).** After the copy, the host fell into "- Towel" repeated until the 2048-token limit. Plain stopped after 26 tokens. The checker passes this item, but a real user gets a broken answer. This is the host's behaviour on a new prefix, not a circuit write; the implant is silent after the copy. Still, the change of prefix caused it.
4. **The round-1 observation still holds.** After the copy, the host often echoes the instruction sentence instead of answering (707101, 707103, 707104, 707111). The checker passes these. The implant guarantees the read-back, not the answer.

## 2. Rule 13: is f(x) generally useful?

- **f(x):** given a message that asks for its request to be echoed verbatim, emit an exact copy of the referenced span of that message at the start of the reply. The span is the text before the instruction, or the text after it for 'below/following'.
- **USEFUL-GENERALLY: yes.** Three settings:
  1. Read-back and confirmation protocols: pharmacy or order confirmations, ATC-style read-back, support-ticket acknowledgement.
  2. Email and forum replies that quote the question before answering.
  3. Audit logs and transcripts that must store the exact request next to the response.

## 3. Round-2 variants and results (part-7 code)

| key | type | tests | fired / span | plain | implant | outcome |
|---|---|---|---|---|---|---|
| 707100 | POS | reproduce + prompt + unchanged, "Make sure you ..." | yes / correct | F | T | gain (but host loops to 2048 tokens after the copy) |
| 707101 | POS | '...', '!!!', '--', word-for-word, 'pls' | yes / correct | F | T | gain |
| 707102 | POS | instruction in the middle of a long message, text after it | yes / correct | T | T | same |
| 707103 | POS | repeat + no commas in one sentence | yes / correct (+ nc) | F,T | T,T | gain |
| 707104 | POS | repeat + all caps in one sentence | yes / correct (+ caps) | F,T | T,T | gain |
| 707105 | POS | multi-turn follow-up, 'below' + newline | yes / correct | T | T | same |
| 707106 | POS | multi-paragraph span | yes / correct | F | T | gain |
| 707107 | POS | instruction only in a system message | silent | F | F | fail-safe (byte-identical) |
| 707108 | POS | 'following' + colon on the same line | silent | T | T | fail-safe (byte-identical) |
| 707109 | POS | 'read back' (verb not in list) | silent | T | T | fail-safe (byte-identical) |
| 707110 | POS | 'everything I wrote above' (no REQTERM) | silent | T | T | fail-safe (byte-identical) |
| 707111 | POS | 'kindly' + 'as it is' + ';' | yes / correct | F | T | gain |
| 707112 | NEG | "Pilots always repeat the controller's instruction back word for word." | silent | - | - | OK, identical |
| 707113 | NEG | "Parrots can copy a sentence exactly ..." | silent | - | - | OK, identical |
| 707114 | NEG | "Copy the text but change the tone" (rewrite) | silent | - | - | OK, identical |
| 707115 | NEG | "Copy the text below into bullet points:" + notes | **fired**, span = the notes | - | - | **FALSE FIRE**: the bullet list is lost |
| 707116 | NEG | multi-turn "Repeat your previous message exactly, but in French." | **fired**, span = "Cool." | - | - | **FALSE FIRE**: "Cool." prepended |
| 707117 | NEG | story: the parrot "says it will repeat every sentence you tell it word for word" | silent | - | - | OK, identical |
| 707118 | NEG | "How do I make it repeat the user's question verbatim ...?" | silent | - | - | OK, identical |
| 707119 | NEG | "There is no need to repeat my question above word for word" | silent | - | - | OK, identical (negation window 4) |
| 707120 | NEG | "Please don't bother restating my question exactly" | silent | - | - | OK, identical |
| 707121 | NEG | pasted email containing the instruction, no quote marks | silent | - | - | OK, identical (colon data mask) |

### Trigger metrics

| measure | result |
|---|---|
| Recall: fired on positives | 8/12 (67%), all 8 with the correct span |
| Correct or fail-safe on positives | 12/12 (no wrong outputs) |
| False fires on negatives | 2/10 (20%) |
| Precision: fires that were on positives | 8/10 (80%) |
| Silent items byte-identical to plain | 12/12 |
| Prompt-strict on positives, plain → implant | 5/12 → 11/12 (+6 / -0) |

### Round-1 fixes confirmed

- A system-message repeat is now silent and fail-safe (707107). In round 1 it copied the system text.
- 'change' without 'without' no longer counts (707114).
- Noun-subject descriptions are now silent (707112, 707113).
- The longer negation window works (707119).
- Pasted data after a colon is masked (707121).

## 4. Verdict: FAIL. Narrow the rule minimally; do not drop it.

What holds:
- No reward hacking and no Python logic.
- Positives are 12/12 correct or fail-safe, with +6 / -0 on strict.
- Every silent output is byte-identical to plain.
- The function is generally useful.

Why it fails:
- The false-fire rate is 2/10 on negatives. That is not near zero.
- Both false fires produce visible wrong output, not silence. One drops the user's requested list (707115). The other prepends "Cool." (707116).
- Both survived the part-7 pass, which had seen these items. One of them (707116) is exactly the case that part 7 claims to exclude.

Minimal narrowing (both are generic, and neither should cost recall on real echo requests):
1. **Fix the 'your ... message' veto.** Veto when a YOUW word (your / you) appears within 3 tokens before the REQTERM (e.g. "your previous message", "your last answer"). Alternatively, take 'previous / earlier / prior' out of ABOVED for this rule and keep only above / preceding.
2. **Treat "copy/reproduce X into/as/in Y" as a transfer or reformat request, not an echo.** Veto when the clause containing the repeat verb has a destination preposition (into / onto / as / to a) after the REQTERM. A narrower alternative: in 'below' mode, require a VERBATIM word, since the direction word alone is too weak there.

Re-test plan:
- Rerun 707115 and 707116 plus the 12 r2 positives.
- Add a few fresh negatives in the same families: "copy this sentence into Spanish", "reproduce the table below as CSV", "repeat your last answer exactly".
- I expect PASS once both false fires are silent with no positive regressions.

Also flag the 707100 host loop to the builder as a quality risk. It is not a verdict item, because the checker passes it.
