# JUDGE_postscript_3: postscript rule-keeper, final verdict on the FROZEN circuit

- **Code judged:** the frozen circuit, frozen/circuit.py md5 cd15769c..., frozen/tables.py md5 78f6181b..., NOTES parts 8-9.
- **Items:** judge/postscript_variants_r2b.jsonl (705200-705205, fresh), plus all round-2 items 705100-705119.
- **Results:** judge/postscript_results_r2b.md.
- **Earlier rounds:** JUDGE_postscript_1.md and JUDGE_postscript_2.md.

**VERDICT: PASS.**

## 1. What changed since round 2

- **My required fix was replaced by a different fix.** The required fix was "'.S' must not be a boundary". The builder showed that this
  merges two instructions when the marker has no closing '.' ("... starting with P.P.S Answer in all caps."). That is a valid reason, so
  '.S' stays the end of the marker. The builder added a generic question-opening guard instead: a sentence that opens with an
  auxiliary/modal not followed by 'you' is a question. With it, 705114 ("Should I include a P.S. ...?") is silent. I accept this
  replacement.
- **Two topic guards were added (part 9):**
  - 'wonder/wondered/asked + whether' is a topic (705202).
  - Reported 'X say ...' is a reported claim, except 'do not say' and imperative 'Say' (705115).

  Both guards are generic. Both were written after the builder saw my items, so these two results are confirmations of the fix, not
  blind tests.
- **Code review of the frozen ps path:** no Python logic at inference, no IFEval strings. Tables and gates only, as in rounds 1-2.

## 2. Results on the frozen circuit (r2 + r2b, 26 items)

### Positives (15)

- **Latched (10/15):** 705103, 705105, 705106, 705108, 705109, 705110, 705111, 705203, 705204, 705205. The three new
  modal-plus-'you' requests ("Could you add...", "Will you please include...", "Would you mind adding...") all fire, so the new question
  guard does not block real requests.
- **Missed (5/15), all byte-identical to plain:** 705100 "throw in", 705101 "leave", 705102 verb after the marker, 705104 "always
  include" in a system message, 705107 bare "PS".
- **Strict ps:** plain 14/15, implant 14/15. 0 gained, 0 lost. The host already complied on these items.

### Negatives (11)

- **ps latched: 0/11.** This includes 705114, 705115, 705200 ("Would it be weird to add a P.S. ...?"), 705201 ("Is it okay to put a
  P.P.S. after a P.S. ...?") and 705202 ("My manager wondered whether we should add a P.S.").
- **pps latched on 705201 only.** It is inert, because pps only selects the marker after ps latches.
- **Every negative output is byte-identical to plain.**

### Trigger precision and recall (frozen code, 26 items)

- **Precision:** 10/10 = 1.00.
- **Recall:** 10/15 = 0.67.
- **Output changes on negatives:** 0/11.
- **Strict losses:** 0.

### Earlier rounds

Round-1 items (part-5 code): +3 strict gains, 0 losses. Over all rounds: 0/19 negatives changed output, 0 strict losses on any positive,
and every miss failed safe.

## 3. Residual limits (recall only; none is a blocking issue)

- **Verbs outside PSV:** "throw in", "leave", "slap on". The detector misses these; the result is fail-safe.
- **A verb after the marker** is not read ("a P.S. at the end would be great"). This is the cost of keeping '.S' as a boundary; the
  result is fail-safe.
- **An adverb before the verb:** "always include a P.S." in a system prompt does not latch. Adding frequency adverbs to IMPPRE would fix
  this.
- **Bare "PS"** without dots is not a cue.
- **A spaced host "p. s."** is not detected, so a duplicate P.S. is possible. This was seen once in round 1, and the score did not change.
- **Placement:** the forced P.S. comes after the host's last text, which can be a meta line rather than the letter body.
- **Diagnostic oddity, not a correctness issue:** 705108 reports "circuit wrote on 927 steps" while the output is byte-identical with
  17 tokens. The 'fires' counter seems to count steps outside the answer, or outside this row. Please check the counter before quoting
  write counts.

## 4. Rule 13

**USEFUL-GENERALLY: yes.** f(x) = "when asked, end the reply with a labelled after-note (P.S. / P.P.S.), unless one is already there".
Three settings outside the benchmark:

1. Email and cover-letter assistants.
2. Fund-raising and marketing letters.
3. Cards and notes, where the P.S. is requested in an app's system message.

## 5. Verdict

**PASS.**

- No reward hacking, no Python logic at inference, no IFEval-specific strings or templates.
- On the frozen code: 0 false latches on 11 fresh and round-2 negatives (precision 1.00). Positives fire or fail safe on 15/15 items, and
  there are 0 strict losses.
- The round-2 condition is met by an accepted alternative: the question-opening guard in place of the '.S' non-boundary.

**Disclosure for the final report:** parts 5-9 were shaped by running detection on this judge's items (rounds 1, 2 and 2b). The judge
numbers are therefore partly in-sample. The held-out IFEval run is the real test.
