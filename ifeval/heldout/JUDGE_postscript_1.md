# JUDGE_postscript_1: postscript rule-keeper (P.S. / P.P.S.)

Judge: independent reviewer, round 1. Protocol: bbh_generic/PROTOCOL.md rules 3, 5, 8, 10-13, applied to IFEval.
Files read: circuit.py, tables.py, gen_data.py (ps templates), NOTES.md (including change-log parts 4 and 5), run.py (row format), judge_run.sh.
Files not opened: no IFEval data, no fresh/ood files.
Variants: judge/postscript_variants.jsonl (keys 705000-705019). Results: judge/postscript_results_r1.md.

**VERDICT: PASS (conditional).** The rule may be used. The conditions and fixes are in section 5. Round 2 must use fresh variants, because
the current code was tuned partly on these items (see 4.3).

## 1. What the tool computes (current code)

- **Detector** (circuit.py line 289). It latches `ps` at the end of a clause if all of these are true:
  - the clause contains a postscript word (tables.PSW: postscript, post script, post-script, p.s, p.s., p.p.s);
  - the clause contains a verb that takes a P.S. as its object (tables.PSV: add, include, append, attach, put, insert, end, close,
    conclude, finish, sign, follow, tack, stick);
  - there is no clause negation, except "don't forget";
  - there is no explain/describe/define/discuss verb;
  - the clause passes the request guards: it is not an information question, has no third-person subject or third-person verb, and has a
    request marker.

  Cue words inside double or single quotes are masked. Clauses also end at 'and' and 'but'.
- **Marker selection.** `pps` latches on any mention of P.P.S. It has no guard. It has no effect alone: it only selects the forced marker
  after `ps` latches.
- **Turn scope.**
  - A latch from a user turn is dropped at the next user turn.
  - A latch from a system message persists through the conversation.
  - A cancel cue (this time, anymore, back to normal, ...) also drops the system latches.
- **Enforcement** (line 658 and the c5 programs). When the host's top token is end-of-turn and no P.S. has been seen in the answer, the
  emitter forces `sep 'P' '.S' '.'` (or `'P' '.P' '.S' '.'`). The host then writes the content of the postscript.
  - A P.S. that the host writes itself is detected: the lowercased token pair 'p' '.s' or '.p' '.s'. The rule then stays silent.

## 2. Code review: reward hacking and Python logic

- **Python logic at inference: none found.** The ps path is tensor gates on 0/1 wires, fixed vocabulary tables, n-gram compares and
  `torch.where`. The Python loops (`for k in self.L`, `pad`) run over static rule names or program literals, not over data. Regex is used
  only offline, in `tables.build`. The readouts are the disclosed LM-head readouts.
- **IFEval-specific strings: none.** PSW is the standard English name of the device and its abbreviations. PSV is a generic list of verbs
  that take a P.S. as object. Neither list holds a sentence or template.
- **Dev data.** The dev templates (gen_data.py lines 113-116, 9 phrasings, 3 of them `drv:` rewordings) always use the word "postscript"
  with an explicit marker. This is a narrow style. The detector is broader than the templates: bare "P.S." and lowercase "p.s." work, as
  the variants show.
- **Test leakage: none seen** from IFEval.
- **Fitting to the judges' items.** This is not IFEval fitting, but it must be disclosed. Change-log part 5 was made after the builder ran
  DETECTION on this judge's variants:
  - 'tack' and 'stick' were added to PSV. Both words come from my items 705011 and 705006.
  - 'write' was removed and the explain guard was added. These match my negatives 705012 and 705014.
  - The turn-scope change matches 705019.

  All of these are generic and reasonable changes. But the round-1 numbers below are optimistic, and round 2 must use new verbs and new
  structures.

### Weaknesses still present in the current code

1. **The verb satisfies the request-marker requirement by itself.** tables.REQMARK contains add, include and put. So the verb that makes a
   ps match (PSV) also counts as the request marker wherever it is in the clause. The marker check is therefore empty for most ps clauses.
   - This is the cause of the false latch on 705016 ("My grandfather always tried to add a P.S. to his letters").
   - Fix: for ps, count a PSV verb as the marker only in imperative position (`imp_any` already does this), or require
     you/please/must/should.
2. **The abbreviation dots are clause boundaries.** The tokenizer splits "P.S." as `' P' '.S' '.'` and "P.P.S." as `' P' '.P' '.S' '.'`.
   Each piece that contains '.' ends the clause.
   - A verb after the marker is never seen ("A P.S. after the main text is required": 705010).
   - A bare "P.P.S." (without the word "postscript") never latches, because '.P' resets the clause before '.S' (705003).
   - Fix (generic, also good for e.g. / U.S.): a '.'+letter piece with no space before it is not a boundary.
3. **A spaced host P.S. is not seen.** The host-P.S. detector needs adjacent 'p' '.s' tokens. In 705001 the host wrote "p. s. ..."
   (which the official checker accepts). The circuit did not see it and forced a second "p.s." block, and the host repeated its sentence.
   The score did not change, but the output has a duplicate.
   - Fix: also accept 'p' '.' 's' '.' and 'p' '.' ' s' '.'.
4. **Placement.** The forced P.S. goes where the host wants to end. In email tasks this is often after the assistant's own meta line
   (705011: "Let me know if you'd like to adjust the tone...\n\nP.S. ..."), so the P.S. is outside the letter. The checker does not
   care, but a user would. This is acceptable for a generic tool, but it is a known limit.
5. **Multi-turn is fail-safe by design.** "Same rules as before" is not carried over (disclosed in NOTES part 5). I agree with this choice.

## 3. Rule 13: is f(x) generally useful?

f(x) = "when the user asks for an after-note, make sure the text ends with a labelled postscript (P.S., or P.P.S. if asked), written by the
model, and do not add one when the text already has one". Nothing in this definition depends on IFEval.

**USEFUL-GENERALLY: yes.** Three settings outside the benchmark:
1. Email and cover-letter drafting assistants: "add a P.S. with my phone number".
2. Newsletters and marketing or fund-raising copy, where the P.S. is a standard place for a call to action.
3. Greeting, thank-you and holiday cards or notes, including a P.S. requested in a system message by an app that always wants a sign-off
   note.

## 4. Variants and results

Run on the current code, after parts 4 and 5. "Latched" means detection. "Strict ps" means the official checker for
detectable_content:postscript (705001 also checks lowercase). "Identical" means the implant output is byte-identical to plain.

### 4.1 Positives (12)

| key | what it tests | latched ps | strict ps plain -> implant | identical |
|---|---|---|---|---|
| 705000 | bare "P.S.", in the middle, 'include', extra constraint after | yes | T -> T | yes (host wrote P.S.) |
| 705001 | lowercase + postscript in one sentence | yes (+lower) | T,T -> T,T | no (duplicate P.S., weakness 3) |
| 705002 | rule only in the SYSTEM message, 'finish' | yes | T -> T | yes (host wrote P.S.) |
| 705003 | multi-turn follow-up, bare "P.P.S." | **no** (pps only) | F -> F | yes (fail-safe) |
| 705004 | buried in a long casual message, 'add a postscript' | yes | F -> **T** | no (P.S. forced) |
| 705005 | "needs a postscript too" (no verb) | **no** | T -> T | yes (fail-safe) |
| 705006 | casual lowercase chat, 'stick a P.S. at the bottom' | yes* | T -> T | yes |
| 705007 | instruction first, "pls add a p.s.!!!" | yes | T -> T | yes |
| 705008 | 'sign off with a P.S.', irrelevant sentence after | yes | T -> T | yes |
| 705009 | end of a long chain of instructions, 'append' | yes | F -> **T** | no (P.S. forced) |
| 705010 | marker first, passive "is required" | **no** (para latched, N=1, inert) | F -> F | yes (fail-safe) |
| 705011 | request in parentheses, 'tack on', 'could you' | yes* | F -> **T** | no (P.S. forced) |

\* 'stick' and 'tack' were added to PSV after the builder saw these items.

- Detection recall: 9/12 = 75%. Without the two contaminated items it would be about 7/12.
- All 3 misses fail safe: the output is byte-identical to plain.
- Strict ps accuracy: plain 7/12, implant 10/12. Gained 3, lost 0.
- No forced-token misses.

### 4.2 Negatives (8): the cue words are mentioned, but no P.S. is asked for

| key | what it tests | latched ps | identical |
|---|---|---|---|
| 705012 | essay ABOUT the postscript ('write' + 'postscript') | no | yes |
| 705013 | "Please don't add a P.S." | no | yes |
| 705014 | "Can you explain what a postscript at the end of a letter is for?" | no | yes |
| 705015 | instruction only inside double-quoted reported text | no | yes |
| 705016 | "My grandfather always tried to add a P.S. to his letters" | **yes (false latch)** | yes, by luck: the host's poem contained "A P.S." |
| 705017 | "What does P.P.S. mean, and when should I include one?" | no (pps only, inert) | yes |
| 705018 | "PostScript printer driver" (other sense) | no | yes |
| 705019 | earlier turn asked for a P.S., the follow-up cancels it | no | yes |

- False latches: 1/8 (12.5%).
- Output-changing false fires: 0/8. The one false latch would have forced an unwanted P.S. if the host had not written "P.S." in its poem.

### 4.3 Trigger precision and recall (latch level, these 20 items)

- Precision: 9 / (9 + 1) = 0.90.
- Recall: 9/12 = 0.75.
- Output level: 3 outputs changed on positives, all correct. 0 outputs changed on negatives. 0 strict losses.
- These numbers are optimistic. The builder ran detection on these items before the final code (part 5) and added 'tack' and 'stick' from
  them.

### 4.4 Predicted (from code reading before the run) vs observed

| | predicted | observed |
|---|---|---|
| False latches on negatives, code as first read | 5/8 (705012, 705013, 705014, 705016, 705019) | — |
| False latches on negatives, part 4/5 code | — | 1/8 (705016) |
| Misses on 705003, 705010 (abbreviation boundary) | yes | yes, both still present |
| Root cause of 705016 | — | the PSV verb also counts as the request marker (weakness 1) |

The part-4/5 changes fixed 4 of the 5 false latches.

## 5. Verdict

**PASS (conditional).** Reasons:

- No reward hacking, no Python logic at inference, no IFEval strings. The word lists are generic.
- On the positives the rule fires correctly or fails safe on all 12 items. It gains 3 strict items and loses 0. All misses produce
  byte-identical output.
- On the negatives no output changes. The false-latch rate is 1/8. The output was saved there only because the host wrote its own P.S.
  This is not "near zero" by design, so it is the main condition.

**Conditions before the final IFEval run, or confirm in round 2.** The rule should be narrowed, not dropped:

1. **Request marker for ps** (required). A PSV verb must not be its own request marker. Count it only in imperative position, or require
   you/please/modal/must. This closes the 705016 class ("<someone> always adds/tried to add a P.S. ...") for every subject noun, not only
   the pronoun list.
2. **Abbreviation dots** (recommended, generic). A '.'+letter token with no space before it should not be a clause boundary. Bare
   "P.P.S." and verb-after-marker requests will then work.
3. **Spaced host P.S.** (recommended). Treat "p. s." / "p. p. s." as host-written, to avoid duplicate postscripts.
4. **Round 2 must use new variants.** It should test verbs that are not in PSV ("throw in", "slap on", "leave a P.S.", "don't forget a
   P.S."), third-person habits with noun subjects, and P.S. in a system message combined with a later cancel cue.
