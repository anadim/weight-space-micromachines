# JUDGE_uppercase_2: uppercase rule-keeper, round 2 (final)

Judge: same independent reviewer as round 1 (JUDGE_uppercase_1.md). Code reviewed: circuit.py and tables.py at 14:16
(change log parts 6 and 7).
Variants: judge/uppercase_variants_r2.jsonl (keys 702100-702118). Results: judge/uppercase_results_r2.md.
I did not open fresh, ood or IFEval data. I ran no models.

**VERDICT: PASS.**
- No reward hacking and no Python logic on data.
- 0 caps false fires on negatives: 0/8 in round 2, 0/18 over both rounds.
- Every miss failed safe (byte-identical to plain).
- 0 checker losses over both rounds.

Open issues:
- Recall is low: 5/11 this round, 12/26 over both rounds.
- One correct fire made the host loop (702104).
- Round 2 is also partly seen data: part 7 was shaped by detection on these items.

## 1. Code review of the current caps path

**Detector** (clause scope, committed at the sentence end):
- `capA = capletters · cw_pos · scope · ¬cw_neg · ¬cw_post · ¬TIMES · ¬startx · ¬FREQ2 · ¬PARTW`
- `capB = scope · LETTER · (lw_neg ∨ lw_post·flip) · ¬FREQ2 · ¬TIMES`
- Both are gated by the request guards.
- A request now needs one of:
  - an instruction verb from INSTV at the clause opening;
  - you / please / must / should / ...;
  - a negation at the opening of the sentence.

**Turn scope:** only the system message persists. Cancel cues are narrowed to anymore / no longer / back to normal / plain text /
regular text / fine again.

**Enforcement:** unchanged. The case transducer respells the host's top token through UPSEQ.

**Reward hacking, test fitting, leakage: none.**
- The new lists (INSTV, PARTW, NEG += stop/quit/cease, CANCELW) are generic English.
- PARTW (just / title / name / acronym) is a reasonable generic partial-scope veto, not tuned to IFEval wording.
- No IFEval strings or data are present.

**Python logic at inference: none.** Every decision is a tensor op on 0/1 wires and fixed tables. The readout is disclosed in the
docstring.

### Status of my earlier findings
- **Closed by part 7, with no false fire in the results:**
  - descriptions that open with a noun, gerund or proper noun (702111-702113);
  - partial scope (702114, 702118);
  - "this time of year" cancelling a system rule (702101 now fires);
  - "stop using capitals" re-latching caps (702115 caps silent).
- **Closed earlier:**
  - F2: a possessive apostrophe opened a quote mask (closed in part 6);
  - F5: `capB` scope (now needs "letters").
- **Regressed:** F3, verbless fragments. The INSTV requirement drops them again:
  - "Capital letters only, start to finish." (702105);
  - ", entirely in capital letters." in a sentence that opened with "Draft" (702107);
  - "ALL CAPS, please: ..." (702108). Here "please" sits in the next clause, so the F1 phrase fix never gets exercised.

  All three fail safe. Fix: inherit the request marker from the sentence, i.e. accept a case clause when its sentence opened with
  an instruction verb or carries please/you.
- **Still open, fail-safe:**
  - an apostrophe before a non-quote after a space ("a '90s-style jingle") masks the rest of the line (702100);
  - a negation after the case word in the same clause, e.g. "only - no exceptions -" (702102);
  - FREQ2 in the same clause, e.g. "in all caps in under 60 words" (702103).
- **New, enforcement:** on 702104 the rule fired correctly, then the host looped ("- BAGS" repeated until the 2048-token cap).
  The plain output was 61 tokens and also failed, so the score is unchanged, but the output is worse. NOTES already lists loops
  in caps mode. A generic remedy is to block an exact repeat of the previous line while a case rule is active.
- **Shared detector, other rule (report to the lowercase judge):** in 702115 the user revokes a system caps rule with
  "You can stop using capitals now - regular text is fine." Caps correctly stays silent. But "stop" is now a negation, so `lowB`
  latches LOWERCASE and changes the output, although the user asked for normal text. Suggested fix: a cancel cue (regular text,
  back to normal, plain text) in the same sentence should also veto `lowB`/`capB`.

## 2. Rule 13
**The function.** When the user asks for the whole reply in capital letters, the tool respells every emitted token into
uppercase inside the decode loop. The host continues from its own uppercase text.

**Three settings outside the benchmark:**
1. Telegraph, radio and aviation message formats (NOTAM, METAR, TELEX).
2. Printed signs, placards, banners and headlines.
3. Legacy forms, terminals and mainframe or EDI fields that accept only uppercase.

**USEFUL-GENERALLY: yes** (moderately; it is close to a post-processing `.upper()`, but done in-context).

## 3. Round-2 variants and results (current code)
"silent=" means the implant output was byte-identical to plain.
POS = positive item, NEG = negative item.

| key | type | what it tests | latched | strict plain → implant | outcome |
|---|---|---|---|---|---|
| 702100 | POS | "a '90s-style jingle ... in all caps" | - | F → F | miss, fail-safe (apostrophe mask) |
| 702101 | POS | caps rule in the SYSTEM message, user writes "this time of year" | caps | T → T (identical) | correct |
| 702102 | POS | "Uppercase letters only - no exceptions -" | - | T → T | miss, fail-safe |
| 702103 | POS | "in all caps in under 60 words" | - | F → F | miss, fail-safe (FREQ2) |
| 702104 | POS | "Lowercase letters are not permitted anywhere in your reply" | caps | F → F | correct fire; **host loops to 2048 tokens** |
| 702105 | POS | "Capital letters only, start to finish." | - | T → T | miss, fail-safe (F3 regression) |
| 702106 | POS | multi-turn, "type them in uppercase this time" | caps | F → F (identical) | correct fire; plain already caps; checker (langdetect) fails both |
| 702107 | POS | "Students' essays ..." + ", entirely in capital letters." | - | F → F | miss, fail-safe (F3 regression) |
| 702108 | POS | "ALL CAPS, please: ..." | - | T → T | miss, fail-safe (F3 regression) |
| 702109 | POS | buried in a long message after a closed 'single-quoted' title | caps | F → **T** | gain |
| 702110 | POS | "Respond ... in capitals, nothing lowercase anywhere." | caps | T → T (identical) | correct |
| 702111 | NEG | "Typing in all caps is considered rude ..." | - | silent= | correct |
| 702112 | NEG | "Teletype machines printed only capital letters." | - | silent= | correct |
| 702113 | NEG | "Grandpa always emails in all caps." | - | silent= | correct |
| 702114 | NEG | "Don't lowercase any of the brand names in your reply." | - | silent= | correct |
| 702115 | NEG | revocation of the system caps rule ("stop using capitals ... regular text") | lower | changed | caps correct (silent); the LOWERCASE sibling false-fires |
| 702116 | NEG | instruction only inside 'single quotes' | - | silent= | correct |
| 702117 | NEG | "Capitalize the first letter of every sentence" | - | silent= | correct |
| 702118 | NEG | "put just the final word in capital letters" | - | silent= | correct |

**Trigger metrics for the caps latch:**

| measure | round 2 | rounds 1 and 2 combined |
|---|---|---|
| Precision | 5/5 = 100% | 12/12 = 100% |
| Recall | 5/11 = 45% | 12/26 = 46% |
| False fires on negatives | 0/8 | 0/18 |

**End-to-end, positives (strict):**

| measure | round 2 | round 1 |
|---|---|---|
| Plain → implant | 5/11 → 6/11 | 6/15 → 8/15 |
| Gained / lost | 1 gained, 0 lost | 2 gained, 0 lost |

**Other outcomes:**
- Every miss was byte-identical to plain.
- One fire degenerated (702104).
- On round-2 negatives, 7/8 outputs were byte-identical. The 8th (702115) was changed by the lowercase rule, not caps.

## 4. Verdict: PASS
Why PASS:
- The code is clean: no IFEval fitting, no Python logic on data, and the readout is disclosed.
- Detection precision is 100% over 26 positives and 18 negatives in two rounds, with zero caps false fires.
- The negatives include every false-fire pattern I predicted from the code: descriptions, proper-noun subjects, partial scope, a
  frequency limit, revocation, quoted text, and other senses of capital/caps.
- Every miss is silent and byte-identical to plain, and the checker never lost a point.

Caveats:
1. Both rounds' items were seen at the detection level before the generation runs. The zero false-fire result is still
   optimistic, so it should be confirmed on real held-out data (fresh/ood).
2. Recall is about 46%. The rule is conservative. The cheapest generic gain is a sentence-level request marker for verbless
   case clauses (F3). After that, the line-long apostrophe mask, and allowing FREQ2 / a trailing negation when they belong to
   another constraint.
3. Caps mode can make the 0.6B host loop (702104). Add an exact-repeat-line block while a case rule is active.
4. Send the 702115 lowercase false fire (revocation read as lowercase) to the lowercase rule's owner.

The rule is neither dropped nor narrowed. Keep it as is. The fixes above would raise recall and are not needed for safety.
