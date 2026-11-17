# JUDGE no_comma, round 2 (final)

Judge: the same independent reviewer as round 1 (JUDGE_no_comma_1.md).
Code reviewed: the CURRENT circuit.py / tables.py (NOTES.md change log part 7).
Results: judge/no_comma_results_r2.md (part-7 code).
Variants: judge/no_comma_variants_r2.jsonl, keys 700100-700115 (9 positives, 7 negatives).

## VERDICT: PASS (precise, fail-safe, low recall)

The PASS criteria are met:
- There is no reward hacking, no IFEval-specific wording and no Python logic on data.
- Every positive is either correct (enforced, strict pass) or silent with output byte-identical to plain.
- The false-fire rate on the negatives is 0/7.

Two qualifications apply:
- **Recall on new phrasings is low:** 2/9 in this round.
- **The negatives are partly seen:** the builder ran detection on these items while writing part 7, and several part-7 patches match my negatives directly. So 0/7 is an optimistic number for precision. The real test is the one-time IFEval run.

Do not drop the rule. The optional recall fixes (section 4) are not conditions for the PASS.

## 1. Code findings (part 7)

**Python logic and test data.** The path is still tensor ops on 0/1 wires plus fixed tables.
- `_cue_neg` has `if name not in R`. This sets up a register lazily, keyed by the rule name, not by data, so it is acceptable.
- The other Python loops go over fixed rule or register lists.
- I found no IFEval strings and no test-data leakage.

**Enforcement.** It did not change: COMMASET mask, argmax readout, host's best comma-free token. ffail was 0 on all items.

**The detector is now much stricter:**
- An imperative opening needs a verb from INSTV.
- An opening negation counts only at the start of a sentence.
- Topic sentences (explain/describe/tell + why/how/what), reported claims and code requests are excluded.
- Clause rules commit at the end of the sentence, so a later non-request '?' cancels them.
- Data after a colon is masked.
- 'never mind' is not a negation.
- REMOVEV ("don't remove/change ... commas") vetoes the rule.
- Cancel cues are narrowed. 'as usual', 'this time' and 'go back' are no longer cancels, which fixes the round-2 risk I flagged.

Concerns (recall only; all fail safe):
- **R1. The topic veto covers the whole sentence.** In 700101 ("Explain how lighthouses work and skip the commas."), "Explain how" vetoes "skip the commas", which is in a separate clause after 'and'. Common phrasings will be missed, for example "Describe what X is, without commas." This is the largest recall hole.
- **R2. A passive or label instruction with no marker is missed.** Examples: "Commas are not allowed." (700108), "House style for this chat: no commas in any reply." (700102, a system message), "One rule: commas are banned." (700104). Dev phrasings with "task" or "cannot" still work.
- **R3. A negation in the middle of a sentence with no INSTV opening is missed.** Example: "...so dont use commas" (700106, casual chat).
- **R4. REMOVEV is a blunt veto.** For example, "Remove the emojis and don't use commas" would be vetoed if both are in one clause. This fails safe.
- **R5. Rule-5 note.** 'never mind', the REMOVEV list and the "explain why" topic exclusion each match one of my round-2 negatives (700113, 700112, 700111). Each is stated generically, and each only removes fires (fail-safe direction). But they were added after detection ran on these items, so the negatives are not an unseen test.

## 2. Rule 13

f(x): if the instructions (a user message or a system message) ask that the reply contain no comma characters, the reply contains no ','. The tool does this by replacing each comma-bearing token the host would choose with the host's best comma-free alternative. Nothing else changes.

Settings outside the benchmark:
1. CSV/TSV cell values, and arguments or tag fields that a parser splits on commas.
2. Legacy SMS, pager or telex gateways, and TTS pipelines that split on commas.
3. House-style or constrained-writing rules.

USEFUL-GENERALLY: yes. The value is modest, and the mechanism is generic.

## 3. Variants and results (round 2)

Columns: latch = nc latched; w = steps written; P / I = official no_comma strict pass, plain / implant; id = implant output byte-identical to plain.

| key | type | what it tests | latch | w | P / I | id | outcome |
|---|---|---|---|---|---|---|---|
| 700100 | POS | "I'd prefer the reply without commas." | no | 0 | F / F | yes | FN, fail-safe (as predicted) |
| 700101 | POS | "Explain how lighthouses work and skip the commas." | no | 0 | F / F | yes | FN, fail-safe (R1 topic veto) |
| 700102 | POS | system "House style...: no commas in any reply" + user "As usual, ..." | no | 0 | F / F | yes | FN, fail-safe (R2: the system rule was never latched) |
| 700103 | POS | system "Never use commas in your replies." + multi-turn | yes | 9 | F / T | no | TP, gained (system persistence works) |
| 700104 | POS | "One rule: commas are banned." | no | 0 | F / F | yes | FN, fail-safe (as predicted) |
| 700105 | POS | "Without a single comma, describe ..." | yes | 4 | F / T | no | TP, gained |
| 700106 | POS | casual "...so dont use commas" | no | 0 | F / F | yes | FN, fail-safe (R3) |
| 700107 | POS | "Answer in all caps with no commas: what is a black hole?" (+caps) | no | 0 | T,T / T,T | yes | FN, fail-safe (the host passed both alone) |
| 700108 | POS | numbered list "2. Commas are not allowed." | no | 0 | F / F | yes | FN, fail-safe (R2) |
| 700109 | NEG | "Telegrams never contained commas." | no | 0 | - | yes | TN |
| 700110 | NEG | "Grandma ... and never uses commas." | no | 0 | - | yes | TN |
| 700111 | NEG | Pythagorean comma "is not audible" | no | 0 | - | yes | TN |
| 700112 | NEG | "don't remove any commas" (must keep commas) | no | 0 | - | yes | TN |
| 700113 | NEG | "Never mind the commas; just fix the spelling" | no | 0 | - | yes | TN |
| 700114 | NEG | reported rule after a colon, then a question | no | 0 | - | yes | TN |
| 700115 | NEG | cat named "Comma" ... "is not allowed" | no | 0 | - | yes | TN |

Round-2 trigger statistics (nc latch):
- Positives: TP 2, FN 7. Recall 2/9 = 22%.
- Negatives: FP 0, TN 7. False-fire rate 0/7.
- Precision 2/2 = 100%.
- Effect on the official no_comma check, positives: plain 1/9, implant 3/9. 2 gained, 0 lost.
- All 7 misses and all 7 negatives give output byte-identical to plain.

Two rounds together (each round ran on the code of its time):
- Round 1 (part 5): recall 6/14, false fires 1/7.
- Round 2 (part 7): recall 2/9, false fires 0/7.
- In total: 8 TP, 15 FN, 1 FP on 37 items. Every positive was correct or fail-safe, and no prompt was lost.

Each round traded recall for precision. Overall the detector is now precise but catches mainly explicit forms: an instruction verb or negation at the start of a sentence, or you/must/should/cannot/task. It misses many natural paraphrases.

## 4. Reasons for the verdict, and optional improvements

PASS because:
- Every brief criterion holds: no hacking or Python logic, positives correct or fail-safe (9/9), 0/7 false fires, and USEFUL-GENERALLY yes.
- The enforcement is exact.
- The tool cannot make a reply worse when it does not latch (byte-identical outputs).

This PASS says the tool is safe. It does not say the tool has high coverage. The expected gain on unseen data is modest.

Optional recall fixes (each is generic; check them on new items, not on these):
1. Make the topic veto clause-scoped (R1). "Explain how X works" should not veto a separate clause "and skip the commas".
2. Accept a standalone passive prohibition when the sentence has no other subject: "Commas are not allowed/permitted/banned." (R2). Run it through the same third-person and description guards.
3. Accept a negation plus an INSTV verb anywhere in a clause, as in "dont use commas" or "so never use commas" (R3).
