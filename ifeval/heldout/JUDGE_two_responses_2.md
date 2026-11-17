# JUDGE two_responses, round 2 (final)

Judge: independent reviewer (subagent). Tool: two_responses.
Code judged: the CURRENT circuit.py / tables.py (NOTES.md change log through part 7).
Variants: judge/two_responses_variants_r2.jsonl (keys 708100-708120). Results: judge/two_responses_results_r2.md. Round 1: JUDGE_two_responses_1.md.

## Verdict: PASS-conditional. Keep the rule. Do not drop it and do not narrow the detector.

- **Why it passes.**
  - No reward hacking and no Python logic on data.
  - The round-1 FAIL cause is fixed: the runaway end-of-turn block. No positive hit the length limit.
  - Every implant change in r2 either matches or improves on plain (6 checker gains, 0 losses).
  - Detection: precision 10/10, false fires 0/9, every silent output byte-identical to plain.
  - Both positive misses fail safe (identical to plain).
- **Conditions** (enforcement quality; not blocking, but needed before the rule's score can be called genuine):
  - C1. Only a `******` the host writes as a standalone divider should count as the divider (definition in section 4).
  - C2. Fix the anti-repeat window that part 6 broke.
  - C3. Block the first response's line openings, not only its first word.
- **Two caveats on the numbers.**
  - Part 7 was shaped by running detection on these r2 items, and all 9 negatives were ones I had predicted to false-fire under part 6. So 0/9 is optimistic for unseen text.
  - 2 of the 6 checker gains are hollow (details below).

## 1. Code findings (current code)

### Detector
- `m_two = se_six * max(SEPV, TWOW) * sentence_end * rqs * reqs * (1 - n_two)`, where:
  - `se_six`: the literal `******`, or 6 + asterisk/star. Part 7 masks the literal in double quotes, single quotes, backticks and data-after-colon, and counts star runs across tokens.
  - `rqs`: not an information question, not third-person, not a topic sentence (explain/tell + why/how), not reported speech, not code.
  - `reqs`: a request marker. That is an instruction VERB from the INSTV list at the clause opening, an opening negation, or you/please/must/should/... anywhere.
  - `n_two`: the per-clause negation of the cue.
- All of these are generic English classes, and all are tensor gates.
- The part-6 hole (any non-subject word at the clause opening counted as an imperative) is closed by INSTV.

### Enforcement
At the first end-of-turn, if no divider has been written yet (`k6 == 0`), the emitter forces `\n\n******\n\n`. After that:
- End-of-turn is blocked until the second part has 5 words, now counting ANY non-whitespace word. The block is capped at 40 tokens (`n6`). This fixes round-1 weakness 1.
- `***` tokens are blocked after the divider, and before it while the answer has fewer than 5 words.
- The first word piece of the first response is blocked at the start of the second part.

### Reward hacking / test fitting
None found.
- The only IFEval-shaped string is `******`, the tool's own delimiter.
- The word lists (SEPV, TWOW, ASTER, INSTV, NEG) are generic.
- No test data is used.

### Python logic at inference
None found. Only `torch.where` / `max` / `clamp` on 0/1 wires, table lookups and counters. Loops run over static lists, and regex runs only at table build time.

### Remaining weaknesses (each seen in r2)
1. **Any `******` the host writes counts as the divider.** Round-1 weakness 2 is not fixed: `k6` counts any answer token containing `******`, even in prose.
   - 708102: the host wrote "each with a `******` placeholder" in its intro sentence. `k6` became 1, the circuit then blocked the later `******` in the code, and no real divider was forced.
   - The checker passes because it splits that intro sentence. The output has no second response, so this is a hollow pass. Plain failed here because it had 3 dividers.
2. **Anti-repeat regression from part 6 (new).** The window `two_on` needs `w6 == 0`. Since part 6, `w6` counts any non-whitespace word, so a leading list number "1." uses up the window before the first real word.
   - 708107: the second part is a verbatim copy of the first ("1. A bike-repair shop ..." twice), so the checker fails ("responses must differ").
   - Plain also failed, but the output changed, so this is not a fail-safe miss.
   - Fix: anchor the window on the first LETTERS word (as before part 6), or skip numbering/bullet tokens.
3. **Hollow second parts.** Round-1 weakness 3 is not fixed.
   - 708103: the second part is only the host's sign-off ("Let me know if you'd like ...").
   - 708101 and 708105: the second part re-uses a line of the first, with partly new content.
4. **Request recall gaps.** Both fail safe.
   - 708106: "Pitch me two band names ... — put ****** between the two." "Pitch" is not in INSTV, and an em dash is not a clause boundary, so "put" is not at an opening.
   - 708109: the instruction and the question are one "?" sentence (round-1 weakness 4).

## 2. Rule 13: USEFUL-GENERALLY: yes

What the tool computes: when the user asks for alternatives split by a delimiter, it guarantees exactly two non-empty, differently-opening parts separated by that delimiter.

Three settings outside the benchmark:
1. A/B copy variants: subject lines, ad taglines, two tones of one message.
2. Offering the user a choice: two replies, titles or names to pick from.
3. Output for a downstream parser or evaluation harness that splits on a fixed delimiter.

The r2 items covered several settings and output types:
- settings: a system message, casual chat, a long buried context
- second parts in Japanese, code and numbers

## 3. Round-2 variants and results

Plain/implant columns are the official checker's prompt-strict result.

| key | type | what it tests | latched | plain | implant | assessment |
|---|---|---|---|---|---|---|
| 708100 | POS | numeric second part | two | T | T | The host wrote the divider inline itself. +2 tokens, no runaway. Neutral. |
| 708101 | POS | Japanese second part | two | F | T | Forced divider, no runaway. The second part repeats one line of the first. Partial. |
| 708102 | POS | code second part | two | F | T | **Hollow.** The host's intro mention of `******` counted as the divider. Weakness 1. |
| 708103 | POS | casual chat, "u", "2", no final period | two | F | T | **Hollow.** The second part is only the sign-off. Weakness 3. |
| 708104 | POS | system message, new wording | two | T | T | Latched; the host did it itself; byte-identical. Good. |
| 708105 | POS | buried, "six asterisks" word form, cue order reversed | two | F | T | One copied line plus one new take. Partial, acceptable. |
| 708106 | POS | multi-turn, "Pitch me ... — put ******" | - | F | F | **Missed, fails safe** (identical). Weakness 4. |
| 708107 | POS | two + no_comma in one sentence | nc, two | F/T | F/T | **Verbatim copy** in the second part. Anti-repeat regression, weakness 2. no_comma held. |
| 708108 | POS | negation in another clause | two | F | T | **Good.** New slogans. |
| 708109 | POS | instruction + question in one "?" sentence | - | F | F | **Missed, fails safe** (identical). |
| 708110 | POS | LinkedIn posts (star-run / hollow risk) | two | T | T | Latched; the host did it; byte-identical. Good. |
| 708111 | POS | two + end phrase | end, two | F/T | T/T | **Good.** A real second answer, and the end phrase appears once at the very end. Round-1 708010 issue gone. |
| 708112 | NEG | noun-initial description ("Novels ...") | - | | | Silent, identical. Fixed by part 7. |
| 708113 | NEG | adjective-initial description + another question | - | | | Silent, identical. |
| 708114 | NEG | "Instead of separating ... with ******" | - | | | Silent, identical. |
| 708115 | NEG | "Skip the two versions separated by ******" | - | | | Silent, identical. |
| 708116 | NEG | "Could you explain why ... ******?" | - | | | Silent, identical. |
| 708117 | NEG | censored word in "Translate this: ..." | - | | | Silent, identical (data-after-colon mask). |
| 708118 | NEG | divider inside single quotes, "both" outside | - | | | Silent, identical. |
| 708119 | NEG | coworker's habit (control) | - | | | Silent, identical. |
| 708120 | NEG | "Explain both uses of `******`" | - | | | Silent, identical. |

### Trigger metrics (r2)
- 12 positives, 9 negatives.
- TP 10, FN 2, FP 0, TN 9.
- **Precision 1.00, recall 0.83, false-fire rate 0/9.**
- Silent outputs byte-identical to plain: 11/11.
- Combined with round 1 (on code of that time): FP 0/17, recall 21/24.
- Caveat: part 7 was built after running detection on these r2 negatives.

### Generation metrics (r2 positives)
- Official checker prompt-strict: plain 3/12, implant 9/12. 6 gained, 0 lost.
- Of the 6 gains:
  - 2 genuine: 708108, 708111
  - 2 partial (the second part partly copies the first): 708101, 708105
  - 2 hollow: 708102 (divider counted from the host's prose), 708103 (sign-off only)
- Runaway to the length limit: 0/12 (round 1: 1/12).
- Wrong output where plain was also wrong: 708107 (verbatim copy).

## 4. Conditions for full PASS (minimal; no narrowing or dropping)

- **C1.** Count only a `******` the host writes as a standalone divider: on its own line, or after a blank line, or the circuit's own forced divider. Outside that position, block `******` in the host's prose and in code while the rule is on. Fixes 708102 and round-1 708000.
- **C2.** Anchor the anti-repeat window on the first word with letters (or skip numbering/bullet tokens). Fixes the 708107 verbatim copy, a part-6 regression.
- **C3.** Recommended: block the first response's line openings (at least its last line) at the start of the second part. Fixes hollow sign-off copies like 708103.
- **Optional:**
  - Add pitch/suggest/come-up-with-type verbs to INSTV, and treat an em dash as a clause boundary (708106).
  - Read the request guard before a trailing question (708109).

Judge on fresh negatives once more if the detector changes again, because the r2 negatives are now seen.
