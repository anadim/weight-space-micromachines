# JUDGE highlights #2: highlight >= N sections (`detectable_format:number_highlighted_sections`)

Judge: same independent reviewer as round 1 (JUDGE_highlights_1.md). Code reviewed: the CURRENT circuit.py / tables.py
(NOTES.md change log parts 6 and 7). Variants: judge/highlights_variants_r2.jsonl (keys 712100-712121, 12 positive,
10 negative, new wordings and topics). Results: judge/highlights_results_r2.md (part-7 code).
Not opened: fresh*, ood, IFEval data. No model ran on the development machine.

**Final verdict: PASS (conditional; the tool is safe but recall is low).** On these items the tool never fires falsely (0/10). Every
miss is silent and byte-identical to plain. There is no reward hacking and no Python logic. On new phrasings the guards are now
too strict: recall is 6/12, and the tool gives only a small gain over plain. Keep the rule. Do not drop it. Loosen 3 guards (section 5).

Disclosure: the builder ran detection on these r2 items while writing part 7, which removed exactly the false fires that I
predicted (712112-712117). The 0/10 false-fire rate is therefore optimistic. The recall losses below were NOT tuned on, so they
are the honest part of this round.

## 1. Code findings (current code)

Path: circuit.py
- request guards: `rqs`, `reqs`, `topic` (lines ~249, 294-295)
- per-clause cue negation and 'times': `_cue_neg` for 'high' / 'hightimes'
- detector: `h_hit` (lines 468-473)
- count: `s_hn_first` / `s_hn_last` (lines ~361-363), with N > 20 changed to 1
- answer tracker and emitter: unchanged since round 1

Word lists in tables.py: INSTV, HIGHOBJ, EXPLAINV, WHQ, REPORTV, CODEW, OTHERS, DESCV.

- **Python logic at inference: none.** The only Python branch in `step` is `if name not in R` in `_cue_neg`. It initialises a
  register for a fixed rule name and does not depend on data. Everything else is `torch.where` / products on 0/1 wires, registers
  and fixed tables. The readouts are the disclosed ones.
- **IFEval-specific vocabulary or templates: none.** The new lists (INSTV instruction verbs, HIGHOBJ =
  section/part/phrase/sentence/passage/term/text, REPORTV, CODEW) are generic English.
- **Fixed since round 1 (confirmed by the run):**
  - Negation and 'times' are now checked per clause, against the cue word. 712100 ("Skip the intro and bold at least 3 key
    terms ...") and 712101 ("... at least twice, and italicize at least two sections") both fire correctly.
  - The "highlight/bold + any number" path is gone, because a highlight/bold word now needs markdown/asterisk/'*' or a HIGHOBJ
    object. 712112-712114 are silent.
  - The imperative opening now needs an instruction verb, so "Teachers usually highlight ..." (712115) is silent.
  - The topic guard covers how-to and explain requests (712116).
  - A casual question with no '?' (712117) is silent.
  - System latches persist into later turns (712104) and a cancel cue drops them (712118).
- **Still open (each causes a safe miss, not a false fire):**
  1. *Third-person guard is sentence-scoped.* `rqs` uses `se('OTHERS')` / `se('DESCV')`. A subordinate clause with 'people',
     'she' or 'they' vetoes a real instruction in the same sentence:
     - 712109 "When you reply, put two or three of the key dates in *italics* so people can spot them." is missed.
     - 712108 "(and highlight at least 2 sections with *asterisks*, she loves that)" is missed.
     Fix: use clause scope (`rqc`) together with the cue word, in the same way as `_cue_neg`.
  2. *Topic guard is sentence-scoped.* `topic = se('EXPLAINV') * se('WHQ')`. "Could you please highlight three sections in
     markdown when you explain how solar panels work?" (712107) is missed. Fix: apply the guard only when the explain/tell verb and
     the cue word share a clause, or when the explain verb opens the sentence.
  3. *Request marker needs a verb or a modal.* These are missed:
     - Terse requirement lists such as "Requirements: at least 2 italicized phrases, under 150 words." (712103) and
       "...; at least 2 italicized phrases." (712102).
     - Chat shorthand such as "can u italicize like 2 bits of it" (712106).
     Possible fix: let a ':' after a requirement noun (requirements/rules/format/constraints) act as a request marker for the
     rest of the line. Add 'u' to the you-list, and 'can'/'could' + you to the opening.
  4. *Count capture is unchanged* (first number after the cue word). In this round it did not cause an error, because 712103 was
     silent anyway. The round-1 failure mode (712011, list index read as N) is still in the code.
  5. *Content of the appended spans* is unchanged. In 712110 the first forced span is a copy of the instruction
     ("*italicize at least two sections*"), and the later spans copy earlier sentences. The checker passes, but the output quality
     is low. This is a quality issue, not a protocol issue.

## 2. Rule 13: is f(x) generally useful?

- **f(x):** when a message asks for at least N emphasized spans (markdown `*...*` / `**...**`), count the emphasized spans that
  the model has written. Before the turn ends, add spans until the count is N. Otherwise do nothing.
- **Settings outside the benchmark:**
  1. Study notes and flashcards that must emphasize key terms.
  2. Skimmable support answers and UI copy where key points must be bold.
  3. House-style system prompts for newsletters or marketing copy that require emphasis. 712104 shows the system-message form working.
- **USEFUL-GENERALLY: yes.**

## 3. Variants and results (round 2)

"hl plain/imp" is the official checker result for the highlight instruction. "strict" is prompt-level strict accuracy.

| key | test | latched (N) | expected N | hl plain/imp | strict plain/imp | note |
|---|---|---|---|---|---|---|
| 712100 | "Skip the intro and bold at least 3 key terms using **...**" | yes (3) | 3 | T/T | T/T | fix check, correct; byte-identical (host complied) |
| 712101 | 'twice' in another clause | yes (2) | 2 | T/T | T/T | fix check, correct; byte-identical |
| 712102 | "Requirements: under 150 words; at least 2 italicized phrases." | no | 2 | F/F | F/F | safe miss (no request marker), as predicted |
| 712103 | "Requirements: at least 2 italicized phrases, under 150 words." | no | 2 | T/T | T/T | safe miss (no verb); host complied anyway |
| 712104 | system rule persists into a later user turn | yes (2) | 2 | T/T | T/T | correct; byte-identical |
| 712105 | "Make the answer skimmable: bold at least four ... with markdown" | yes (4) | 4 | T/T | T/T | correct; byte-identical |
| 712106 | "can u italicize like 2 bits of it 🙏 thx" | no | 2 | F/F | F/F | safe miss ('u', 'can' not a request marker) |
| 712107 | "Could you please highlight three sections ... when you explain how ...?" | no | 3 | T/T | T/T | safe miss (topic guard is sentence-scoped) |
| 712108 | parenthetical with "she loves that" | no | 2 | T/T | T/T | safe miss (OTHERS sentence-scoped), as predicted |
| 712109 | buried in a long message, "...so people can spot them" | no | 2 | F/F | F/F | safe miss (OTHERS 'people' sentence-scoped) |
| 712110 | lowercase + highlight in one sentence | yes (2), + lower | 2 | F/**T** | F/**T** | gain; first appended span echoes the instruction |
| 712111 | "Give three tips ... in **bold**" | yes (3) | 3 | T/T | T/T | correct; byte-identical |
| 712112 | NEG "Highlight the top 3 attractions in Rome" | no | - | - | - | byte-identical |
| 712113 | NEG "Write a bold 5-step plan" | no | - | - | - | byte-identical |
| 712114 | NEG "emphasize the 2 biggest benefits" | no | - | - | - | byte-identical |
| 712115 | NEG "Teachers usually highlight two sections in markdown..." | no | - | - | - | byte-identical |
| 712116 | NEG "Could you tell me how to italicize a section in Google Docs?" | no | - | - | - | byte-identical |
| 712117 | NEG "quick q - whats the shortcut to italicize stuff in word" | no | - | - | - | byte-identical |
| 712118 | NEG system rule cancelled with "plain text please, no markdown" | no | - | - | - | byte-identical |
| 712119 | NEG "Please avoid italics and bold entirely ... asterisks ..." | no | - | - | - | byte-identical |
| 712120 | NEG description of a notes app's feature | no | - | - | - | byte-identical |
| 712121 | NEG single-quoted instruction on a sticky note | no | - | - | - | byte-identical |

## 4. Summary numbers (round 2, part-7 code)

- **Trigger recall**: 6/12 = 50%. All 6 misses are silent for this rule and byte-identical to plain.
- **Trigger precision**: 6/6 = 100%. On negatives: 0/10 latched and 0/10 output changes (optimistic; see the disclosure).
- **Argument N correct when fired**: 6/6.
- **Highlight checker on positives**: plain 8/12, implant 9/12 (1 gain, 0 losses). **Prompt-strict**: plain 8/12, implant 9/12.
- **Rounds 1 + 2 together** (r1 on the part-5 code, r2 on the part-7 code):
  - Recall: 20/28.
  - N correct when fired: 18/20.
  - Negative output changes: 0/19.
  - Prompt-strict losses: 0.
  - Instruction-level losses: 1 (712011, the word limit broken by appended spans).

## 5. Verdict

**PASS (conditional).** The protocol criteria are met:
- No reward hacking, no IFEval vocabulary, no leakage, no Python logic at inference.
- On positives, the tool fires with the correct N (6/12) or fails safe (6/12, byte-identical).
- The false-fire rate on negatives is zero.
- USEFUL-GENERALLY: yes.

Main reservation: the three rounds of guard changes have pushed the trigger from "too eager" (round 1) to "too shy" (round 2).
A tool that fires on half of the realistic requests and in this round helped on 1/12 prompts gives little value outside
IFEval-like wording. This is not a reason to drop it, because the misses are safe. Minimal changes, ranked by recall gained per risk:
1. Clause-scope the third-person guard (`rqs` OTHERS/DESCV) and the topic guard (EXPLAINV x WHQ), checked against the clause
   that holds the highlight cue. This recovers 712107, 712108 and 712109 without reopening the round-1 negatives, which had the
   third-person subject or the how-to verb in the same clause as the cue.
2. Read the count next to the item word (N <= 3 tokens before HIGHOBJ / the cue), as the bullet and paragraph rules do. This stops
   the list-index capture (r1 712011).
3. Optional: count a requirement-list colon ("Requirements:/Rules:/Format:") as a request marker for that line, and add 'u' to the
   you-list.

Any of these changes must be checked against fresh negatives that were not used to tune parts 5-7. Do not drop the rule.
