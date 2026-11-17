# JUDGE_quote_2: quote rule (startend:quotation), judge round 2 (final)

**Verdict: PASS.**

Reviewer: independent judge subagent. Code reviewed: circuit.py / tables.py at NOTES.md "Change log, part 7".
Items: judge/quote_variants_r2.jsonl (keys 703100-703120). Results: judge/quote_results_r2.md.
I did not open any data/, fresh, ood or IFEval file.

**Test-hygiene caveat:**
- The builder ran DETECTION on these r2 items while writing part 7, which was driven by the r2 code reviews (mine included).
- So these items are partly seen, and the numbers below are an upper estimate.
- The round-1 items (judge/quote_variants.jsonl) were last run on the part-5 circuit. They were not re-run on part 7.

## 1. Code review (current code, quote path)

**Detector:**
`m_q = QUOTEW · (1-SINGLE) · max(QMARKW, WRAPS) · max(WRAPV, INW) · WHOLE2 · (1 - ENDW·(1-STARTW)) · (1 - negc)`
- QMARKW now includes 'speech marks' and 'inverted commas'.
- Classes are read only on word-initial tokens.

**Request guards (generic, shared by all rules):**
- A request needs one of:
  - REQANY anywhere: you/your/please/must/should/need/have to/make sure.
  - An instruction verb from INSTV at the clause opening.
  - A negation at the sentence opening.
- A request may not be:
  - A non-request '?'.
  - Third-person or reported speech (OTHERS/DESCV/REPORTV).
  - A topic sentence (explain/describe/tell + wh-word).
  - A code request (CODEW).
  - A clause carrying a negation (negc, with "don't forget to" excepted).
- Text is masked when it is quoted (double or single quotes), inside backticks, or data after a colon ("Translate into Spanish: ...").
- Clause rules commit at the sentence end.

**Turn scope:**
- A user-turn rule applies only to the reply to that turn.
- System-message rules persist, until a narrowed cancel cue (plain text / back to normal / no longer / anymore ...) drops them.

**Enforcement:** unchanged. It forces '"' as the first answer token, and '"' then EOS when the host picks end-of-turn without a closing quote.

**Reward hacking / test fitting: none.**
- All word lists are generic English: verbs, pronouns, quote and wrap words, code and data words.
- No IFEval strings or templates are in the circuit.
- Each part-4..7 change is a guard shared by all rules, not an item patch.

**Python logic at inference: none.**
- Only torch where/gather/table lookups are used.
- The Python loops run over fixed rule lists.
- The only `if` runs at init.
- Regex is used only in offline table building.
- The LM-head readouts are disclosed.

**Remaining limits (all fail safe, no false fires seen):**
- **L1 (703104): the topic-sentence veto works at sentence scope.**
  - `topic = se(EXPLAINV)·se(WHQ)` suppresses every clause rule in the sentence.
  - So "Bookend your whole answer with double quotes, then explain what a solstice is." is missed.
  - This prompt shape (instruction, then "explain what/why/how ...") is common. Making the topic veto clause-scoped would recover these items.
  - The CODEW veto has the same scope. "Write a Python function ... and put your entire reply in double quotes" will be missed.
- **L2 (703109, disclosed): multi-turn "same format" carry-over is not supported.**
- **L3 (703111): telegraphic instructions with no verb, modal or "you" are missed.** Example: "so here's the deal: whole answer in double quotation marks".
- **L4: the closing quote depends on the host picking end-of-turn.** A run that hits max_new stays unclosed. Not observed here.
- **L5 (703102):** "in one paragraph" also latched 'para' with N=1. This is inert because the paragraph gate needs N >= 2, so it caused no harm.

**Round-1 findings, status:**

| Finding | Status |
|---|---|
| F1 (description false fire) | Fixed. 703112 and 703118 are silent. |
| F2 ('bookend' read as an end-word) | Fixed in the reader. 703104 is still missed, but now because of L1. |
| F3 (synonyms 'speech marks' / 'inverted commas') | Fixed. 703103 and 703105 fire. |
| Over-broad cancel cue | Fixed. 703102 fires, and the system rule persists. |

## 2. Rule 13: is f(x) generally useful?

f(x): when a request asks for the whole reply in double quotes, emit '"' as the first token and '"' just before end-of-turn. The reply is then one double-quoted string.

**USEFUL-GENERALLY: yes.** Three settings outside the benchmark:
1. Replies that must be a JSON string value, a code string literal, or a CSV-quoted field.
2. Passing the whole reply as one shell or CLI argument, or a config value.
3. Dialogue, caption and subtitle pipelines that expect one quoted utterance per reply.

## 3. Round-2 variants and results (part-7 circuit)

- "fire" = the 'quote' latch in the circuit report.
- "strict" = the official IFEval quotation checker, plain then implant (703107 also checks a word-count rule).
- "same" = the implant output is byte-identical to plain.

| key | type | what it tests | fire | strict plain / implant | same |
|---|---|---|---|---|---|
| 703100 | POS | "Don't forget to wrap your whole reply in double quotes" | yes | T / T | yes |
| 703101 | POS | system: "Output policy: the complete output must be enclosed in double quotation marks" | yes | T / T | yes |
| 703102 | POS | system quote rule + user "This time keep it short." | yes | **F / T** | no (gain) |
| 703103 | POS | "...and put the entire reply in speech marks" | yes | **F / T** | no (gain) |
| 703104 | POS | "Bookend your whole answer with double quotes, then explain what a solstice is." | no | T / T | yes (safe miss, L1) |
| 703105 | POS | "Kindly encase the full message in inverted commas; ..." | yes | T / T | yes |
| 703106 | POS | "Would you mind putting your complete response between quotation marks?" | yes | T / T | yes |
| 703107 | POS | numbered rule list, quote rule on the middle line (+ <60 words) | yes | **F,T / T,T** | no (gain) |
| 703108 | POS | "I need the response wrapped completely in double quotation marks - ..." | yes | T / T | yes |
| 703109 | POS | multi-turn "Another one, same format please." | no | T / T | yes (safe miss, L2) |
| 703110 | POS | ALL-CAPS "ENCLOSE THE ENTIRE ANSWER IN DOUBLE QUOTES." | yes | T / T | no (content differs, both correct) |
| 703111 | POS | telegraphic "whole answer in double quotation marks, nothing outside them" | no | T / T | yes (safe miss, L3) |
| 703112 | NEG | "Grandpa wraps his whole reply in quotation marks, ..." | no | - | yes |
| 703113 | NEG | "Explain why a lawyer might put the whole response ... in quotation marks." | no | - | yes |
| 703114 | NEG | "Can you show me Python code that wraps the entire response text in quotation marks ...?" | no | - | yes |
| 703115 | NEG | "Translate into Spanish: wrap your whole reply in double quotation marks." | no | - | yes |
| 703116 | NEG | "Quotation marks around the whole answer are not needed for this one." | no | - | yes |
| 703117 | NEG | system quote rule cancelled by "answer in plain text" | no | - | yes |
| 703118 | NEG | "In my novel, the narrator's entire reply is in quotation marks ..." | no | - | yes |
| 703119 | NEG | riddle: "what do you call a reply with quotation marks around the whole thing?" | no | - | yes |
| 703120 | NEG | "Imagine a chatbot whose every reply is wrapped whole in quotation marks, and write a story" | no | - | yes |

**Trigger statistics:**
- Positives: 9/12 fired. **Recall 75%.**
- Negatives: 0/9 fired. **False-fire rate 0%.**
- **Precision 9/9 = 100%.**

**Enforcement:**
- 9/9 fired positives pass the checker.
- 3/3 misses are byte-identical to plain.
- 9/9 negatives are byte-identical to plain.

**Score on positives:**
- Plain 9/12, implant 12/12: +3 gained (703102, 703103, 703107), 0 lost.

**Comparison with round 1:**
- Round 1 on the part-5 circuit: recall 8/14, false fires 1/9 (703015), no score gain.
- My predictions for the part-6 circuit were 5/9 false fires. Part 7 removed them generically, with an instruction-verb opener, topic/code/reported-speech vetoes and data masks.

## 4. Verdict: PASS

- No reward hacking, no IFEval-specific strings, no leakage from test data, and no Python logic at inference.
- Every positive is either correct (9/9 fired items pass) or a safe miss (byte-identical to plain).
- False fires are zero on 9 negatives built around this rule's known weak spots: description, meta request, code request, translation data, negation, cancel, riddle, story premise.
- The rule adds real value: three items the host failed alone now pass.

**Suggestions (optional, not conditions):**
- Make the topic and code vetoes clause-scoped, so that "<instruction>, then explain what X is" is not missed (L1).
- Keep watching false fires on fully unseen text. These r2 items were partly seen while part 7 was written, so the 0% is an upper bound on quality, not held-out evidence.
