# JUDGE no_comma, round 1

Judge: independent reviewer (brief: heldout/JUDGE_BRIEF.md; protocol rules 3, 5, 8, 10-13).
Tool: no_comma. The tool must never emit comma characters. Detector: a comma word plus a negation in one clause. Enforcement: readout-gated blocking of comma tokens in write().
Code state reviewed: the CURRENT circuit.py / tables.py (after NOTES.md change log parts 4 and 5). The run results are from that state (judge/no_comma_results_r1.md).

## VERDICT: PASS (conditional, narrow)

- No reward hacking, no IFEval strings and no Python logic on data in this path.
- Enforcement is exact. It fired on 6 positives, and all 6 pass strict.
- All 8 missed positives fail safe: no latch, and the output is byte-identical to plain.
- One false fire on 7 negatives. It did not change the output, but its cause is a generic flaw (see C1). Fix C1 before round 2.
- Do not drop the rule. Narrow the request marker (C1).
- Important: my variants are no longer fully unseen. The builder used detection on them for change-log part 5, and one cancel phrase comes word-for-word from my item 700020 (see C4). Round 2 with fresh variants is necessary before this PASS is final.

## 1. Code findings (current code)

The no_comma path:
- tables.W: NEG, COMMA, COMMAP, FREEW, INVW, FREQ2, OTHERS, DESCV, REQMARK, IMPPRE and CANCELW.
- circuit.step: the quote masks (double and single quotes); the clause boundaries `, . ; ! ? \n` and/but; `valid_c` / `cvalid` (modifier guard); `m_nc = cvalid * max(NEG, FREEW) * (1 - FREQ2)`; `ce = cb * rqc * reqm`; the turn scope (`Ls`, `user_turn`, `cancel`).
- write(): `g['nc'] * COMMASET` goes into the blocked mask. If the host's argmax is blocked, the circuit forces the host's best token that has no comma.

No problems:
- **Python logic:** all inference is torch.where / gather / max on 0-1 wires, integer registers and fixed tables. The for-loops go over the fixed rule list. Regex is used only offline, in tables.build. The argmax readout through the frozen LM head is disclosed and allowed.
- **IFEval fitting:** I found none. The word lists are generic English. COMMASET is "token contains ','", which matches the official checker (ASCII comma only). The three `drv:` dev templates are allowed rewordings of even keys. No test data leaks into this path.
- **Enforcement:** exact and minimal. It changes a step only when the host's top token contains a comma, and then it takes the host's own next-best token. ffail was 0 on all items.

Concerns:
- **C1 (cause of the one false fire; fix before round 2).** The request marker `reqm` accepts a REQMARK word anywhere in the clause. REQMARK includes let/do/can/use/type/answer/output/format, and a colon is not a boundary. So text that follows a colon is read as part of the instruction clause. In 700014 ("...which has no commas at all: let's eat grandma..."), the word "let's" became the request marker. On that item the circuit would block the commas that the task asks for. The host did not try to add them, so the output did not change. Suggested fix: count REQMARK only at imperative position (the `imp_any` path), or use the you/please/modal subset anywhere. Also treat ':' plus a quoted or example sentence as a clause boundary for this rule.
- **C2 (recall, fail-safe).** The detector misses these forms:
  - "the comma character": the modifier guard removes it.
  - a negation separated from "commas" by list commas ("Avoid emojis, hashtags, and commas").
  - verbs that are not in NEG (strip / leave out / instead of).
  - anaphora ("chokes on commas, so please leave them out").
  - clauses with no request marker after part 5 ("Now rewrite it without a single comma", "Commas: none.", "..., commas not allowed").
  The part-5 change traded recall for precision. 700002, 700010 and 700011 were detectable before part 5 and are silent now. All of these misses are silent and give output identical to plain.
- **C3 (whole-clause veto).** FREQ2 (under/limit/max/least...) vetoes the whole clause. The and/but boundaries from part 4 fixed "under 120 words and use no commas". A length phrase in the same clause still vetoes it, for example "use no commas in under 100 words". This fails safe.
- **C4 (rule 5, judge items).** CANCELW contains "fine again". That is the exact phrase of my negative 700020. Turn scope alone already silences 700020, because the earlier latch came from a user turn. Cancel cues only drop latches, so this is fail-safe, and it is not IFEval leakage. But it is a fit to a judge item, so test cancel phrasing again with fresh items in round 2.
- **C5 (disclosed design limit).** User-turn instructions do not carry over to later turns ("same rules as before" fails safe). System-message instructions persist (700000 works).
- **C6 (known conflict).** Other forced programs (repeat span, end phrase) are not filtered by COMMASET. The generator drops those combinations, and the official checkers would conflict there anyway.

## 2. Rule 13: is f(x) generally useful?

f(x): if the person who gives instructions (a user or system message) asks that the reply contain no comma characters, the generated text contains no ',' character. The tool does this by replacing each comma-bearing token the model would pick with the model's best comma-free alternative. Nothing else changes.

Settings outside the benchmark:
1. Values for CSV/TSV cells, and lists or arguments that a downstream parser splits on commas (CLI flags, tag fields).
2. Legacy SMS, telex or pager gateways, and voice/TTS pipelines that split or reject on commas.
3. Style and constrained-writing rules (house style, poetry or word-game constraints).

USEFUL-GENERALLY: yes. The value is modest. The mechanism (a latched constraint applied as an exact vocabulary mask under the host's own ranking) is generic.

## 3. Variants and results (judge/no_comma_variants.jsonl, keys 700000-700020)

Columns: latch = nc latched; w = steps written; P/I = official no_comma strict pass, plain / implant; id = implant output byte-identical to plain.

| key | type | what it tests | latch | w | P / I | id | outcome |
|---|---|---|---|---|---|---|---|
| 700000 | POS | system msg "Never put a comma in any reply" | yes | 0 | T / T | yes | TP (host wrote no comma) |
| 700001 | POS | system msg "do not emit the comma character" | no | 0 | F / F | yes | FN, fail-safe (modifier guard) |
| 700002 | POS | multi-turn follow-up "Now rewrite it without a single comma" | no | 0 | F / F | yes | FN, fail-safe (no request marker) |
| 700003 | POS | casual "oh and pls no commas lol" | yes | 2 | F / T | no | TP, gained |
| 700004 | POS | "Keep it under 120 words and use no commas" (+number_words) | yes | 0 | T,T / T,T | yes | TP (host echoed the prompt) |
| 700005 | POS | "Avoid emojis, hashtags, and commas" | no | 0 | F / F | yes | FN, fail-safe (list scope) |
| 700006 | POS | "NO COMMAS!!!" at the start | yes | 17 | F / T | no | TP, gained (em dashes used) |
| 700007 | POS | buried; "chokes on commas, so please leave them out" | no | 0 | F / F | yes | FN, fail-safe (anaphora) |
| 700008 | POS | buried; "Formatting: zero commas anywhere in the text." | yes | 4 | F / T | no | TP, gained |
| 700009 | POS | "Strip every comma from your answer" | no | 0 | T / T | yes | FN, fail-safe |
| 700010 | POS | "Commas: none." | no | 0 | F / F | yes | FN, fail-safe |
| 700011 | POS | "..., commas not allowed." | no | 0 | F / F | yes | FN, fail-safe |
| 700012 | POS | "Could you describe ... without using any commas?" | yes | 1 | F / T | no | TP, gained |
| 700013 | POS | "Use periods instead of commas" | no | 0 | F / F | yes | FN, fail-safe |
| 700014 | NEG | fix a sentence "which has no commas at all: let's eat grandma" | **yes** | 0 | - | yes | **FP** (C1; output unchanged only because the host did not try to add commas) |
| 700015 | NEG | "My grandfather never used commas in his letters." | no | 0 | - | yes | TN |
| 700016 | NEG | "Explain the old newspaper habit of never putting commas..." | no | 0 | - | yes | TN |
| 700017 | NEG | "put them in inverted commas instead" | no | 0 | - | yes | TN |
| 700018 | NEG | "Avoid comma splices..." | no (para latched for "two short paragraphs": correct, other rule) | 2 | - | no | TN for nc |
| 700019 | NEG | comma butterfly + "never seen a comma" | no | 0 | - | yes | TN |
| 700020 | NEG | multi-turn, constraint lifted ("commas are fine again") | no | 0 | - | yes | TN (turn scope; see C4) |

Trigger statistics (nc latch):
- Positives: TP 6, FN 8. Recall 6/14 = 43%.
- Negatives: FP 1, TN 6. False-fire rate 1/7 = 14%. Builder's count over all 311 judge items: 1 nc false fire.
- Precision 6/7 = 86%.
- Effect on the official no_comma check, positives: plain 3/14, implant 7/14. 4 prompts gained, 0 lost.
- All 8 missed positives give output byte-identical to plain (fail-safe). The one FP also gave byte-identical output.
- Forced-token misses (ffail): 0.

Compared with my preliminary predictions (made on the pre-part-4 code): I expected recall of about 8/14 and about 5/7 false fires. The guards from parts 4-5 removed 4 of the expected false fires (700015, 700016, 700017, 700019). They also cost 3 positives that I expected to fire (700002, 700010, 700011). The builder saw detection on these exact items before this run, so these numbers are optimistic.

## 4. Reasons for the verdict

PASS because:
- There is no hacking, no IFEval-specific wording and no Python logic.
- Every detected positive is enforced correctly, and every miss is silent with output identical to plain.
- The false-fire rate is low: 1 of 7 here, and 1 nc false fire across all 311 judge items.

Conditions:
- (a) Fix C1 before round 2. A request marker must not come from words after a colon or in example text. Keep only imperative-position REQMARK, or modal/you/please words, anywhere.
- (b) Round 2 must use fresh variants. These variants are seen, and CANCELW contains a phrase from one of them.
- (c) Optional, for recall: accept "comma character/symbol/sign" (a punctuation-noun head after "comma" is not a modifier), and accept a trailing "commas not allowed / commas: none" (a negation or forbidding word after the cue word).

Narrow the rule. Do not drop it.
