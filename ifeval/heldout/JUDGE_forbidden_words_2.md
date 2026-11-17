# JUDGE forbidden_words, round 2 (final verdict on the CURRENT code, NOTES change log part 7)

Judge: independent reviewer (subagent). Round-1 file: JUDGE_forbidden_words_1.md (FAIL, narrow).
Round-2 variants: judge/forbidden_words_variants_r2.jsonl, keys 706100-706117 (10 POS, 8 NEG). Results: judge/forbidden_words_results_r2.md.
I opened no test data: no fresh, ood or IFEval files.

**Verdict: PASS, with conditions.** The rule is usable. It is not dropped.

**Conditions:**
1. Document the residual risks of section 5 in NOTES. The two misspelling cases and the 'take the word X literally' false fire are the main ones.
2. The production-verb narrowing (section 6.1) is recommended. If it is applied, dev, seed-2 and the 30 mention negatives must stay exact.

**Evidence caveat (important).** The part-7 changes were made after the builder ran detection on these round-2 items. The same happened in round 1 with part 5. The silent results on 706110, 706111, 706112, 706113 and 706116 therefore show that those specific holes are closed. They are not clean held-out evidence. I expect a truly fresh set to show a higher false-fire rate than the 1/8 measured here.

## 1. Code findings (current circuit.py / tables.py, forbid path)

- **No reward hacking.**
  - The cue lists are generic English closed-class words: NEG, WORDTERM, DETW, SEPW, APPEARS, NEGADJ, STOPF, FREQ3, TIMES, and the new INSTV instruction-verb list.
  - There are no IFEval sentences, templates or entity lists.
- **Borderline patch.** STOPF now contains 'count/counts/limit/limits/choice/order'. This patch exists because of my round-1 item "the word count". It is generic in form (common noun compounds of 'word'), but it is a list patch, not a structural rule. Other compounds ("the term paper") are now handled by the part-7 'forget' fix, not by the list.
- **No Python logic at inference.**
  - All gates use torch.where, gather and integer registers.
  - The Python loops run over fixed key lists only.
  - The readout "top token is in the blocked set" is disclosed.
- **Rolling-hash word identity.** It uses integer registers mod 2^31-1 and is disclosed in NOTES. I accept it as disclosed.
- **Part-7 request marker.** An imperative opening now needs an instruction verb (INSTV). An opening negation counts only at the sentence opening, and must/should/please/you anywhere still count. Topic sentences (explain/tell + why/how), reported claims and code requests are excluded. 'forget'/'mind' now cancel a preceding negation for the forbid recency register. This closed my predicted false fires on:
  - 706111 "Good writers never use the word 'very'"
  - 706112 system "Students are not allowed ..."
  - 706113 "Don't forget the term paper ..."
  - 706110 "Can you tell me why some style guides say never ..."
  - 706116 reported teacher instruction
- **Still open in the code:**
  - **Frequency words checked per sentence.** The commit still checks TIMES/FREQ3 over the whole sentence. NOTES part 6 claimed this was per clause. As a result:
    - "Never once use the word X" is dropped (706100).
    - Any sentence that pairs the ban with "at least / more than / fewer than N ..." drops the ban (706101).

    These are fail-safe misses, but the second one is a common IFEval-style combination.
  - **System statements without a request marker.** The request marker now drops system-message house rules that have no verb, modal or 'you'. Example: "House style ...: the terms 'unfortunately' and 'sorry' are off-limits in every reply." (706102). This is a fail-safe miss, but plain failed this item, so a case is lost. Recall cost of part 7.
  - **Bans not tied to a production verb.** A negation is not required to govern a production verb (use/say/write/include/mention). "Don't take the word 'literally' literally" still commits (706114).
  - **Misspelling evasion.** The block acts on the piece that completes the word, so a meta-mention becomes a misspelling. In 706104 the host wrote "vibees"/"journeys" instead of "vibes"/"journey". In 706114 it wrote "literaly" throughout the answer. This passes the \b checker. In real uses (profanity, brand or legal filters) a misspelled banned word usually defeats the purpose of the ban. This is a general-usefulness weakness, not a test hack.
  - **Unchanged since round 1:**
    - A multi-word phrase is split into separate slots.
    - Only 6 slots.
    - Negation synonyms 'keep ... out of', 'steer clear of' and 'leave out' are not recognised (706107, fail-safe miss).

## 2. Rule 13: is the function useful in general?

f(x): given words taken from a natural-language prohibition, the tool guarantees the output never spells any of those words as a whole word, in any case or tokenization. When needed, it substitutes the host's best other token.

Settings outside the benchmark:
1. Brand, legal or compliance banned-term lists, and competitor-name filters.
2. Anonymisation and spoiler avoidance in summaries.
3. Word games (Taboo), children's-content filters, and style rules that ban cliche words.

**USEFUL-GENERALLY: yes.** For settings 1 and 3 it needs the misspelling fix in section 6.2 to be fully useful.

## 3. Round-2 variants and results (part-7 code)

"Fired" means the rule latched with the right words. "Strict" is the official checker result (plain then implant). "Same" means the implant output is byte-identical to plain.

| key | type | tests | expected | fired | words | strict plain / implant | same |
|---|---|---|---|---|---|---|---|
| 706100 | POS | "Never once use the word 'basically'" | miss (TIMES) | no | - | F / F | yes |
| 706101 | POS | forbid + "at least 120 words" in one sentence | miss (FREQ3) | no | - | TF / TF | yes |
| 706102 | POS | system "the terms ... are off-limits in every reply" | fire | **no** | - | F / F | yes (host said "sorry") |
| 706103 | POS | multi-turn, "without using 'bread' or 'fresh'" | fire | yes | bread, fresh | T / T | yes |
| 706104 | POS | casual "pls no 'vibes' or 'journey'" after a '?' | fire | yes | vibes, journey | F / **T** | no (host's meta-mention became "vibees"/"journeys") |
| 706105 | POS | rule buried in a long product message | fire | yes | revolutionary | T / T | yes |
| 706106 | POS | toast + "under 150 words" + "don't say ..." in one sentence | fire | yes | marriage, forever | TT / TT | yes |
| 706107 | POS | "Keep the words ... out of" | miss | no | - | F / F | yes |
| 706108 | POS | caps rule + "Do not use the word 'pain'" | fire | yes | pain | TT / TT | yes |
| 706109 | POS | "without using the word 'light'" (photosynthesis) | fire | yes | light | T / T | yes |
| 706110 | NEG | "Can you tell me why style guides say never to use the word 'utilize'?" | false fire | no | - | - | yes |
| 706111 | NEG | "Good writers never use the word 'very'. Why is that?" | false fire | no | - | - | yes |
| 706112 | NEG | system: students not allowed to use 'gonna'; correct my sentence | false fire | no | - | - | yes |
| 706113 | NEG | "Don't forget the term paper is due Friday; help me outline it" | false fire | no | - | - | yes |
| 706114 | NEG | "Don't take the word 'literally' literally. Explain ..." | false fire | **YES** | literally | - | **no**: "literaly" throughout, 189 vs 109 tokens |
| 706115 | NEG | multi-turn: previous turn banned 'wet', new task needs 'wet' | silent | no | - | - | yes |
| 706116 | NEG | teacher told the class never to use 'stuff'; please explain | false fire | no | - | - | yes |
| 706117 | NEG | "There's no single word for this feeling ..." | silent | no | - | - | yes |

Trigger statistics, round 2:
- Positives: 6/10 fired with exact words. Recall = 60%.
- Every fire was enforced. The forbid checker passed on all 6 fired items, with 0 forced-token misses.
- The 4 misses are byte-identical to plain, so they fail safe.
- Negatives: 1/8 fired (706114). False-fire rate = 12.5%, and that false fire is harmful.
- Precision = 6/7 = 85.7%.

Generation, round 2:
- Forbid checker passed on 6/10 positives plain and 7/10 with the implant (+1, 0 lost). This counts the forbid instruction only; on 706101 the length instruction failed for both.
- 7/8 negatives are byte-identical to plain.

Rounds 1 and 2 together (23 + 18 items; each round on the code of its own run):
- Positives: 17/25 fired, all enforced with no leaks. The 8 misses are all fail-safe.
- Negatives: 3/16 fired (2 on part-5 code, 1 on part-7 code).
- Positives that passed the forbid checker gained +5 (4 in round 1, 1 in round 2), with 0 lost.

## 4. Verdict against the criteria

- No reward hacking or Python logic at inference: **met**. The STOPF compound patch is a borderline list patch, noted above.
- Correct or fail-safe on positives: **met**. 17/17 fired positives across both rounds were enforced without leaks. Every miss is silent and byte-identical.
- Near-zero false fires on negatives: **borderline**.
  - Measured: 1/8 on the current code, on adversarial items that the builder had already seen at detection time.
  - The remaining false fire is an idiom ("don't take the word X literally"). The current structure has no defence against it: a ban not tied to a production verb.
  - Its damage is limited to the one quoted word. It is visible as misspellings.
- USEFUL-GENERALLY: **yes**.

**PASS, with conditions.** Do not drop the rule. Its failure profile is narrow:
- it only fires on negation + "the word(s) X" / quoted-word structures;
- when it misfires, it only affects the quoted word;
- when it misses, the output is byte-identical to plain.

## 5. Residual risks to disclose

1. **A ban not tied to a production verb still commits.** Example: "don't take the word X literally".
2. **Misspelling evasion.** When the host wants to mention a banned word, the last-piece block yields misspellings ("vibees", "literaly"). This passes the checker but defeats the intent of a real ban.
3. **Frequency words over the whole sentence.** A sentence that combines the ban with a frequency or length phrase ("once", "at least N", "more than N") drops the ban. This fails safe.
4. **System house rules without a verb, modal or 'you'** ("X and Y are off-limits") are missed. This fails safe.
5. **Phrase split.** A multi-word phrase is split into words, which over-blocks.
6. **6-slot limit.**
7. **Negation synonyms** ("keep out of", "steer clear of") are not recognised. This fails safe.

## 6. Recommended minimal narrowing and fixes (generic; re-check dev, seed-2 and mentions after each)

1. **Commit pattern A only when a production verb follows the negation.** The verb (use/say/write/include/mention/type/utter/print/output/employ) must come after the negation and before the list. Pattern B (should not appear / is banned / off-limits) is unchanged. This removes 706114-type idioms. It costs nothing on the positives seen in either round: all used use/say/include/mention, or pattern B, or 'without' + list (keep 'without' as its own path).
2. **Block at word start (quality fix, optional).** When a forbidden word's first piece is the host's top token at a word start, and the word can continue into the forbidden spelling, block that first piece. This prevents "vibees" and "literaly".
3. **Recall fixes (optional).**
   - Check TIMES/FREQ3 at clause level for the forbid commit, as NOTES part 6 already claims.
   - Accept pattern-B statements with a negative adjective (banned/forbidden/off-limits) about "the word(s)/term(s) X" in a system message without a request marker.
