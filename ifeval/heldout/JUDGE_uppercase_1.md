# JUDGE_uppercase_1: uppercase rule-keeper (whole response in capital letters)

Judge: independent reviewer, round 1. Keys 702000-702024. Variants: judge/uppercase_variants.jsonl. Raw results: judge/uppercase_results_r1.md.
Brief: JUDGE_BRIEF.md. Protocol: bbh_generic/PROTOCOL.md (rules 3, 5, 8, 10-13).
I did not open fresh.jsonl, fresh_templates.json, ood.jsonl or any IFEval data. I ran no models.

**VERDICT: PASS (provisional).** There is no reward hacking and no Python logic on data. Every fire was correct and lost nothing.
Every miss failed safe (silent, output byte-identical to plain). There were 0/10 false fires. Recall is low (7/15), with two
generic bugs to fix (F1, F2 below). Two conditions apply:
- The 0/10 false-fire result is optimistic. The builder ran detection on these items before this run (change log part 5).
- Round 2 must confirm the result on fresh variants.

## 1. Code review (CURRENT code: circuit.py 13:41, tables.py 13:21, after change log parts 4 and 5)

### Path of this rule
**Detector** (clause-scoped, read at the clause end):
- `capA = capletters · cw_pos · scope · ¬cw_neg · ¬TIMES · ¬STARTW · ¬FREQ2`
- `capB = lw_neg · ¬FREQ2 · ¬TIMES` ("no lowercase letters")
- Both are multiplied by the request guards `ce = clause end · ¬(non-request question) · ¬(OTHERS/DESCV) · reqm`.
- `capletters` requires one of:
  - an UPPERW word ("uppercase", "upper case", "block letters", "all-caps");
  - CAPW together with LETTER ("capital letters");
  - CAPSW ("caps", "capitals") directly after an ALLQ or INW token.
- Negation is position-sensitive: a case word counts as negated only if the negation comes before it in the clause.
- The clause boundaries are `, . ; ! ? \n` plus "and"/"but".

**Turn scope:**
- System-message latches persist for the whole conversation.
- User-turn latches are dropped at each new user turn.
- A cancel cue drops the system latches.

**Enforcement** (case transducer):
- The circuit reads the host's top token (the disclosed readout).
- If that token contains lowercase letters, the emitter respells it through the fixed UPSEQ table (each token's uppercase
  spelling, at most 6 tokens).
- It force-writes those tokens one at a time. The host then continues from its own uppercase text.

### Reward hacking, test fitting and leakage: none found
- The word lists are generic English: case words, letter words, all/only/entire, negations, request markers.
- No IFEval sentence, template or entity is present.
- The TIMES and FREQ2 vetoes exist for capital-word-frequency instructions. Our own generator has such distractors ('capfreq'),
  so dev data justifies the vetoes.
- The `drv:` caps templates (keys 3186, 3572, 3456) are rewordings of even-key sentences. The author allows this for dev data.
- No dev, fresh or OOD material appears in the circuit or the tables.

### Python logic at inference: none
- All decisions are `torch.where`, gather and clamp operations on 0/1 wires, integer registers and fixed tables.
- Python loops run only over static lists: rule names and program prefixes.
- Regex appears only in the offline table build.
- `report()` is a diagnostic only.

### Disclosure: fixed
- The round-1 finding was that the transducer uses the identity of the host's top token, beyond the "three scalar readouts".
- The docstring now discloses the full readout: top-token identity, masked hard-max, and no solved vectors.

### Open generic issues
These are not hacks. They are bugs or limits that a fix could address without fitting the test.

**F1. "ALL CAPS" in capitals is not recognised. This is the commonest way people write it. (Missed item: 702009.)**
- `capsall` checks the previous TOKEN (`win[-2]`) for ALLQ/INW.
- In the Qwen vocabulary, " CAPS" is not one token ("ĠCAPS" is absent; "ĠCAP" + "S" are present). So CAPSW fires on the last
  piece, and the previous token is " CAP", not " ALL".
- Lowercase "all caps" works only because " caps" is a single token.
- Fix: test the token before the FIRST piece of the matched n-gram, or keep a "previous word" register.

**F2. A plural-possessive apostrophe opens a single-quote mask that runs to the end of the line. (Missed item: 702008.)**
- "kids' clothes" tokenizes as " kids" + "'".
- A bare "'" matches OPENSQ. The CLOSESQ test needs a quote that no letter follows, and "'m" and "'ll" do not match it.
- So every cue word after "kids'" in that line is masked. The garage-sale message has no newline, so its caps instruction is
  masked.
- Any message with "students'", "parents'" and so on before the instruction fails the same way, for every clause rule.
- Fix: a "'" directly after a letter (no space) is an apostrophe, not an opening quote. Also cap the mask length, as the
  forbidden-words reader already does at 12 tokens.

**F3. The request-marker requirement drops verbless fragments. (Missed items: 702003, 702013.)**
- Examples are "..., all in uppercase, ..." and "... and with every letter in uppercase."
- The new and/but clause split cuts the verb off, and `reqm` is clause-scoped.
- Fix: inherit the request marker from the earlier clauses of the same sentence. A sentence-level `reqm` is acceptable when the
  case clause has no guard hit of its own.

**F4. By-design misses (fail safe; acceptable, listed for completeness):**
- A guard word anywhere in a colon-joined clause ("everyone ... : write all of it in capital letters"; 702010).
- "from beginning to end" hits the STARTW veto (702011).
- A paraphrase with no cue word ("SHOUTING", "every letter big"; 702014).
- No carry-over of a user-turn "from now on" (702007). This is the deliberate turn-scope choice, which trades multi-turn support
  for safety.

**F5. Remaining false-fire risk (not tested in this round; for round 2):**
- `capB` has no scope requirement, so "Don't lowercase the brand names" latches whole-response CAPS.
- Mirror case on the lowercase rule: "don't write in all caps" latches LOWERCASE (lowB), although the user means normal case.
- Both should need a whole-response or scope word, as capA does.

**F6. Minor:**
- UPSEQ truncates at 6 tokens; it is rare for a token's uppercase spelling to need more.
- When caps fires, the 0.6B host reads off-distribution uppercase context, and content quality visibly drops ("CATS KNEAD
  BLANCS ..."; "GIVE IT A FRESH TON"). The checker still passes. This cost is in the host, not the detector, but it should be
  stated.
- The official english_capital checker also requires langdetect == 'en'. It failed both plain and implant on 702001: the output
  was already all caps and byte-identical between the two.

## 2. Rule 13: is f(x) useful in general?
**The function.** When the user asks for the whole reply in capital letters, the tool respells every emitted token into its
uppercase form, inside the decode loop. The model then continues from its own uppercase text.

**Three settings outside the benchmark:**
1. Telegraph, radio and aviation message formats (NOTAM, METAR, teleprinter or TELEX output).
2. Text meant to be printed as signs, placards, banners or headlines.
3. Legacy forms, terminals and mainframe or EDI fields that accept only uppercase.

**USEFUL-GENERALLY: yes** (moderately). A post-processing `.upper()` does nearly the same. The in-weights version adds that the
host conditions on the uppercase text, and that it composes with other forced output such as quotes, end phrases and the
title.

## 3. Variants and results
Run: plain vs implant, greedy, Qwen3-0.6B with thinking off. "strict" is the official checker.
"silent=" means the implant output was byte-identical to plain.
POS = positive item, NEG = negative item.

| key | type | what it tests | latched | strict plain → implant | outcome |
|---|---|---|---|---|---|
| 702000 | POS | "IN BLOCK LETTERS ONLY:" at the start | caps | F → **T** | gain |
| 702001 | POS | casual chat, "all caps pls" mid-message | caps | F → F (identical) | fired; plain already all caps; checker fails (langdetect) |
| 702002 | POS | "in all caps and do not mention brands" | caps | T → T (identical) | correct (the round-1 lowercase flip is fixed) |
| 702003 | POS | "... and with every letter in uppercase" | - | T → T | miss, fail-safe (F3) |
| 702004 | POS | "not a single lowercase letter" | caps | F → **T** | gain |
| 702005 | POS | SYSTEM message: teleprinter persona | caps | T → T (identical) | correct |
| 702006 | POS | multi-turn: caps asked in the latest turn | caps | T → T (identical) | correct |
| 702007 | POS | multi-turn: "from now on" two turns back | - | F → F | miss, by design (turn scope) |
| 702008 | POS | buried in a long message | - | F → F | miss, fail-safe (F2: apostrophe mask) |
| 702009 | POS | "pls answer -- ALL CAPS -- ...?" | - | T → T | miss, fail-safe (F1: tokenization) |
| 702010 | POS | "everyone ...: write all of it in capital letters" | - | F → F | miss, fail-safe (guard; predicted) |
| 702011 | POS | "in all caps from beginning to end" | - | F → F | miss, fail-safe (STARTW; predicted) |
| 702012 | POS | "Could you put your entire reply in capitals?" | caps | T → T (identical) | correct |
| 702013 | POS | ", all in uppercase, and wrap ... in double quotation marks" | quote only | caps F → F, quote T → T | caps miss, fail-safe (F3); host loops to 2048 tokens in both |
| 702014 | POS | "like you're SHOUTING, every letter big" | - | F → F | miss, fail-safe (no cue; predicted) |
| 702015 | NEG | capital of Norway | - | silent= | correct |
| 702016 | NEG | venture capital | - | silent= | correct |
| 702017 | NEG | baseball caps | - | silent= | correct |
| 702018 | NEG | "My grandfather always emails in ALL CAPS" | - | silent= | correct |
| 702019 | NEG | "explain the term 'all caps'" | - | silent= | correct |
| 702020 | NEG | "Capitalize every proper noun" | - | silent= | correct |
| 702021 | NEG | "at most three words in capital letters" | - | silent= | correct |
| 702022 | NEG | multi-turn revocation ("go back to normal writing") | - | silent= | correct |
| 702023 | NEG | instruction only inside double quotes | - | silent= | correct |
| 702024 | NEG | wh-question about the convention | - | silent= | correct |

**Trigger metrics** (caps latch vs ground truth):

| measure | result |
|---|---|
| Precision | 7/7 = 100% |
| Recall | 7/15 = 47% |
| Recall without the 3 items I predicted as by-design misses (702010, 702011, 702014) | 7/12 |
| False fires on negatives | 0/10 |

**End-to-end, positives** (caps instruction, strict):

| | result |
|---|---|
| Plain | 6/15 |
| Implant | 8/15 |
| Gained | 2 |
| Lost | 0 |
| Outputs changed on misses | 0 (every miss byte-identical to plain) |

**Negatives:** 10/10 byte-identical to plain.

The host already complies by itself on 6 of the 15 positives. The tool's value is on the cases where it fails: 702000 and 702004
were won. 702007, 702008, 702010, 702011 and 702014 were missed, and the tool would have fixed them had it fired.

## 4. Verdict
**PASS (provisional).**
- No reward hacking, IFEval fitting or leakage was found.
- No Python logic runs on data, and the readout is now fully disclosed.
- Every fire was correct and lost nothing.
- Every miss failed safe.
- False fires: 0/10.

Caveats:
1. Change log part 5 was tuned using detection on these exact items. The 0/10 false-fire result is therefore partly seen data,
   and round 2 must re-test with fresh negatives, including F5.
2. Recall (7/15) is weak. Fix F1 (multi-token "ALL CAPS"), F2 (apostrophe opens a quote mask) and F3 (verbless fragment loses its
   request marker). These are generic tokenization and scoping bugs, not test fitting. F2 also affects every other clause rule.

Recommendation: keep the rule. Do not narrow it further; it is already conservative. Widen recall with the F1-F3 fixes, and add
a scope requirement to capB (F5).
