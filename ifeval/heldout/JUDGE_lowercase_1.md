# JUDGE_lowercase_1 — rule-keeper "lowercase" (change_case:english_lowercase)

Judge: independent reviewer, round 1. Variants: `judge/lowercase_variants.jsonl` (keys 701000-701022; 13 positive, 10 negative).
Run results: `judge/lowercase_results_r1.md` (builder's run; circuit after change-log parts 4+5, BEFORE the lw_neg/cw_neg
negation-order fix). Files read: JUDGE_BRIEF.md, PROTOCOL.md, circuit.py, tables.py, gen_data.py, NOTES.md, run.py, score.py,
detect_test.py, data_mentions.py, the results file. Not opened: data/fresh.jsonl, fresh_templates.json, data/ood.jsonl, any IFEval data.
No model was run on the development machine.

## VERDICT: FAIL (narrow, do NOT drop) — re-judge in round 2 after two generic fixes

The tool contains no reward hacking. The case transducer works: it enforced correctly on every positive where it latched (8/8).
Every missed positive except one failed safe. Two problems block a PASS:
1. **Wrong-direction enforcement in the current code (blocking).** The post-run negation-order fix (lw_neg/cw_neg) counts a
   negation only when it comes BEFORE the case word. A negation that comes AFTER the case word now latches the OPPOSITE rule:
   - "Capital letters are not allowed in your response." -> capA = capletters x cw_pos x scope x reqm('your') = 1 -> **ALL CAPS
     enforced** (the user asked for lowercase).
   - "Uppercase letters should not appear anywhere in your answer." -> caps (same path).
   - "Lowercase letters must not be used in your reply." -> lowA -> **lowercase enforced** (the user asked for caps).

   I found this by reading the code (circuit.py lines 245-253). It is not in the run. The r1 code (negation anywhere in the clause)
   handled these sentences correctly through lowB/capB, so the fix for 701009 added this new failure. In r1 the same class of error
   already happened once on my items: 701009, "Write in lowercase without using any emojis", latched CAPS and turned a passing
   plain answer into an all-caps failure. A wrong-direction fire is the worst outcome, because the output is changed AND wrong.
   It is not fail-safe.
2. **The request marker is a bag of words (minor; 1/10 false fire).** `reqm = cl('REQMARK') + ...` accepts 'write/use/keep/you...'
   anywhere in the clause, also when the clause describes a third party. 701020 "I read that some brands write their logos in
   lowercase to look friendly." latched lowercase ('write' is REQMARK; 'brands' is not in OTHERS). The reply was rewritten in
   lowercase.

Required fixes (generic, not test-specific):
- (a) Negation AFTER a case word in the same clause: by default latch NEITHER case rule (fail-safe silence). Flip polarity only
  when the negation is followed by a predicate of the case word itself (the APPEARS class already exists: allowed/used/permitted/
  appear/... -> "X letters are not allowed" = opposite case). "lowercase without (using) any emojis" then stays lowercase
  only if the negation clearly takes another object; otherwise it stays silent. Silence is acceptable; the wrong case is not.
- (b) A REQMARK word counts only in request position (clause-initial imperative, after please/also/..., or with a 2nd-person subject
  "you/your"). A content verb with a third-person/noun subject ("brands write", "my teacher keeps") does not mark a request.
- Then re-judge with FRESH variants (round 2): my items are no longer unseen (see the caveat below).

## 1. Code review (current code, circuit.py / tables.py, lowercase path)

Detector (clause scope, read at clause end; clause boundaries , ; . ! ? newline and/but):
- `lowA = lw_pos * scope * (1-lw_neg) * (1-TIMES) * (1-STARTW)`: a lowercase-word, a scope word (all/only/in/using/with/entire/
  response/letters...), and no negation before the lowercase-word.
- `lowB = nocap * cw_neg * (1-FREQ2) * (1-TIMES)`: a capital-word ABOUT LETTERS (uppercase / capital+letter(s) / caps / capitals)
  with a negation before it.
- Guards: `ce = clause_end * rqc * reqm`. rqc = not an information question and no third-person subject/verb (OTHERS/DESCV). reqm =
  a request marker. Quoted spans (both double and single quotes) are masked.
- Turn scope: latches from earlier user turns are dropped at each new user turn. System-message latches persist. Cancel cues
  (back to normal / this time / plain text / ...) drop the system latches.
- caps vs lower: `g['caps'] *= (1 - g['lower'])`, so lower wins if both latch.

Enforcement: `need_case = g['lower'] * UPPERSET[chosen]`. The emitter re-spells the host's chosen token through LOWSEQ (that
token's lowercase string, re-tokenized, max 6 tokens) and forces those tokens. The host then conditions on the re-spelled text.

Findings:
- Reward hacking / IFEval fitting: **none found.** The word lists (LOWERW, CAPW, UPPERW, CAPSW, NEG, ALLQ, INW, WHOLE2, REQMARK,
  STARTW, TIMES) are generic English. They contain no IFEval sentences or templates and no strings that only make sense for
  IFEval. The gen_data templates tagged drv:<even key> are allowed by the author for dev data. Nothing indicates leakage from
  odd keys or fresh data.
- Python logic at inference: **none in this path.** Everything is table lookups, 0/1 wire arithmetic, torch.where/gather and
  argmax. Python loops run only over fixed rule names. `start_prog` has a Python loop over a fixed prefix list (structural, not data).
- Readout: the transducer reads the identity of the host's top token, i.e. the argmax over the frozen LM head. The docstring
  now discloses this (part 4: "identity of the host's top token ... of the wrong case"). OK.
- Generality limits (all are fail-safe misses, acceptable):
  - The scope-word + request-marker requirement drops terse noun-phrase instructions ("LOWERCASE ONLY!!!", 701007).
  - 'capitalization' no longer counts as a letters word (701011 "No capitalization at all, please." missed; a deliberate trade
    that buys 701015, "Don't worry about capitalization", staying silent).
  - Idioms ("pretend your shift key is broken", 701010) are not understood.
  - A participle clause without a marker ("..., written entirely in lowercase and without any commas", 701002) is missed.
- Guard lists are short closed lists, written against the team's own mention negatives, not IFEval: OTHERS (he/she/they/people/...)
  and DESCV (writes/types/texts/prefers/likes/capitalizes/says). These are the reason for finding 2 above. They miss other
  subjects ('brands', 'my boss') and verb forms ('said', 'signs', 'wrote'). The REQMARK position fix (b) helps more than
  extending these lists.
- Minor: a LOWSEQ expansion is capped at 6 tokens (`e[:6]`). A pathological token whose lowercase spelling needs more than 6
  tokens would be cut. Rare; not seen.
- Multi-turn: the r1 concern that a latch persists after a retraction is fixed (701016 was silent and byte-identical). As
  disclosed in part 5, carry-over ("same rules as before") now fails safe.

## 2. Rule 13 — is f(x) generally useful?

f(x): "From the moment a request for all-lowercase output is detected, every token the model emits is re-spelled to its lowercase
form, so every letter of the reply is lowercase. Nothing else changes."
USEFUL-GENERALLY: **yes** (for the function). Three settings outside the benchmark:
1. Generating slugs, tags, usernames, hashtags and identifiers that must be lowercase.
2. Brand or persona style guides with an all-lowercase voice (chat personas, product copy, social posts).
3. Writing into case-normalised or case-insensitive pipelines: keys, emails, search indices, config values that must be lowercase.

The detector, not the function, limits how useful the tool is. Positive: it stays silent on code tasks that merely mention
lowercasing (701013), so it does not break `toLowerCase` identifiers.

## 3. Variants and results (r1 run; "lat" = latched rules; P/I = plain/implant strict pass on the expected ids)

| key | type | what it tests | lat | P | I | identical | outcome |
|---|---|---|---|---|---|---|---|
| 701000 | POS | casual "ur/pls", cue first, "i hate caps" clause | lower | T | T | yes | correct (host already lowercase) |
| 701001 | POS | synonym "small letters", instruction in the middle | lower | F | T | no | **gain** |
| 701002 | POS | lowercase + no-comma in one participle clause | nc | T,T | T,T | yes | lower missed, fail-safe (plain already passes) |
| 701003 | POS | instruction only in SYSTEM message | lower | T | T | yes | correct |
| 701004 | POS | multi-turn follow-up, "don't use any uppercase letters" | lower | F | T | no | **gain** (but see the quality note) |
| 701005 | POS | buried in a long message | lower | T | T | yes | correct |
| 701006 | POS | "never use a capital letter", proper noun forced | lower | F | T | no | **gain** ("isaac newton") |
| 701007 | POS | "LOWERCASE ONLY!!!" at start, no marker | - | F | F | yes | missed, fail-safe |
| 701008 | POS | "Start ... and keep all of it lowercase:" | lower | T | T | yes | correct |
| 701009 | POS | "lowercase without using any emojis" | **caps** | T | **F** | no | **WRONG DIRECTION** (ALL CAPS); builder reports detection fixed after the run (not re-generated) |
| 701010 | POS | idiom: shift key broken / caps lock doesn't exist | - | F | F | yes | missed, fail-safe |
| 701011 | POS | "No capitalization at all, please." + proper nouns | - | F | F | yes | missed, fail-safe (by design) |
| 701012 | POS | lowercase + end phrase in one sentence | lower,end | T,F | T,T | no | correct (end tool appended the phrase) |
| 701013 | NEG | JS function: convert email to lowercase | - | | | yes | silent, correct |
| 701014 | NEG | "Explain the difference between lowercase and uppercase" | - | | | yes | silent, correct |
| 701015 | NEG | "Don't worry about capitalization or spelling" | - | | | yes | silent, correct |
| 701016 | NEG | multi-turn retraction ("back to normal capitalization") | - | | | yes | silent, correct |
| 701017 | NEG | friend "who always signs his emails in lowercase" | - | | | yes | silent, correct |
| 701018 | NEG | instruction inside double quotes to proofread | - | | | yes | silent, correct |
| 701019 | NEG | "the lower case of my display cabinet" | - | | | yes | silent, correct |
| 701020 | NEG | "some brands write their logos in lowercase" | **lower** | | | no | **FALSE FIRE** (reply lowercased) |
| 701021 | NEG | "Explain what 'lowercase' means" (single quotes) | - | | | yes | silent, correct |
| 701022 | NEG | "Why do ... brands use lowercase logos?" | - | | | yes | silent, correct |

Trigger statistics (lowercase latch, r1 run):
- Positives: TP 8, FN 5. 4 of the 5 FN are silent and byte-identical; 701009 latched the WRONG case rule.
- Negatives: FP 1 (701020), TN 9. All 9 TN are byte-identical to plain.
- **Precision 8/9 = 0.89, recall 8/13 = 0.62.** With the builder's post-run fix (detection only, not re-generated): recall 9/13 = 0.69.
- Wrong-direction fires: 1/23 in the run (701009). The current code has a new wrong-direction path by reading (verdict, point 1).
- End-to-end, lowercase instruction strict pass on the positives: plain 7/13 -> implant 9/13 (gained 701001, 701004, 701006; lost 701009).
  The host often writes lowercase by itself when asked. The real value is enforcement on capitalised starts and proper nouns.
- Enforcement quality: forced-token misses 0 on all latched items, no degeneration.
  - Quality note for 701004: the implant reply repeats the fact from the history turn ("octopuses have three hearts and blue
    blood ...") instead of giving two NEW facts. The plain reply gave new facts but capitalised them. The checker passes, but the
    content got worse.

## 4. Caveats

- These items are no longer fully unseen. Part 5 of the change log was informed by running detection on all round-1 judge
  variants, including these 23. The 9/10 negative silence and the multi-turn fixes are therefore partly fitted to this set, so
  the numbers above are optimistic. A round-2 judge must write fresh items, and they should include post-posed negations of case
  words ("Capital letters are not allowed in your reply", "Lowercase letters must not be used") and third-party descriptions with
  a request verb ("my editor writes headlines in lowercase").
- The 701009 fix was checked by detection only, as reported by the builder. I did not see its generation.
- Finding 1 (verdict) comes from reading the code. It should be confirmed with detect_test on those three sentences before the fix
  is designed.
