# JUDGE_lowercase_2 — rule-keeper "lowercase" (change_case:english_lowercase), round 2

Judge: the same independent reviewer as round 1 (JUDGE_lowercase_1.md: FAIL, narrow).
Code judged: the CURRENT circuit.py / tables.py (NOTES change log parts 6 + 7).
Variants: `judge/lowercase_variants_r2.jsonl` (keys 701100-701120; 12 positive, 9 negative; 701114 is a caps request and a negative
for lowercase). Results: `judge/lowercase_results_r2.md` (generated with the part-7 code).
Not opened: fresh/ood data, fresh_templates.json, IFEval data. No model was run on the development machine.

## VERDICT: PASS (conservative detector; high precision, moderate recall)

- No reward hacking, no IFEval fitting, no Python logic at inference in this path.
- On the 21 fresh items: **0 false fires** (0/9 negatives) and **0 wrong-direction fires** (0/21).
- Every miss failed safe: 6/6 missed positives were silent and byte-identical to plain.
- All 6 latched positives passed the official checker. The tool gained 3 prompts and lost 0.
- Both round-1 blocking issues are fixed:
  - negation after the case word ("Capital letters are not allowed ..." -> lowercase);
  - the request marker no longer accepts descriptions as requests.
- My two new wrong-direction risks from the round-2 code review ("lowercase letters with no hashtags allowed", "all caps with no
  emojis allowed") are now silent: the part-7 rule flips polarity only when allowed/used comes directly after the negation.

The PASS criteria are met: correct or fail-safe on 12/12 positives, false-fire rate near zero, USEFUL-GENERALLY yes.
Recall is the weak side (6/12). The non-blocking recommendations below would raise it.

Caveat (weighs against the numbers): the builder ran DETECTION on these r2 items while writing part 7, after my r2 code review
had predicted most of the outcomes. So the 0/9 false-fire result and the 0 wrong-direction result are partly fitted to this set.
The fixes I checked by reading are generic, not keyed to my sentences:
- an instruction-verb list for an imperative opening;
- adjacency of the negation and allowed/used for the polarity flip;
- data-after-colon masks.

The final held-out numbers (fresh/ood, then real IFEval) remain the true test.

## 1. Code review (current code)

Detector (circuit.py lines ~255-285, 206; tables.py INSTV/REQANY/PARTW/APPEARS/NEG). Clause-scoped and committed at the sentence end:
- `lowA = lw_pos * scope * (1-lw_neg) * (1-lw_post) * (1-TIMES) * (1-startx) * (1-PARTW)`: a lowercase-word with a scope word, and no
  negation before or after it.
- `lowB = nocap * scope * max(cw_neg, cw_post * flip)`: a capital-word about letters, negated before it, or negated after it with
  allowed/used DIRECTLY after the negation.
- Gates: request marker (an instruction verb from INSTV opening the clause, you/please/must/should/... anywhere, or an opening
  negation at a sentence start), no information question, no third-person / reported / topic / code sentence, quote and
  data-after-colon masks, turn scope (earlier user-turn latches dropped; system latches persist).

Enforcement: unchanged since round 1. The emitter re-spells the host's chosen token (the disclosed top-token readout) through the fixed
LOWSEQ table. Forced-token misses were 0 on every latched item.

Findings:
- Reward hacking / IFEval fitting: **none.** The word lists are generic English: INSTV (about 90 instruction verbs), PARTW
  (just/title/heading/name/acronym), REPORTV, CODEW, APPEARS. No IFEval templates or strings. No leakage path.
- Python logic at inference: **none in this path.** It is tables, 0/1 wires, where/gather, and an argmax readout (disclosed).
- Recall gaps (all fail-safe, not blocking):
  - **Adverb before the verb.** "Always respond using lowercase letters only ..." (701101, a system persona) is missed. 'Always'
    takes the clause-opening position, so 'respond' no longer counts as an imperative opening. 'just' is worse: it is an
    imperative prefix (IMPPRE) AND a partial-scope veto (PARTW), so "Just answer in lowercase." is vetoed.
    Generic fix: add frequency/manner adverbs (always, simply, now, only, also) to IMPPRE. Use 'just' as a veto only when it
    modifies an object ("just the title").
  - **Elliptical case phrase after a comma.** "keep the caps lock off for this one, all small letters." (701100) and "Could you
    write the club motto too, all lowercase this time?" (701102) are missed. The request marker is in the previous clause of the
    same sentence, and the clause with the case phrase has none. This is a very common way to phrase the rule ("Write a haiku,
    all lowercase.").
    Generic fix: a clause with no verb inherits the request marker of the preceding clause in the same sentence.
  - Notes-list instruction ("- formatting: lowercase only", 701103), terse "Capital letters: none." (701106) and the
    hashtags-allowed case (701109) are silent. Silence is the right fail-safe there.
- Cross-tool finding (for the end-phrase judge): in 701110, "please stick to lowercase letters from start to finish: a ceramic mug
  that keeps coffee warm for two hours." latched **end** with the phrase " a ceramic mug that keeps coffee warm for two hours.".
  This is a false end-phrase latch: 'finish' plus a colon followed by data. The output happened to be byte-identical (the host ended
  with that text), but in other cases the end tool would append the phrase.

## 2. Rule 13 — USEFUL-GENERALLY: yes

f(x): "Once the user (or a system message) asks for an all-lowercase reply, re-spell every emitted token to lowercase. Otherwise
change nothing." Three settings outside the benchmark:
1. Slugs, tags, usernames, hashtags and identifiers.
2. All-lowercase brand or persona voices (journaling bots, product copy, social posts). Note: the system-persona form is currently
   missed (701101).
3. Writing into case-normalised pipelines: keys, emails, search indices, config values.

## 3. Variants and results (part-7 code; P/I = plain/implant strict pass on the expected ids)

| key | type | what it tests | latched | P | I | identical | outcome |
|---|---|---|---|---|---|---|---|
| 701100 | POS | casual, "caps lock off ..., all small letters" | - | F | F | yes | miss, fail-safe (elliptical clause) |
| 701101 | POS | system persona "Always respond using lowercase letters only, even for names ..." | - | T | T | yes | miss, fail-safe (adverb opening) |
| 701102 | POS | multi-turn "Could you write ..., all lowercase this time?" | - | F | F | yes | miss, fail-safe (elliptical clause) |
| 701103 | POS | notes bullet "- formatting: lowercase only" | - | T | T | yes | miss, fail-safe (expected) |
| 701104 | POS | "Capital letters are not allowed anywhere in your reply." | lower | F | T | no | **gain**; round-1 blocker fixed |
| 701105 | POS | "Uppercase characters must not be used at all: ..." | lower | F | T | no | **gain** |
| 701106 | POS | "Capital letters: none." | - | F | F | yes | fail-safe (expected), never caps |
| 701107 | POS | task, then "and make every single letter lowercase" | lower | T | T | yes | correct |
| 701108 | POS | "no caps in ur reply plz!!" | lower | F | T | no | **gain** |
| 701109 | POS | "lowercase letters with no hashtags allowed" | - | T | T | yes | fail-safe; the r2 wrong-direction risk is fixed (no caps) |
| 701110 | POS | mid-sentence "please stick to lowercase letters from start to finish:" | lower (+end, spurious) | T | T | yes | correct for lower; end false latch (cross-tool) |
| 701111 | POS | "Write it entirely in lowercase, and skip any emojis." | lower | T | T | yes | correct |
| 701112 | NEG | "Honestly, lowercase letters look friendlier on a logo." | - | | | yes | silent |
| 701113 | NEG | "Developers usually write ... in lowercase" | - | | | yes | silent |
| 701114 | NEG* | "Write in all caps with no emojis allowed" (caps request) | - | T(caps) | T(caps) | yes | lowercase silent (no wrong direction); caps also silent |
| 701115 | NEG | typewriter "only had capital letters" | - | | | yes | silent |
| 701116 | NEG | "Is 'iPhone' supposed to start with a lowercase letter?" | - | | | yes | silent |
| 701117 | NEG | earlier-turn lowercase, now "with proper capitalization" | - | | | yes | silent |
| 701118 | NEG | "Convert these CSS class names to lowercase letters ..." | - | | | yes | silent |
| 701119 | NEG | "Lena answers every message in lowercase letters." | - | | | yes | silent |
| 701120 | NEG | "Translate this into Spanish: Please write your answer in lowercase letters." | - | | | yes | silent |

Trigger statistics (lowercase latch):
- Positives: TP 6, FN 6 (6/6 silent and byte-identical). Negatives: FP 0, TN 9 (9/9 byte-identical).
- **Precision 6/6 = 1.00, recall 6/12 = 0.50.** Wrong-direction fires: 0/21.
- The lowercase instruction passed (strict) on 6/12 positives in plain and 9/12 with the implant. Gained 701104, 701105, 701108; lost none.
- Combined with round 1 (on older code, so not directly comparable): the tool never lowered a score through a correct latch.
  The only harm across both rounds came from fires on the wrong rule (r1 701009) or on non-requests (r1 701020). Both classes
  are fixed in the current code, and neither appears in this round.

## 4. Recommendations (non-blocking)

1. Re-run DETECTION on the round-1 items (701000-701022) with the part-7 code to confirm part 7 did not lose them. In round 1,
   8/13 positives latched. Several used forms that part 7 tightened, for example "write ur whole answer in all lowercase pls".
2. Recall fixes (generic): adverb prefixes in IMPPRE; a verbless clause inherits the request marker of its sentence; 'just' as a
   veto only before a noun object.
3. Forward the 701110 end-phrase false latch to the end-phrase owner.
