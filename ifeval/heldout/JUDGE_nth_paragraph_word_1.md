# JUDGE nth_paragraph_word, round 1

**Tool:** paragraph k must start with word X, with N paragraphs separated by blank lines.
**Detector:** a start verb, the word "paragraph", an ordinal / "paragraph k" / "last", and "with" plus a word, all in one sentence.
**Enforcement:** force X (capitalised) at the start of paragraph k, plus the paragraph count.
**Code reviewed:** the CURRENT circuit.py and tables.py (after NOTES change log parts 4 and 5).
**Variants:** judge/nth_paragraph_word_variants.jsonl (keys 710000 to 710020).
**Run results:** judge/nth_paragraph_word_results_r1.md.

Caveat: the builder ran detection on these variants before writing part 5. So these items are partly "seen". This round is not a clean held-out measure, and round 2 needs fresh variants.

## 1. Code findings (current code)

### Reward hacking, test fitting, Python logic: none found
- **Detection** is tensor ops on 0/1 and integer registers:
  - `armw`: arms the capture when a start verb (STARTV) is in the sentence and the current token is "with".
  - `cap0` / `cont`: capture the span of X by pointer. They skip the, word, term, a, an, quote openers and colons.
  - `s_k` / `s_klast`: read k from an ordinal word or digit+suffix, a number within 2 tokens after "paragraph", or last/final.
  - `nth_hit`: fires at the sentence end, through the request, question, third-person and negation guards.
- **Enforcement** is `n_go` (k=1, at the answer start) and `n2` (at a blank-line break when kp == k-1). Both copy the captured span through the CAP1 table.
- **Gate:** enforcement needs the paragraph-count latch with N ≥ 2 and blank-line mode (no `***`).
- No Python branching on data at inference. `report()` is diagnostic only.
- **Word lists are generic English:**
  - STARTV: start, begin, open, lead, and their inflections.
  - WITHW: "with".
  - Ordinals: first to tenth.
  - SKIPNTH: the, word, term, a, an.
  - PARAW: paragraph, para.
- X is captured by pointer. There is no closed list of target words; the dev word list exists only in gen_data.py.
- No IFEval strings or templates are compiled in. The phrase shape ("start ... paragraph ... with the word X") is ordinary English.
- Regular expressions run only offline, to build the tables.
- **Change-log fixes are generic:** start verb instead of the ordinal "first"; request and negation guards; turn scope; single-quote masking.

### Weaknesses still present in the current code
1. **No request marker is required for nth.** Part 5 added this to the clause rules only. So a plain first-person statement passes every guard and latches: "My essay has 4 paragraphs, and the second paragraph begins with the word However." This caused the one real false fire (710013). The same gap explains the latent latch in 710020 ("Our book club meeting will open with a reading of the first paragraph").
2. **Negation is scoped to the whole sentence.** "... starting the fourth paragraph with the word Additionally and avoiding any commas." fails because "avoiding" vetoes nth, even though "and" is now a clause boundary (710004 missed).
3. **Capture takes the first word after "with"** that is not the, word, a, an. "start paragraph 3 with an example" would capture "example". This is protected only by the paragraph-count gate.
4. **Recall needs the word "with" and a start verb.** These miss, and fail safe for nth:
   - "kick off" (no start verb);
   - "closing paragraph" (not read as last);
   - "The opening word of paragraph 2 must be X";
   - the word stated before the cue.
5. **Enforcement quality (shared with the paragraph-count tool).** The forced first word and the end-of-turn to blank-line swap often lead the host to re-copy earlier text. The checker passes, but the content is degraded:
   - 710003: paragraph 2 repeats the last sentence of paragraph 1 verbatim.
   - 710005: the email body repeats, with a "**[Blank line]**" artifact.
   - 710010: paragraph 1 repeated at the end.
   - 710000: trailing "-  \n-" paragraph.
   - 710002: a second, invented memo.

## 2. Rule 13: is f(x) generally useful?
- **f(x):** given a requested paragraph count N (blank-line separated), an index k (ordinal, "paragraph k", or last/final) and a word X, emit a reply whose paragraph k begins with X.
- **USEFUL-GENERALLY: yes.** Three settings outside the benchmark:
  1. Legal and contract drafting with mandated lead words ("Whereas ...", "Accordingly ...").
  2. Templated business writing, for example status emails whose delays paragraph opens with "Unfortunately", or a report whose last paragraph opens with "Summary".
  3. Structured teaching material and form letters (fixed lead words per paragraph, acrostic exercises).

## 3. Variants and results
"Latched" means the circuit latched the nth rule and recorded k, X and N. "nth strict" is the official checker for this instruction, plain → implant. "Identical" means the implant output is byte-identical to plain.

### Positives (13)

| key | what it tests | latched (k, X, N) | nth strict | identical |
|---|---|---|---|---|
| 710000 | casual, no punctuation, "paras", "open para 2" | yes (2, honestly, 3) | F → T | no (trailing junk "-\n-") |
| 710001 | middle of message, "4-paragraph", "3rd" | yes (3, Remember, 4) | F → T | no |
| 710002 | only in SYSTEM message, "final paragraph ... starts" | yes (last, Accordingly, 3) | F → T | no (invents a 2nd memo) |
| 710003 | multi-turn follow-up, "make paragraph 2 start" | yes (2, Today, 2) | F → T | no (paragraph 2 duplicates text) |
| 710004 | nth + no commas in one sentence, "starting" | **no** (negation "avoiding" in sentence) | T → F | no (comma tool) |
| 710005 | buried in long message, "lead with" | yes (2, Unfortunately, 3) | F → T | no (repetition) |
| 710006 | word before cue, no "with" | no (expected) | F → F | **yes** |
| 710007 | "kick off" | no (expected); count only | F → T | no |
| 710008 | "closing paragraph" | no (expected); count only | F → T | no |
| 710009 | nth before count, "paragraph two", ";" | yes (2, Interestingly, 4) | F → T | no |
| 710010 | request as question ("Could you ...?"), k=1 | yes (1, Picture, 3) | F → T | no |
| 710011 | "opening word of paragraph 2 must be" | no (expected); count only | F → F | no |
| 710012 | "last paragraph has to begin", word + comma | yes (last, Finally, 3) | T → T | yes (forced token = host's) |

**Totals on positives:**
- Detection: 8 of 13 latched, and all 8 had correct arguments (k, X and N).
- Of the 5 misses, 4 were by design (no "with" or no start verb) and 1 came from the negation guard (710004).
- nth checker passes: 2/13 plain → 10/13 implant.
- Prompt level: 1/13 → 10/13 (710004 is lost on nth but gained on commas, and fails at prompt level both ways).

### Negatives (8): the nth tool must stay silent

| key | what it tests | latched | enforced | identical |
|---|---|---|---|---|
| 710013 | user DESCRIBES own essay ("second paragraph begins with the word However") | **yes (2, However, 4)** | **yes**: adds a "However, ..." paragraph and filler | no: **FALSE FIRE** |
| 710014 | "compare the first paragraph with the last one", 3 paragraphs | no (fixed in part 4) | count only (a real request) | no (count tool, legitimate) |
| 710015 | "rewrite the first paragraph with simpler words", 2 paragraphs | no | no | yes |
| 710016 | instruction inside single quotes (teacher's rubric) | no (fixed in part 5) | no | yes |
| 710017 | instruction inside double quotes (translate) | no | no | yes |
| 710018 | "what does it mean when ... 2nd paragraph must start with ...?" + "I have to write 4 paragraphs" | no | count latched N=4 (count-tool false fire) | no |
| 710019 | stale nth instruction from a HISTORY turn | no (fixed in part 5) | no | yes |
| 710020 | "meeting will open with a reading of the first paragraph" | **yes (1, reading)** | no (no paragraph count; gated) | yes |

**Totals on negatives:**
- nth latched on 2 of 8. One was enforced (710013); one was latent and gated off (710020).
- Byte-identical to plain: 5 of 8. 710014 differs legitimately (count tool); 710018 differs because of a count-tool false fire.

### Trigger precision and recall (nth)
- **Recall** (latched with correct arguments): 8/13 = 62%. Excluding the 4 by-design misses: 8/9.
- **Latch precision:** 8 / (8 + 2) = 80%.
- **Enforced-fire precision:** 8 / (8 + 1) = 89%.
- **False-fire rate on negatives:** 1/8 enforced (12.5%), 2/8 latched.

## 4. Verdict: FAIL (narrow the rule; do not drop it)

The code is clean: no reward hacking, no IFEval strings, and no Python logic at inference. The function is generally useful. On positives, detection is precise (8/8 correct arguments), and the nth checker goes from 2/13 to 10/13. Misses mostly fail safe for nth.

The verdict is FAIL because the brief requires a false-fire rate near zero:
1. **710013 is a real false fire on a natural prompt** ("My essay has 4 paragraphs, and the second paragraph begins with the word However. Is that a weak transition?"). The implant appended an unrequested "However, ..." paragraph and filler. The cause is generic: nth needs no request marker, unlike the clause rules since part 5.
2. **710020 latches on a narrative sentence** ("will open with a reading of the first paragraph"). It is harmless only because no paragraph count was present.

**Required fixes (generic, small):**
- **(a) Require a request marker in the nth sentence:** the REQMARK class (you, your, please, must, should, need, have to, make, keep, ...) or a start verb in imperative position. Check this against our variants: all 8 true positives carry one ("must / should / has to / please / you / make / keep" or an imperative "Begin"), and both false latches (710013, 710020) carry none. Verify it on dev and seed-2 rather than on these items.
- **(b) Scope the negation guard to the clause that holds the start verb,** not the whole sentence. This recovers 710004.

**Recommended:**
- Accept "the word X" or a quoted or capitalised X more readily than a bare noun after "with an/a".
- Look at the duplicated-text and filler outputs that the forced paragraph breaks produce (710000, 710002, 710003, 710005, 710010). The checker passes them, but the user-facing quality is worse than plain in several cases. The source is mainly the paragraph-count tool's end-of-turn to blank-line swap.

Re-judge in round 2 with fresh variants. These items are partly seen because they informed the part-5 changes.
