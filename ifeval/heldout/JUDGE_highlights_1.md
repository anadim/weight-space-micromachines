# JUDGE highlights #1: highlight >= N sections (`detectable_format:number_highlighted_sections`)

Judge: independent reviewer (round 1). Code reviewed: circuit.py (CURRENT version, after change log parts 4 and 5), tables.py,
NOTES.md, gen_data.py, run.py. Not opened: fresh*, ood, any IFEval data. No model ran on the development machine.
Variants: judge/highlights_variants.jsonl (keys 712000-712024). Results: judge/highlights_results_r1.md.

**Verdict: PASS (conditional).** There is no reward hacking and no Python logic. Detection recall is 14/16, and both misses fail
safe. On 9 negatives the output did not change. The conditions are a round 2 with unseen items and the narrowing items in section 5.

## 1. Code findings (current code)

Path: circuit.py
- reader: lines 334-338 (the count registers)
- detector: lines 409-414 (`h_hit`)
- turn scope: lines 441-457
- answer tracker: lines 515-523
- gate: line 567
- emitter: the `hc` close and the `c4` append programs (the `hsep` + `*` program)

Word lists in tables.py: HIGHW, ITALW, MARKDOWN, ASTERISK, CANCELW, TIMES, NEG, FORGETW.

- **Python logic at inference: none found.** All decisions are `torch.where` / products on 0/1 wires, integer registers, fixed
  tables and pointer reads. The loops (`for k in self.L`, program lists) go over fixed rule keys, not over data. Regex is used
  only in the offline table build. The token ids 872/8948/77091 (role names) are tokenizer constants. The readouts are the disclosed ones.
- **IFEval-specific vocabulary or templates: none.** The lists are generic English: highlight/bold/emphasize, italic*, markdown,
  asterisk, and generic cancel phrases. The detector shape "highlight word + markdown/asterisk/'*'" follows the usual way that
  people ask for markdown emphasis. It is not one IFEval sentence. Phrasings without those words (for example "bold the 3 key
  terms") stay silent, so the narrowness fails safe.
- **Leakage: none found.** Dev data has 7 'high' templates. 3 are drv rewordings, which the protocol allows.
- **Test-fitting note (disclosed by the builder):** change log part 5 was made after the builder ran detection on all round-1
  judge variants, including these. The removal of 'star', the turn scope, and the third-person/negation guards answer weaknesses
  that I predicted in my preliminary review (items 712017-712020, 712023, 712024 were predicted false fires). These fixes are
  generic, but this item set is now "seen". The precision numbers below are therefore optimistic, and round 2 is necessary.
- **Remaining weaknesses in the current code:**
  1. *Sentence-level negation veto* (`negs`). A negation anywhere in the sentence stops the rule. Effect: "...without using any
     commas and italicize at least three sections" is missed (712003). Clause scope (`negc`; 'and' is now a clause boundary)
     would keep the guard and stop this miss.
  2. *Sentence-level TIMES veto.* "Italicize at least 2 sections, and mention the word reef at least twice" is missed (712015).
     Clause scope would fix this too.
  3. *Count capture.* The rule takes the first number after the cue word, or else the last number before it, with N > 20
     changed to 1. Effects: a numbered rule list "(1) at least 3 italicized phrases, (2) under 120 words" gives N=2, not 3
     (712011). "a few / several / a couple" gives 1 (712009). Recommendation: read the count the same way as the bullet/paragraph
     counts in part 4 (number next to the item word, not after a determiner, no unit word in between).
  4. *New precision risk from part 5.* A highlight/bold word plus ANY number in the sentence now fires. Examples that I expect
     to false-fire (not tested in this round): "Highlight the top 3 attractions in Rome", "Emphasize 2 benefits of exercise",
     "Write a bold 5-step plan". Each would add N italic spans at the end of the reply. Round 2 must test this.
  5. *Information request with 'you'.* "Can you explain why some novelists italicize entire sections?" latches (712016),
     because ITALW alone fires and "can you" counts as a request. The output did not change, because the host already wrote a
     highlight. The EXPLAINV guard that part 5 added for postscript could also apply here.
  6. *Content of the appended spans.* The emitter writes '*' + host text + '*'. This text is often a copy of earlier sentences
     (712007, 712011) or meta text ("*italicize the first section*" in 712014, "*Word count: 150*" in 712009). The checker passes,
     but the content is low quality. This is a quality issue, not a protocol issue.
- **Cross-tool note:** in 712012 ("make at least two parts **bold**") the paragraph/section rule also latched (N_sec=2). The
  section gate needs the *** mode, so it stayed off and the output was byte-identical. This is harmless, but it shows that
  "N parts" plus '**' is read as a section count.

## 2. Rule 13: is f(x) generally useful?

- **f(x):** when a message asks for at least N emphasized spans (markdown `*...*` or `**...**`), count the spans that the model
  has written so far. Before the turn ends, add emphasized spans until the count is N. Do nothing when no such request is present.
- **Settings outside the benchmark:**
  1. Study notes and flashcards that must emphasize key terms.
  2. Skimmable support answers and UI copy where key points must be bold.
  3. Newsletters and marketing copy with required emphasis.
- **USEFUL-GENERALLY: yes.** The interface is generic cue words plus the markdown marker. The function is useful. The open
  question is trigger precision, not the function.

## 3. Variants and results (rule 11, rule 12)

Out-of-style settings covered: system message (712004), multi-turn follow-up (712005), long buried message (712006), casual
chat (712000, 712010), and two instructions in one sentence (712003, 712015).

"hl plain/imp" is the official checker result for the highlight instruction. "strict" is prompt-level strict accuracy.

| key | test | latched (N) | expected N | hl plain/imp | strict plain/imp | note |
|---|---|---|---|---|---|---|
| 712000 | casual, "put like 3 key phrases in italics" | yes (3) | 3 | T/T | T/T | output changed, still passes |
| 712001 | bold via double asterisks, "items" | yes (4) | 4 | T/T | T/T | byte-identical (host complied) |
| 712002 | middle of message + irrelevant sentence | yes (2) | 2 | T/T | T/T | byte-identical |
| 712003 | no_comma + highlight in one sentence | **no** | 3 | F/F | F/F | missed: 'without' vetoes the sentence (no_comma fired and passed) |
| 712004 | system message only | yes (3) | 3 | T/T | T/T | byte-identical |
| 712005 | multi-turn follow-up | yes (2) | 2 | F/**T** | F/**T** | gain |
| 712006 | buried in long message, other numbers | yes (2) | 2 | T/T | T/T | byte-identical |
| 712007 | passive, "Three of the sentences..." | yes (3) | 3 | F/**T** | F/**T** | gain; appended spans copy earlier sentences |
| 712008 | no number (implicit 1) | yes (1) | 1 | F/**T** | F/**T** | gain |
| 712009 | "a few" + 150-word distractor | yes (1) | 2 (my reading of "a few") | F/F | F/F | N>20 cap stopped runaway; "a few" read as 1 |
| 712010 | ALL CAPS, odd punctuation | yes (2) | 2 | T/T | T/T | byte-identical |
| 712011 | numbered rule list | yes (**2**) | 3 | F/**T** | F/F | wrong N (list index); appended text broke the 120-word limit (T->F) |
| 712012 | only a '**bold**' example | yes (2) | 2 | T/T | T/T | byte-identical (section rule also latched, harmless) |
| 712013 | "Emphasise ... single stars" | yes (3) | 3 | F/**T** | F/**T** | gain |
| 712014 | "Don't forget to italicize" | yes (2) | 2 | F/**T** | F/**T** | gain; appended spans are meta text |
| 712015 | highlight + 'twice' in one sentence | **no** | 2 | T/T | T/T | missed (TIMES veto); safe, byte-identical |
| 712016 | NEG: "Can you explain why novelists italicize..." | **yes (1)** | - | - | - | false latch; output byte-identical |
| 712017 | NEG: someone else's habit | no | - | - | - | byte-identical |
| 712018 | NEG: "highlight the star players" | no | - | - | - | byte-identical |
| 712019 | NEG: "bold opening section" | no | - | - | - | byte-identical |
| 712020 | NEG: "Don't use any italics or bold" | no | - | - | - | byte-identical |
| 712021 | NEG: quoted instruction | no | - | - | - | byte-identical |
| 712022 | NEG: what italics mean (question) | no | - | - | - | byte-identical |
| 712023 | NEG: asked in an earlier turn, revoked now | no | - | - | - | byte-identical (turn scope works) |
| 712024 | NEG: "emphasize my ICU experience" | no | - | - | - | byte-identical |

## 4. Summary numbers

- **Trigger recall** (positives): 14/16 = 87.5%. Both misses (712003, 712015) are silent for this rule, so they fail safe.
- **Trigger precision**: 14/15 = 93.3% (1 false latch, 712016). On negatives, the latch false-fire rate is 1/9, and the
  output-change rate is **0/9** (all 9 negatives are byte-identical to plain).
- **Argument (N) correct when fired**: 12/14. Wrong: 712011 (list index). 712009 depends on how "a few" is read.
- **Checker, highlight instruction on positives**: plain 9/16, implant 14/16 (6 gains, 0 losses).
- **Prompt-level strict on positives**: plain 8/16, implant 13/16 (5 gains, 0 losses).
- **Instruction-level loss**: 1. In 712011 the appended spans broke the word limit. The highlight changed F->T, so the prompt
  stayed F.
- **Fail-safe check**: every silent item that had no other rule (712015 and all negatives) is byte-identical to plain.

## 5. Verdict

**PASS (conditional).**
- There is no reward hacking, no IFEval vocabulary, no leakage and no Python logic at inference.
- Detection and enforcement are correct on most positives (14/16 fire, 12/14 with the correct N, highlight compliance
  9/16 -> 14/16), including all four out-of-style settings.
- The misses fail safe.
- The output false-fire rate on negatives is 0/9 (1/9 at the latch level).

Conditions. The rule should be kept and narrowed, not dropped:
1. Round 2 with unseen variants is necessary, because part 5 was tuned after these items were run for detection. Round 2 must
   include "highlight/emphasize/bold + a number, non-markdown sense" negatives (weakness 4).
2. Move the negation and TIMES vetoes from sentence scope to clause scope (weaknesses 1 and 2).
3. Read the count next to the item word, as the bullets/paragraphs rules do (weakness 3).
4. Optional: apply the explain/describe guard to the ITALW-only path (weakness 5). Prefer content-bearing appended spans over
   meta text (weakness 6).
