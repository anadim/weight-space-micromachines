# JUDGE_bullets_2: rule-keeper "exactly N markdown bullet points" (round 2, final)

Judge: independent reviewer. Tool: `bullets` -> IFEval `detectable_format:number_bullet_lists {num_bullets}`.
Round-2 variants: `judge/bullets_variants_r2.jsonl` (keys 713100-713119, 12 positive, 8 negative). Results: `judge/bullets_results_r2.md`.
Code reviewed: the CURRENT `circuit.py` / `tables.py` (NOTES change logs part 6 + part 7). Round 1: `JUDGE_bullets_1.md`.

**FINAL VERDICT: PASS (conditional; keep the rule, no narrowing required).**
- No reward hacking, IFEval strings or Python logic.
- 0/8 false fires on new negatives (0/16 over both rounds).
- Every miss fails safe (byte-identical to plain).
- 2 gained, 0 lost on positives.

The conditions below concern output quality, not safety.

## Caveat: how held-out these items are
The builder made the part-7 changes after my round-2 code review. They were also partly shaped by running DETECTION on all r2
items, mine included. Part 7 fixes exactly the false-fire class I predicted for 713112, 713113, 713114 and 713117 (a noun, name or
adverb at the opening counted as an "imperative"). Those four negatives are therefore "seen". The other four negatives and all
generation-side results are not affected by that change.

## 1. Code review (current code)

**Detector** (circuit.py, the counts block and `okc`/`okb`):
- A number 1..20 comes at most 3 tokens before 'bullets/bulleted', or before 'bullet' + point/list/item.
- The number is not after a determiner or possessive, and no unit word comes between the number and the cue.
- The count is committed at the sentence end only if all of these hold:
  - the sentence is in a user or system turn;
  - it is not an information question (`nonreq_q`);
  - it has no third-person subject or verb (`rqs`);
  - it has a sentence-level request marker (`reqs`): an instruction verb from INSTV at a clause opening, you/please/must/..., or
    a negation that opens the sentence;
  - it is not a topic, reported or code sentence (part 7);
  - no negation shares a clause with the bullet cue (`_cue_neg`).
- Text inside single or double quotes, backticks, or a data span introduced by a colon is masked.
- Turn scope: user-turn latches are dropped at the next user turn. System latches persist until a cancel cue.

**Enforcement:**
- When the host starts a line and fewer than N bullets exist, the circuit emits a '*' line start (part 6).
- Once N bullets exist, bullet starts are blocked.
- Empty bullet lines are blocked.
- The end-of-turn appender (c3) still exists as a fallback.

**Reward hacking / test fitting: none found.**
- All new lists are generic English: INSTV, REQMARK, REQANY, SUBJ, EXPLAINV, REPORTV, CODEW, DATAV, DATAW, CANCELW.
- No IFEval sentence, template or entity is used.

**Python logic: none.**
- `_cue_neg` holds a pair of clause/sentence registers per rule. Its dict is set up once per batch, not per data value.
- The line-start conversion is a gated `start_prog`.
- The only readout is the disclosed LM-head top-token readout.

**Remaining weaknesses (generic; none is unsafe):**
1. **The answer's first line is converted too** (`at_ls` = 1 at answer start).
   - An intro line becomes bullet 1, and the host's last real item then loses its bullet.
   - Confirmed in 713108. The host's correct 3-item numbered list became: "*Fixing a squeaky door hinge can be done by:*", then
     "*1. ...*" and "*2. ...*", then "3. ..." with no bullet.
   - The checker passes, but the user gets 2 real bullets plus a bulleted intro.
   - Fix: do not convert a line that ends in ':' or that comes before the first list-like line. Or count a numbered '1.' line start
     as the place to substitute.
2. **The bare '*' marker is read as italics by the host.**
   - The forced token is '*' with no following space, so the host often treats it as an opening italic and closes it:
     "*1. Easy to ride and maintain.*" (713106, 713108).
   - The checker counts these lines, but the markdown renders as italic text, not as a bullet list.
   - Fix: force '* ' or '- ' as one token or as two forced tokens.
3. **Section headers count as bullets.**
   - In 713106 the host's "- **Pros:**" header line counted as bullet 1. Five forced pros followed, then "**Cons:**" and a numbered
     list that is not bulleted.
   - The checker passes, but the bullet count includes the header.
   - This is inherent to the checker's own definition. Still, it shows that line-start conversion cannot structure a two-part
     answer.
4. **Recall gaps remain.** All were predicted, and all fail safe:
   - a third-person pronoun anywhere in the sentence ("..., since they love museums", 713104);
   - 'N-bullet <noun>' ("3-bullet recap", 713105);
   - the header form "In 5 bullet points:\n..." (713107);
   - N > 20 (713109).
5. The c3 end-of-turn appender (the round-1 filler source) is still present. It was not triggered in round 2, because line-start
   conversion reached N first in every written case. Gating c3 to fire only when the answer has fewer than N non-empty lines would
   remove the filler path entirely.

## 2. Rule 13
f(x) = the reply contains exactly N list items in markdown bullet syntax. Extra bullet starts are blocked, and missing bullets are
created at line starts.

Three settings outside the benchmark:
- executive or slide summaries with N takeaways;
- UI cards or notifications that render exactly N items;
- changelog, release-note or stand-up templates with a fixed slot count.

**USEFUL-GENERALLY: yes.**

## 3. Round-2 variants and results

Columns:
- det = latched + N.
- plain / implant = strict check of the bullets instruction (713110 also checks title).
- ident = implant output byte-identical to plain.

| key | type | what it tests | det (N) | plain | implant | ident |
|---|---|---|---|---|---|---|
| 713100 | POS | system message "Format every answer as exactly 3 bullet points." | yes (3, system latch) | pass | pass | yes |
| 713101 | POS | multi-turn "Make it 2 bullet points instead." | yes (2) | pass | pass | yes |
| 713102 | POS | "..., without any introduction or conclusion" (negation in another clause) | yes (4) | pass | pass | yes |
| 713103 | POS | "in 3 bullet points and no emojis" | yes (3) | pass | pass | yes |
| 713104 | POS | "..., since they love museums" (third person elsewhere in the sentence) | no (expected) | pass | pass | yes |
| 713105 | POS | "3-bullet recap" | no (expected) | pass | pass | yes |
| 713106 | POS | "Could you sum up ... in 6 bullet points?" | yes (6) | FAIL | pass* | no |
| 713107 | POS | "In 5 bullet points:\n<question>" | no (expected) | pass | pass | yes |
| 713108 | POS | "pls do 3 bullets on ..." (casual) | yes (3) | FAIL | pass* | no |
| 713109 | POS | N = 25 (above the cap) | no (expected) | FAIL | FAIL | yes |
| 713110 | POS | title + 5 bullets in one sentence | yes (title, 5) | pass | pass | yes |
| 713111 | POS | numbered-list topic, "as exactly 6 bullet points" | yes (6) | pass | pass | yes |
| 713112 | NEG | "Slides with 3 bullet points are easier ..." | silent | - | - | yes |
| 713113 | NEG | "Could you explain why presenters stick to 3 bullet points ...?" | silent | - | - | yes |
| 713114 | NEG | "Honestly, 5 bullet points felt like too many ..., so tell me ..." | silent | - | - | yes |
| 713115 | NEG | instruction inside single quotes (translate) | silent | - | - | yes |
| 713116 | NEG | system says 3 bullets, user says "plain text this time" | silent (cancelled) | - | - | yes |
| 713117 | NEG | "Yesterday my team agreed ... gets 2 bullet points ..." | silent | - | - | yes |
| 713118 | NEG | "How do I make 2 bullet points line up ...?" | silent | - | - | yes |
| 713119 | NEG | "Do you think 4 bullet points are enough ...?" | silent | - | - | yes |

\* passes the checker, with the quality issues in weaknesses 1-3: 713108 has an intro turned into a bullet and its 3rd item left
unbulleted; 713106 has a header counted as a bullet and italic-wrapped items.

**Trigger statistics (round 2):**
- Recall on positives: 8/12 = 67%. The 4 misses are the 4 expected gaps, and all 4 are byte-identical to plain.
- Arguments: all 8 detections have the correct N.
- Precision: 8/8 = 100%. False fires on negatives: 0/8.
- Score on positives (checker): plain 9/12 -> implant 11/12. 2 gained, 0 lost.

**Combined over both rounds:**
- Recall 14/24 = 58%.
- 0/16 false fires.
- 4 gained, 0 lost.
- Every silent or unwritten item is byte-identical to plain.

## 4. Verdict
**PASS.** It meets every criterion in the brief:
- no reward hacking, test-fitted vocabulary or Python logic;
- each positive is either enforced correctly by the checker or fails safe, with 0 regressions;
- false fires are 0 on 16 negatives across two rounds. These include descriptions, information questions, quoted and
  single-quoted text, other word senses (round 1), negation, turn carry-over and system-message cancel.

Decision: **keep the rule.** No narrowing or dropping is needed for safety.

Recommended quality fixes (not required for PASS):
- (1) Skip converting an intro line (a line ending in ':' or the first line before any list-like line).
- (2) Force '* ' or '- ' instead of a bare '*', to avoid italics.
- (3) Gate the c3 appender so that it cannot produce filler.

Report the gained items as checker passes. In round 2 the content was the host's own text in both cases, which is better than the
round-1 filler, but the formatting is imperfect.
