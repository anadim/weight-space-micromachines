# JUDGE_bullets_1: rule-keeper "exactly N markdown bullet points"

Judge: independent reviewer (round 1). Tool: `bullets` -> IFEval `detectable_format:number_bullet_lists {num_bullets}`.
Variants: `judge/bullets_variants.jsonl` (keys 713000-713019, 12 positive, 8 negative). Results: `judge/bullets_results_r1.md`.
Code reviewed: the CURRENT `circuit.py` / `tables.py` (after change logs part 4 and part 5).

**VERDICT: PASS (with conditions).** There is no reward hacking and no Python logic. The rule fires 0 times on 8 negatives. Every
missed positive fails safe (byte-identical to plain). But the rule detects only 6/12 positives. Also, BOTH outputs that the circuit
actually changed pass the checker with low-quality filler bullets. See "Conditions" below.

## Important caveat about the test set
The builder changed the circuit after my preliminary review and BEFORE running my items. Part 4 was driven by the code reviews
only. Part 5 was also driven by running DETECTION on all judges' variants, mine included. My multi-turn negative (713019) is exactly
the case the part-5 turn-scope change fixes. My items are therefore not fully held out. In particular, the 0/8 false-fire result is
partly "seen". A round 2 with fresh variants is needed before this result is final.

## 1. Code review (current code, bullets path)

Where the rule lives:
- Detector: circuit.py about L262-279.
- Commit and request guards: `okc`, `rqs`, `negs`.
- Turn scope: about L445-458.
- Gate: L566.
- Blocked set: L622-628.
- End-of-turn program c3: L656 / program 8.
- Answer counter: `bcount`, `at_ls`, `ln_bul`, `ln_cont`.
- Tables: `BULLETS`, `BULLET1`, `POINTW`, `UNITW`, `POSSW`, `NEG`, `OTHERS`, `DESCV`, `BULST`.

**Reward hacking / test fitting: none found.**
- The cue words are generic English: bullet(s), bulleted, bulletpoint(s), and point/list/item after 'bullet'.
- UNITW, POSSW and CANCELW are generic linguistic lists.
- The number reader is generic: digits and number words up to twenty, with N limited to 1..20.
- The answer counter counts line-initial '*' (not '**') or '-'. That is markdown syntax, and it matches the official checker.
- No IFEval sentence, template or entity appears in the tables.
- The `drv:1634` and `drv:2832` dev templates are allowed rewordings. The detector does not depend on their exact wording.

**Python logic at inference: none.**
- Everything is torch.where / gather / 0-1 masks / integer registers.
- The `for k in self.L` loops are static loops over rule names, not branches on data.
- Regexes in tables.py run offline to build fixed tables. That is allowed.
- The LM-head readout (host top token, and the masked hard-max that gives the best token outside the blocked set) is disclosed.

**Leakage: none seen.** The dev data is our own templates plus allowed rewordings.

**Weaknesses (generic, not test fitting):**
1. The negation and third-person guards are crude and apply to the whole sentence.
   - Any NEG word anywhere in the sentence vetoes the count. Examples: "...exactly 4 bullet points and nothing else" (713000) and
     "4 bullets, no intro, no outro!!" (713009).
   - Any he/she/they anywhere in the sentence also vetoes it. Example: "...needs exactly 2 bullet points she can paste straight
     in" (713006).
   - These misses fail safe, but they cost recall on very common real phrasings. In particular, the system-message path
     (persistent latches) never got exercised because of "nothing else".
   - Better: let the negation count only when it comes before the count and takes it as its object (the case rules got this fix).
2. "5-bullet summary" / "a 3-bullet recap" are not covered (713007). `BULLET1` needs point/list/item right after it. Adding a
   general head noun (summary, recap, overview, answer, list) would be generic.
3. **Enforcement is append-only, and that is the main quality problem.** The circuit never makes the host START a list.
   - When the host writes prose or a numbered list ("1. **...**"), the only action is at end of turn: force a '*' line start,
     then let the host fill it.
   - In both items where the circuit wrote (713002, 713004), the result passes the checker, but the added bullets are filler:
     - 713004: correct lowercase prose, then `*bullet points*` three times.
     - 713002: the host's five-item numbered list, then five lines of italic "*P.P.P.P.P.S. ...*" text.
   - This is not test fitting. It is the disclosed design. But it satisfies the checker rather than the user. So the score gain
     (+2 items) overstates how useful the tool is in real use.
   - Generic fixes:
     - when the bullets gate is on and the answer has not started a list, force a bullet start at the answer start;
     - force a bullet start after each line break until N bullets are reached;
     - treat a host's numbered item ("1.") at line start as a place to substitute a bullet marker.
4. Minor: if an indent is a separate whitespace token, the bullet that follows it is not counted or blocked (`at_ls` is 0 after
   the indent token). The checker counts indented sub-bullets. This is rare at this host size.
5. Disclosed design limit: an earlier user-turn count is dropped at the next user turn, so "same format as before" fails safe.
   This is acceptable.

## 2. Rule 13: is f(x) generally useful?

f(x) = make the reply contain exactly N list items in markdown bullet syntax. Extra bullet lines are blocked, and missing ones are
added at the end of the turn.

Three settings outside the benchmark:
- executive or slide summaries with a fixed number of points ("give me 3 takeaways");
- UI cards, push notifications or widgets that render exactly N items;
- release notes, changelog or stand-up templates with a fixed slot count.

**USEFUL-GENERALLY: yes.** The function is general. The interface is a generic cue ("N bullets / N bullet points / an N-item
bulleted list"), not IFEval wording. The weak parts are recall and enforcement quality, not generality.

## 3. Variants and results (round 1)

Columns:
- detected = latched rule + N.
- plain / implant = strict check of the bullets instruction (for 713004, also lowercase; for 713008, also no_comma).
- identical = implant output byte-identical to plain.

| key | type | what it tests | detected (N) | plain | implant | identical |
|---|---|---|---|---|---|---|
| 713000 | POS | instruction only in a system message | no ("nothing else" = negation) | pass | pass | yes |
| 713001 | POS | multi-turn follow-up "boil that down to 3 bullet points" | yes (3) | pass | pass | yes (no write needed) |
| 713002 | POS | casual "like 5 bullets ... lol" | yes (5) | FAIL | pass* | no (filler bullets) |
| 713003 | POS | mid-message "six bullet points" + irrelevant sentence | yes (6) | pass | pass | yes |
| 713004 | POS | bullets + lowercase in one sentence | yes (3) + lower | FAIL | pass* | no (filler bullets) |
| 713005 | POS | "four-item bulleted list" | yes (4) | pass | pass | yes |
| 713006 | POS | buried in a long message, numbers elsewhere | no ("she" in sentence) | pass | pass | yes |
| 713007 | POS | "5-bullet summary" | no (no point/list/item) | FAIL | FAIL | yes |
| 713008 | POS | no-comma + "3 short bullet points" | yes (3) + nc | pass | pass | yes |
| 713009 | POS | "4 bullets, no intro, no outro!!" | no ("no" = negation) | FAIL | FAIL | yes |
| 713010 | POS | number after the cue ("give me four of them") | no (expected) | pass | pass | yes |
| 713011 | POS | "lines starting with a dash", no cue word | no (expected) | pass | pass | yes |
| 713012 | NEG | boss's habit, opinion question | silent | - | - | yes |
| 713013 | NEG | what does the "3 bullet points rule" mean | silent | - | - | yes |
| 713014 | NEG | instruction inside a quoted rubric | silent | - | - | yes |
| 713015 | NEG | "1964 bullet train" | silent | - | - | yes |
| 713016 | NEG | "12 bullet casings" in a passage to rewrite | silent | - | - | yes |
| 713017 | NEG | "2 bullet journals" | silent | - | - | yes |
| 713018 | NEG | "Please don't give me 4 bullet points again" | silent | - | - | yes |
| 713019 | NEG | earlier turn asked for 3 bullets, new turn asks for prose | silent (dropped at the new turn) | - | - | yes |

\* passes the checker with appended filler bullets (see weakness 3).

**Trigger statistics:**

| measure | result |
|---|---|
| recall on positives | 6/12 = 50% (6/10 = 60% without the two items built to have no cue) |
| arguments | all 6 detections have the correct N |
| precision | 6/6 = 100% |
| false fires on negatives | 0/8 |
| fail-safe | every non-detected item, and every item that needed no change, is byte-identical to plain |

**Score on positives (checker, prompt-strict): plain 8/12 -> implant 10/12.** 2 items gained, 0 lost, and 0 negatives changed.

What changed since my preliminary review: in my code-reading prediction, 7/8 negatives fired with an unbounded N (e.g. 1964). The
part-4 and part-5 changes fixed all of these generically:
- N limited to 1..20;
- 'bullet' must head point/list/item;
- request, negation and third-person guards;
- turn scope.

## 4. Verdict

**PASS** against the brief's criteria:
- no reward hacking, IFEval strings or Python logic;
- false fires near zero (0/8);
- positives are either handled correctly or fail safe (12/12 checker-correct or byte-identical; 0 regressions).

**Conditions and recommendations** (narrow or improve; do not drop):
- (a) Re-judge in round 2 with fresh variants. Round-1 items informed part 5.
- (b) Fix the enforcement quality so that bullets are started at the answer start and at line starts, not appended as filler.
  Until then, report the "gained" items as checker passes with poor content.
- (c) Make the negation and third-person guards local to the count phrase, to recover recall on "N bullet points and nothing
  else", "no intro", and "she can paste".
- (d) Optionally cover "N-bullet <noun>".
