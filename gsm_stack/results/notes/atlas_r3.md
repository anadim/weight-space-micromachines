# atlas-r3: remaining failures of the rung-3 stack (avf), error-led discovery (task [2].4)

Lane: implants/atlas_r3 (scripts, log, data copies). All CPU, offline, on existing rows. No new rollouts.
- Rung 3 (avf): dev 90.29, dev2 87.33. Seeds 7/11/13/17, up to 24 attempts per row.
- trainC rung-3 rows: 89.14, seeds 23/29/31/37, from implants/vfrefit/aw/out/tc_trainC.
- Rung-2 r2f rows are used as a reference.

Row accounting: 1 pt is 24 rows on dev and 12 rows on dev2 (600 or 300 items × 4 seeds). trainC has 1499 items × 4 seeds.
Item order matches the S3 atlas, so the never.md indices are valid.

## 1. Where the failures are

| bucket | dev rows (pts) | dev2 rows (pts) | trainC rows (pts) | r2f dev / dev2 pts |
|---|---|---|---|---|
| failures total | 233 (9.71) | 152 (12.67) | 651 (10.86) | 10.42 / 13.08 |
| **selection gap**: a right attempt is in a kept slot, but the writer picked wrong | 108 (4.50) | 91 (7.58) | 386 (6.44) | 4.96 / 7.83 |
| · right group outnumbered (right count < winner count, right count ≥ 2) | 50 (2.08) | 47 (3.92) | 177 (2.95) | 2.75 / 5.42 |
| · right group is a singleton | 34 (1.42) | 18 (1.50) | 119 (1.98) | 1.83 / 2.17 |
| · right group larger, lost on VF features | 20 (0.83) | 22 (1.83) | 69 (1.15) | 0.08 / 0.17 |
| · tied | 4 (0.17) | 4 (0.33) | 21 (0.35) | 0.29 / 0.08 |
| **generation miss**: no right attempt in the row | 125 (5.21) | 61 (5.08) | 265 (4.42) | 5.46 / 5.25 |
| · of which on gold-label-error items | 16 (0.67) | 7 (0.58) | – | 0.67 / 0.67 |
| · of which on S3 gold-ambiguous items | 8 (0.33) | 13 (1.08) | – | |
| right answer rejected by the sanity filter, or missing from the register | 0 | 0 | 0 | 0 |
| oracle@slots | 94.79 | 94.92 | 95.58 | 94.54 / 94.75 |

**Items** (vote over 4 seeds):

| set | always right | mixed | never right | never by attempts (no right attempt in any slot of any seed) |
|---|---|---|---|---|
| dev | 524 | 35 | 41 | 16 |
| dev2 | 251 | 23 | 26 | 8 |
| trainC | 1269 | 121 | 109 | 34 |

**Gold-label errors.** The never.md items (dev 6, 303, 488, 519; dev2 115, 255) are all 0/4 by vote.
- They cost 16 dev rows and 8 dev2 rows (0.67 pts per set).
- dev2 115 has 1 right attempt in 60, which shows up as one singleton-gap row.
- Adding the gold-ambiguous items gives an unfixable floor of ≈ 1.0 pts (dev) and ≈ 1.75 pts (dev2):
  - dev 564/567: cents vs dollars.
  - dev2 51/55/263/298.

**Rung 3 vs rung 2.** VF and Nmax 24 turned about 0.25 pts of misses into gaps and fixed gap rows. They also created a new sub-bucket: the right group is larger but VF scores it lower (dev 20, dev2 22).
- The VF still nets positive against the base voter: pick0 on the same rows gives trainC −69, dev −14, dev2 −3 rows.
- Gap rows where pick0 was right and VF broke it: dev 17, dev2 15, trainC 55.

**Seed variance.** Gap rows on items where 2–3 seeds vote right: dev 26, dev2 21 (≈ 1.1 / 1.75 pts). These are coin-flip items.

**Budget.** 65 of 108 dev gap rows and 63 of 91 dev2 gap rows already used all 24 attempts. Rule A extends while the winner share is < 0.7, so these rows ran out of budget with the right group still in the minority.

## 2. Mechanical / stack-caused checks (all small)

From mech.py; the output is in implants/atlas_r3/mech.md.

| check | dev | dev2 | trainC |
|---|---|---|---|
| S stop at 4 (4 unanimous wrong answers, σ(lg) ≥ 0.5) on miss rows | 31 rows; 25 on items with pooled per-attempt p = 0; only dev:116 has p = .66 | 20 rows; p > .1 only for 2 items (.20, .35) | 70 rows; 46 with p = 0 |
| all S-stopped rows / wrong | 1462 / 31 | 712 / 20 | 3706 / 70 |
| expected miss rows with ≥ 1 right in 8 extra tries (oracle, not accuracy) | 21 (≤ 0.89 pts) | 9 (≤ 0.73) | 52 (≤ 0.87) |
| miss rows by item p: 0 / < 5% / 5–15% / ≥ 15% | 64 / 35 / 20 / 6 | 32 / 19 / 8 / 2 | 136 / 60 / 47 / 22 |
| failing rows with any cap-truncated attempt (no answer at 400 tokens) | 23 | 8 | 23 |
| no-answer attempts / with gold somewhere in the body | 138 / 28 (mostly coincidental: gold appears in the question, e.g. dev:106) | 52 / 4 | 207 / 15 |
| failing rows with a repairs-exhausted attempt (B: 3 repairs on a non-integer answer) | 14 | 21 | 33 |
| register parse failures (all attempts / failing rows) | 3 / 1 | 4 / 4 rows on dev2:263 ("#### 146" for 14666.67, gold-ambiguous) | 9 / 4 |
| failing rows whose picked answer is non-integer | 5 (dev:567 ×4, cents; dev:5 ×1) | 0 | 5 |
| #### value ≠ last "= v" in the body / last "= v" is gold | 489 / 23 attempts | 298 / 19 | 1321 / 42 |
| failing rows with such an attempt (oracle) | 7 (0.29) | 2 (0.17) | 11 (0.18) |
| failing rows with a non-integer attempt value x (final or pre-repair) where ceil, floor or round of x = gold (loose bound) | 21 (ceil 15, floor 10) | 14 (ceil 11, floor 8) | 30 (ceil 16, floor 20) |
| picked wrong value v where 2v or v/2 = gold / v ± 1 = gold | 25 / 10 | 12 / 4 | 63 / 47 |

Notes on these checks:
- **Sanity filter.** No dev, dev2 or trainC item has a non-integer gold. The only negative gold is dev2:176, which is 4/4 right. The sanity filter never rejects a right answer.
- **S stop and caps.** The S stop and the 400-token cap are not real failure causes.
  - Capped attempts on the "cap" items (dev 64, 170, 297) never contain the gold. They are failed plans, for example algebra that drifts at 4.5X − 28.4 = 130.
  - Raising the cap or Nmax only adds oracle, at most ≈ 0.9 pts. That is an upper bound before the vote, and those right attempts would be singletons.
- **"####" copy errors.** Mostly these are not copy errors. The model computes the gold, then re-interprets: dev/131 (404 is the friend's total), dev2/33 (the white lights "is 0"), trainC/400 (the meeting point). A true digit-copy error happened once: dev:374, "$25200000" written as "#### 252000000". An answer-copy guard is worth ≤ 0.05 pts.
- **B-repair drift.** Example dev2:5 (gold 9): the answer goes 8.75 → 6.5 → 9.75 → 2.75 over the repairs, and the repairs never make the model reconsider the plan. Attempts with 3 repairs are 13% accurate. The "round" items need ceil, not a re-solve: dev2:159 needs ceil(137.33) = 138, and the model writes 137.333 three times.

## 3. Hand-read generation misses (all 49 dev and 23 dev2 items with a miss row)

Labels are in implants/atlas_r3/data/hand_miss.tsv (hand.py). Of these items, 41/49 (dev) and 19/23 (dev2) are the same items as the S3 never.md list: the stack has not changed which items are hard.

| category | dev items | dev miss rows (pts) | dev gap rows on the same items | dev2 items | dev2 miss rows (pts) | dev2 gap rows |
|---|---|---|---|---|---|---|
| gold_err | 4 | 16 (0.67) | 0 | 2 | 7 (0.58) | 1 |
| gold_ambig | 2 | 8 (0.33) | 0 | 4 | 13 (1.08) | 2 |
| world / unit fact / convention | 9 | 31 (1.29) | 5 | 1 | 2 (0.17) | 1 |
| dropped quantity / phase (often a verbal one) | 8 | 20 (0.83) | 10 | 5 | 17 (1.42) | 3 |
| misread relation / reference | 14 | 25 (1.04) | 26 | 7 | 15 (1.25) | 12 |
| plan (multi-step / algebra; 5 dev items are cap-heavy) | 11 | 24 (1.00) | 10 | 2 | 2 (0.17) | 6 |
| target | 1 | 1 (0.04) | 2 | 0 | 0 | 0 |
| round (non-integer, needs ceil or % format; B repairs exhausted) | 0 | 0 | 0 | 2 | 5 (0.42) | 2 |
| total | 49 | 125 (5.21) | 53 | 23 | 61 (5.08) | 27 |

Each line gives the question gist → the model's dominant wrong reading.

**world**
- dev:29: horseshoes need 4 per horse.
- dev:432: a pair = 2.
- dev:496: 4 tires per car.
- dev:136: a couples room holds 2 people.
- dev:374: 365 days/year.
- dev:533: BC calendar.
- dev:191: birth year + age.
- dev:530: $10 per *whole* $100 (floor).
- dev:599: peak-hour window.
- dev2:284: rate period.

**dropped**
- dev:120: rent shared.
- dev:386: splits with a friend.
- dev:107: ×20 birds.
- dev:121: the worm keeps eating in week 3.
- dev:184: the spreading phase (20 acres / 4 per day).
- dev:236: "sleeps 8 h" is mentioned but not subtracted.
- dev:129: drainage.
- dev2:234: the first hose keeps running.
- dev2:262: morning + evening.
- dev2:296: ×3 bags.

**misread**
- Increment vs replacement (double count):
  - dev:1: 5×30 + 35 + 55.
  - dev:31: 40×5 + 56.
  - dev2:291: "most … but one": 2×250 + 1000.
- Direction or rate inversion:
  - dev2:137: "5× as fast", so the slower reader takes 20/5.
  - dev:423 and dev2:195: one way vs round trip.
- Fencepost: dev2:70, "4 pens in the first week … after a month", written as 4·2⁴.
- dev:36: "sum and product … are 131" becomes 10 + A = 131.
- dev:5: "space for 4 more" becomes 12 + 4.
- Other misreads: dev:215, 445, 283, 349, 360, 522, 272, 217, 205, and dev2:212, 294, 69.

**plan**
- dev:424: maximize balloons for $24, but the model buys every bag size with the full $24 each.
- dev:116: fractions of the remainder.
- Cap-heavy: dev:64, 297, 106, 170, 213.
- dev:88 and dev2:84 share a template.

**round**
- dev2:159: needs ceil(137.33).
- dev2:109: answers 0.14 for 6%.

The diverge_aw heuristic (all wrong attempts on mixed items) is in implants/atlas_r3/data/r3/diverge.md and matches S3. The share of wrong attempts by category is relation 47–54%, dropped 14–22%, operand 11–17%, target 4–7%. Top subcategories: relation/wrong_ref, dropped/missing_operand, relation/op_or_order, operand/wrong_q_number, target/missing_last_scale.

Signature of the misreads: in 10–11% of failing rows the picked value is exactly 2× or ½ of the gold (dev 25, dev2 12, trainC 63). In 4–7% it is gold ± 1 (dev 10, dev2 4, trainC 47). These are per-person/both/round-trip/half misreadings and fencepost errors.

## 4. Selection gap: what separates the right group from the winner?

From gap.py; the output is in implants/atlas_r3/gap.md.

**Feature comparison.** On gap rows (trainC 386, dev 108, dev2 91), the right group compared with the VF winner:
- **Count is lower** in 71–78% of rows, and **Σσ(lg) is lower** in 77–83%.
- mean lg, min lg, nn, z_ntok and magnitude are coin flips (39–63%).
- qc, min qc, sanity, integerness, in-question value and "last = v consistent" are **equal in 82–100%** of rows. Sanity and integerness are fully exhausted by the keep filter, and qc is saturated: every group mentions every question digit.
- New text signals:
  - **cross-support** (the share of out-of-group attempts that write the group's value as an intermediate);
  - **intermediate agreement** (the group's own intermediates seen elsewhere);
  - **repairs used**.

  On contested rows that the VF wins, these point *against* the right answer: wrong groups have more cross-support (xs 24–26% vs 15–12% on dev/dev2), because a wrong answer is often an intermediate value of the right chain (missing last step). On gap rows they are ≈ 50/50.

**Paired re-picks vs the in-stream VF pick** (fixes/breaks; weights chosen on trainC):

| rule | trainC | dev | dev2 |
|---|---|---|---|
| base voter pick0 | −69 (+55/−124) | −14 (+17/−31) | −3 (+15/−18) |
| max count, ties → VF | −79 | −28 | −10 |
| VF, but pick0 if VF margin < 0.25 | −14 | −2 | +4 (+6/−2) |
| VF + w·xsn (w = −0.25) | +2 | −1 | +4 (+4/−0, p = .06) |
| VF + w·iag / rep / rep0 / leq / inq / mag / ntok / int | +3 … +4 / 0 … −3 | −2 … +1 | −1 … +2 |
| VF + 1·usew (operand-use proxy over the question's number words, see below) | +4 (+9/−5) | −1 | +4 (+5/−1, p = .11) |
| P-ridge refit BASE18 (control, no new features) | CV −8 … −16 | −1 … −2 | **+7 (+7/−0)** |
| P-ridge refit BASE18 + use + usew (l2 = 1e-2) | CV −1 (+25/−26) | +1 | +11 (+11/−0) |
| P-ridge refit BASE18 + xs + iag + rep + int + leq + inq | CV −16 … −8 | −3 … −1 | +8 / +9 |

Reading the table:
- No rule on existing or new text signals beats the VF on trainC in item-grouped CV.
- The dev2 +7…+9 comes from the refit itself (the BASE18-only control gives +7/−0), not from any new signal. This matches vfrefit, where ClusterLin refits were saturated.
- **Operand-use proxy.** This is a regex proxy for idea 1:
  - use = the share of the question's digit numbers that appear left of "=" in a body line;
  - usew = the same for number words (pair, twice, half, both, dozen, week, …).

  It is the only signal that stays non-negative everywhere. Relative to the refit control it adds about +7 rows in trainC CV, +2 on dev and +4 on dev2. That is weak and not significant, but the direction is consistent. On gap rows usew is higher for the right group in 12–16% of rows and lower in 4–11%.
- **Selection is generation-limited.** The right answer is a minority, and most gap rows (65 of 108, 63 of 91) already ran 24 attempts. Realistic gain from writer changes on the current registers is ≤ 0.1–0.2 pts.

## 5. Ranked fixable buckets and implant ideas

Ceilings are in pts per set (dev / dev2). "Ceiling" is the bucket size; "realistic" is my estimate.

| # | idea (weight-space, single call) | evidence | ceiling dev / dev2 | realistic |
|---|---|---|---|---|
| 1 | **Operand-use register**: a question-quantity *use* coverage, i.e. whether the quantity is an operand in an "a op b = c" line rather than just mentioned. It should cover number words (pair, twice, half, both, each, dozen, round trip, morning and evening). Use it as a VF group feature and as an A-extend trigger. | dropped bucket 0.83 / 1.42 miss + 10 / 3 gap rows. qc is equal across groups in 82–96% of gap rows (saturated by mentions; dev:236 mentions 8 h but never uses it). relation/dropped are the top diverge categories. The offline regex proxy (usew) is the only selection signal that is non-negative on all three sets (§4), though weak. | ≈ 1.2 / 1.7 | 0.2–0.5 (≤ 0.2 from selection alone) |
| 2 | **Relation-cue steering implant** (attention bias or steering vector on cue tokens): "each / per / both / round trip / as fast as / more on <day> / but one / after N weeks", aimed at the ×2, ½ and ±1 misreads | 2v or v/2 = gold in dev 25 / dev2 12 / trainC 63 failing rows; v ± 1 = gold in 10 / 4 / 47; misread miss 1.04 / 1.25 pts plus 26 / 12 gap rows on the same items | ≈ 2.1 / 2.3 (misread miss + gap) | 0.3–0.6; needs a generation change, and trainC has the same signature, so it can be trained there |
| 3 | **World-fact memory** (a key→count MLP edit: horseshoes/horse 4, pair 2, tires/car 4, days/year 365, couples room 2, BC years, "per whole $100" = floor) | world 1.29 pts dev, 0.17 pts dev2; 9 dev items, each 3–4/4 miss | 1.5 / 0.25 | 0.3 / 0; concentrated in dev, so overfit risk; check trainC coverage before building |
| 4 | **Ceil/round writer** for an exhausted B repair: when the 3rd repair is still non-integer and the question asks "how many … need / buy / trips / boxes", commit ceil(x) as the slot value instead of a fresh re-solve | dev2:159 (4/4 miss) and dev2:109; the loose bound is 21 / 14 / 30 failing rows with some x whose ceil, floor or round is gold, but the direction is ambiguous (trainC floor 20 > ceil 16); 3-repair attempts are 13% accurate | 0.9 / 1.2 (loose) | ≤ 0.1 / 0.4 |
| 5 | **Plan / algebra cap** (raise the 400 cap for algebra-style attempts, or seed "let X" plans) | dev 64 / 297 / 106 / 170 / 213: 12 miss rows; the gold never appears in the truncated bodies | 0.5 / 0.1 | ≈ 0; these are plan failures, not truncation |
| 6 | **Writer re-pick on existing registers** (margin fallback, cross-support, repairs, integerness, …) | §4: no CV-positive rule on trainC; the dev2 gain is a refit artifact | 4.5 / 7.6 (gap bucket) | ≤ 0.1–0.2 |
| 7 | **Mechanical guards**: an answer-copy guard (#### vs last "= v"), S-stop guard, register parse | 1 true copy error (dev:374); S-stop misses are on p = 0 items except 3 rows; parse failures are on a gold-ambiguous item | 0.3 / 0.2 (oracle) | ≤ 0.05 each |
| – | Unfixable: gold-label errors plus gold-ambiguous items | never.md, S3 gold_ambig | 1.0 / 1.75 | 0 |

**Bottom line.** The rung-3 stack's remaining errors are ≈ ½ selection gap and ≈ ½ generation miss. Neither the mechanical bucket nor the register-based selection bucket has slack.

Every bucket with real headroom needs a *generation-side* change:
1. Use the verbal and implicit quantities: ideas 1 and 2.
2. Know a small set of world counts: idea 3.

The most testable next implant is an operand-use register (idea 1). It is computable from the numreg/qc automaton by marking numbers that appear left of "=" (or as an operand in "a op b"). It also extends to number words. It can be tried offline first. The regex proxy in gap.py is already weakly non-negative as a VF feature. The bigger lever is using it as an A-extend / re-attempt trigger, which needs rollouts.
