# Never / near-never items (lane 4, todo `never-solved`)

Scope: every dev600/dev2 item with n_right <= 4/32 under the current S3 stack at T=1. That is **76 items**: dev 50 (17 never + 33 near-never) and dev2 26 (6 + 20).
Method: each item was read by hand from atlas/never/dump.txt, which shows the question, the GSM8K train gold solution (matched by exact question text, 76/76 found), one right attempt, two attempts with the top wrong answer, and one other wrong attempt.
Files:
- labels: `atlas/never/labels.py` → `atlas/data/never_labels.json` (per item: gold, top-3 wrong answers, category, secondary tags, note, trigger stats);
- trigger fire rates: `atlas/never/triggers.py` → `atlas/never/triggers.json`.
`None` = no `####` in the attempt. Of the 206 None attempts across dev+dev2, about 75% stopped at the 320-token attempt cap (`acap`).

## Category counts

| category | total | never (0) | near (1-4) | dev | dev2 |
|---|---|---|---|---|---|
| dropped quantity / condition | 20 | 3 | 17 | 13 | 7 |
| question misreading | 17 | 3 | 14 | 11 | 6 |
| gold-label error / ambiguous | 13 | 9 | 4 | 6 | 7 |
| plan / decomposition | 13 | 4 | 9 | 10 | 3 |
| world knowledge / units / calendar | 11 | 4 | 7 | 9 | 2 |
| wrong target | 1 | 0 | 1 | 1 | 0 |
| arithmetic / rounding / format | 1 | 0 | 1 | 0 | 1 |

Notes on the counts:
- **Gold errors.** In 6 items the gold is plainly wrong and the model's dominant answer is literally correct: dev 6, 303, 488, 519 and dev2 115, 255. Of dev's 17 never items, 6 are gold/ambiguous, so the reachable never-ceiling is about 11/17.
- **Dropped quantities with no digit.** 9 dropped/misread items drop a *verbal* quantity with no digit in the question ("splits with his friend", "rent shared with Shelby", "two week", "three bags", "morning and evening", "sleeps eight hours", "another hose"). A digit-based check cannot see these.
- **Cap-truncated.** 9 items are cap-dominated (tag `cap`): dev 64, 106, 117, 170, 213, 297 and dev2 199, 263, 298. These are mostly algebra or long multi-part plans that run past 320 tokens.
- **Recurring families across categories:**
  - round trip / ×2: dev 64, 423; dev2 195, 262;
  - rate-period mix (per day + per week): dev 361; dev2 284;
  - calendar: dev 191, 533;
  - "infer per-box size from a total": dev 88; dev2 84, the same template.

## Per-item labels (sorted by set, n_right)

| item | n_right | gold | top wrong (count/32) | category | what goes wrong [secondary tags] |
|---|---|---|---|---|---|
| dev:5 | 0 | 3 | 1.5 (17), 0.666666 (6) | misread | "currently space for 4 more" read as has 12+4=16 (should be 12-4=8) |
| dev:6 | 0 | 96 | 576 (32) | gold_ambig | "each tile is 6 square inches" literally = area 6 -> 3456/6=576 is right; gold treats 6 in as tile side [gold_wrong] |
| dev:29 | 0 | 36 | 186 (12), 100 (10) | world_units | a horse needs 4 horseshoes (model uses 1 or 2 per horse / 2 kg per horse) |
| dev:64 | 0 | 118 | None (18), 15 (3) | plan | 5-part toll+gas weekly cost, both directions; 18/32 hit the 320-token attempt cap [cap] |
| dev:120 | 0 | 2000 | 1400 (24), 3200 (5) | dropped | rent shared with Shelby (1200/2) ignored: a verbal condition, no number [no_digit] |
| dev:170 | 0 | 360 | None (9), 210 (6) | plan | grazing months (144/18=8) then corn for 4 months, sheep 1 bag/2 months; 9/32 capped [cap] |
| dev:215 | 0 | 65 | 68 (23), 76 (3) | misread | "3 planks in each bedroom" = 2 bedrooms (bedroom + guest); model uses 3x3 |
| dev:297 | 0 | 320 | None (26), 280 (1) | plan | nested-fraction algebra (9/20 x - 14 = 130); 26/32 hit the attempt cap in LaTeX algebra [cap] |
| dev:303 | 0 | 90 | 180 (32) | gold_ambig | "shed 3/4 times as many leaves as it shed on the second week" = 180 shed -> 180 left; gold uses 3/4 of remaining [gold_wrong] |
| dev:374 | 0 | 25600000 | 252000000 (7), 28800000 (6) | world_units | days per year = 365 (model 30 days x 12 months); also #### digit-copy errors (252000000 vs 25200000) [format] |
| dev:386 | 0 | 3 | 6 (32) | dropped | "splits 8 shots with his friend" -> /2 ignored (no number) [no_digit] |
| dev:432 | 0 | 35 | 45 (31), 23 (1) | world_units | a pair = 2 individual items (socks, shoes; gold keeps pants as 1!) |
| dev:488 | 0 | 7 | 13 (32) | gold_ambig | Emily = Logan + 6 = 26, Kate = 13; gold computes 20-6 (gold wrong) [gold_wrong] |
| dev:496 | 0 | 84 | 32 (14), 80 (7) | world_units | a car has 4 tires (+1 spare), a motorcycle 2 |
| dev:519 | 0 | 4 | 10 (25), 5 (4) | gold_ambig | gold solution itself derives x=10; label 4 is wrong [gold_wrong] |
| dev:564 | 0 | 40 | 30 (23), 15 (8) | gold_ambig | "three times older than" read by gold as 4x as old; model 3x (common usage) |
| dev:567 | 0 | 350 | 3.50 (28), 26 (1) | gold_ambig | answer unit: gold in cents (350), model in dollars (3.50); question does not say |
| dev:117 | 1 | 100 | 150 (6), 225 (4) | plan | stations needed = max over car types (iron 12/3=4); model multiplies/sums [cap] |
| dev:121 | 1 | 26 | 40 (30), 0 (1) | dropped | worm keeps eating during the third week ("a week later": 7x2 more fish) [no_digit] |
| dev:190 | 1 | 4 | 15 (15), 6 (8) | dropped | Caleb's received 11 candies dropped in the final difference (uses stale 11, not 22) |
| dev:213 | 1 | 4 | None (14), 3 (8) | plan | study hours + per-hour/per-day breaks, ceil days; gold itself counts snack breaks once; 14/32 capped [cap,gold_questionable] |
| dev:217 | 1 | 2 | 0.4 (5), 5 (3) | misread | "each time a student eats vegetables ... 2 points" (points per vegetable) and "per school week" (10 days = 2 weeks) |
| dev:273 | 1 | 100 | 10 (17), 800 (13) | wrong_target | answers calories per piece (10) instead of per person (10 pieces x 10) |
| dev:349 | 1 | 300 | 200 (9), 100 (7) | misread | "if she hits 200 drums she can start to make money": paid only beyond 200 |
| dev:361 | 1 | 10 | 4 (18), 17 (5) | world_units | rate period mix: adds 1 lb/day to 3 lb/week (needs x7 days/week) [period] |
| dev:445 | 1 | 66 | 43 (28), 86 (1) | misread | "half of the birds in the cage": white birds are next to the cage, not in it |
| dev:522 | 1 | 28 | 47 (11), 27 (11) | misread | "5 players play 5 rounds in which each player removes one block" = 25, model 5 |
| dev:530 | 1 | 230 | 225 (19), 25 (7) | world_units | convention: "$10 for every $100" counts whole $100 (floor 2.5 -> 2) [floor] |
| dev:533 | 1 | 852 | 148 (23), 4348 (3) | world_units | BC calendar: "more recent" = smaller BC year; model sets first = second + 352 [calendar] |
| dev:184 | 2 | 25 | 20 (14), 2000 (12) | dropped | spreading phase (20 acres / 4 acres per day = 5 days) dropped; answers collection days 20 |
| dev:191 | 2 | 1990 | 1950 (16), 1890 (3) | world_units | calendar: younger brother born later = 1970 + 20; model 1970 - 20 [calendar] |
| dev:254 | 2 | 24 | 27 (14), 12 (6) | plan | needs algebra for unknown plate price (x+8 = 2x); model invents $10 |
| dev:347 | 2 | 35 | 47.5 (8), 37.5 (5) | dropped | "after the first 10 paid lessons only every other" condition dropped (19 x 5) [no_digit] |
| dev:360 | 2 | 60 | 600 (17), None (5) | misread | "10 boxes ... 240 pomelos in all" read as 240 per box |
| dev:423 | 2 | 42 | 21 (28), 980 (1) | misread | "one way" implies a round trip (x2); model drives one way [roundtrip] |
| dev:599 | 2 | 80 | 64 (11), 160 (6) | dropped | 5-6 pm is peak (12 cars/15 min); model uses off-peak for both hours |
| dev:14 | 3 | 480 | 120 (9), 240 (4) | dropped | "two week vacation" x 5 workdays dropped -> 5-6 = -1 days [no_digit] |
| dev:21 | 3 | 10 | 9 (14), 8 (6) | dropped | the first day (10 chips) is itself a day: 9 + 1 (off-by-one) |
| dev:88 | 3 | 2 | 6 (5), 1 (4) | plan | infer box size from total 85 ((85-5)/5); model invents 100 or 10 per box |
| dev:107 | 3 | 2200 | 110 (28), 220 (1) | dropped | 20 birds x 110 miles ("combined distance all of the birds") |
| dev:129 | 3 | 21 | 12 (13), 0 (4) | dropped | drainage 3 in/day x 3 days ignored (72-60=12) |
| dev:136 | 3 | 400 | 270 (28), 260 (1) | world_units | a couples room holds 2 people |
| dev:167 | 3 | 200 | 250 (23), 300 (6) | plan | double count: the 50 cut slices are no longer slices (100+150 vs 50+150) [double_count] |
| dev:175 | 3 | 88 | 48 (16), 96 (4) | plan | drive/cool cycles: 2 full 6-h cycles + 1 h; model stops at one cycle |
| dev:205 | 3 | 2 | 0 (13), 8 (13) | misread | "they ... divided among 4 weeks" pooled (both 10/wk -> 0); also monthly diff 8 (wrong target) [wrong_target] |
| dev:236 | 3 | 5 | 9 (14), 23 (5) | dropped | "sleeps eight hours" ignored (24-6)/2 = 9 [no_digit] |
| dev:272 | 3 | 61 | 60 (22), 94 (2) | misread | "beats last week" = strictly more -> 324-264+1 [strict] |
| dev:444 | 3 | 125 | 65 (8), 60 (6) | dropped | friends' $5 each dropped ("she and her friends"); n_att 27 |
| dev:36 | 4 | 11 | 121 (26), None (1) | misread | "total for the sum and product ... are 131": product dropped (10+A=131) |
| dev:106 | 4 | 10 | None (18), 5 (3) | plan | algebra on a 50/50 mix (5.5 good oranges per tree); 18/32 capped [cap] |
| dev2:51 | 0 | 41 | 860 (31), 677 (1) | gold_ambig | gold prices one day (41) despite "for 14 days" (natural 14x41=574); model 860 also wrong (4 of 126 pills) |
| dev2:84 | 0 | 2 | 70 (7), 5 (7) | plan | infer box size from 3 boxes = 45 chocolates; model assumes 1 per box (same template as dev 88) |
| dev2:115 | 0 | 560 | 840 (23), 280 (3) | gold_ambig | final leg "twice ... all other legs" = 560, total 840 (model); gold uses 280 (gold wrong) [gold_wrong] |
| dev2:212 | 0 | 80 | 120 (23), 150 (6) | misread | "triples the number of total trees" -> plants 60 (90-30); model plants 90 |
| dev2:255 | 0 | 40 | 80 (32) | gold_ambig | drivers 120 - transit 40 = 80 (model); gold subtracts from non-drivers 80 (gold wrong) [gold_wrong] |
| dev2:286 | 0 | 144 | 104 (31), 109 (1) | dropped | "sell and paint 8 bikes": painting $5 x 8 dropped (8 x 13) |
| dev2:39 | 1 | 8 | 2 (8), 4 (8) | misread | half-distance field -> twice the laps (4 x 2); model halves |
| dev2:55 | 1 | 4 | 0 (24), 3 (7) | gold_ambig | "using a gallon" read as one shared gallon -> 0 left; gold assumes a gallon each |
| dev2:109 | 1 | 6 | 0.14 (8), 0.06 (6) | misread | "willing to work nights" = 1 - 70% day-only; model uses 0.7 (0.14) ; also 0.06 vs 6 (percent format) [format] |
| dev2:159 | 1 | 138 | 137.333333 (23), 2060 (2) | arith_format | ceil: 137.33 hours -> 138 (model leaves 137.333333) [ceil] |
| dev2:195 | 1 | 8 | 4 (31) | misread | "to both appointments and back home again": each trip is from home -> 2 x (30+50) [roundtrip] |
| dev2:217 | 1 | 19 | 204 (7), 17 (5) | plan | break cycles: 17 min typing + one 2-min break; model treats 17 as 10-min intervals |
| dev2:234 | 1 | 390 | 290 (31) | dropped | "add another hose": first hose keeps running the last 2 h (120 x 2) [no_digit] |
| dev2:284 | 1 | 64 | 88 (19), 280 (4) | world_units | rate period: baseline 12 chores/week over two weeks = 24 (model subtracts 12) [period] |
| dev2:289 | 1 | 11 | 13 (29), 15 (2) | dropped | the 2 new fish are removed before the exchange (16-6-2+3) |
| dev2:294 | 1 | 62 | 69 (13), 60 (11) | misread | "three times as many hours catching up as working" direction (3x3 vs 3/3); 60 = weekend dropped |
| dev2:298 | 1 | 56 | None (13), 126 (4) | gold_ambig | sales volume unstated; gold assumes one portion each per day x 7; 13/32 capped introducing x, y [cap] |
| dev2:7 | 2 | 128 | 768 (12), 748 (7) | misread | "each one must contribute the same amount": Jules owes 200, model uses 1000 - 40 |
| dev2:189 | 2 | 42 | 36 (5), 17.6 (4) | world_units | rectangular room: two 12x10 walls + two 16x10 walls (model 4 x 120) |
| dev2:263 | 2 | 13600 | None (7), 10400 (5) | gold_ambig | "2/5 times fewer": gold reads Chris = 1.4 x Brian; model 3/5 C = 400 (non-integer); 7 capped [cap] |
| dev2:199 | 3 | 80 | None (8), 60 (3) | plan | algebra on a constant practice rate (current = 40 days); 8 capped [cap] |
| dev2:296 | 3 | 126 | 42 (17), 453 (7) | dropped | "three bags" x3 dropped: answers one bag (42) [no_digit] |
| dev2:16 | 4 | 72 | 42 (16), 78 (5) | dropped | nut-only cookies (15 x 2) dropped; counts only the mixed ones |
| dev2:46 | 4 | 960 | 1740 (5), 60 (3) | dropped | months fed (12, 16) not multiplied into feed cost (3 x 10) |
| dev2:262 | 4 | 365 | 182.5 (10), 26 (7) | dropped | "same amount of time in the morning and the evening" x2 dropped [roundtrip,no_digit] |
| dev2:274 | 4 | 110 | 146 (6), 70 (5) | gold_ambig | "sells all of the wigs for that play for $4": $4 each (gold) vs $4 total (model 146) |

## Trigger fire rates (all dev+dev2 attempts with a `####`: 23,144 right, 5,441 wrong; 900 questions)

| trigger (what an implant would key on) | fires on right attempts | fires on wrong attempts | precision for "wrong" |
|---|---|---|---|
| question digit unused before `####` | 731 (3.2%) | 490 (9.0%) | 40% |
| question digit or number word (two..twelve, twice, half, triple) unused | 1,987 (8.6%) | 861 (15.8%) | 30% |
| final answer non-integer | **0 (0.0%)** | 408 (7.5%) | **100%** (0/900 golds non-integer) |
| "how many X": X absent from last line before `####` | 3,989 (20.9%) | 1,140 (26.8%) | 22% |
| round-trip phrase in q and no ×2/"back" in attempt | 32/96 (33%) | 80/110 (73%) | 71% (7 questions only) |

Question-side lexicon. Columns: number of questions containing the trigger; per-attempt accuracy on those questions; number that are always right (32/32); slice items among them.
- **Rare, high-value words:**
  - horseshoe: 1 q, 0% (dev 29)
  - tire(s): 2 q, 45% (dev 496)
  - couple(s): 4 q, 63%, 1 always (dev 136)
  - BC|born: 7 q, 52%, 0 always (dev 191, 533)
  - "for every $": 1 q, 3% (dev 530)
  - one-way|and back|round trip|morning and evening: 7 q, 43%, 1 always (dev 64, 423; dev2 195, 262)
  - Total: **22 questions (2.4%)**, containing 10 slice items.
- **Common words:**
  - week: 88 q, 63%, 26 always, 15 slice items
  - year/annual: 22 q, 74%, 8 always
  - pair: 16 q, 82%, 5 always (dev 432 — a pair=2 fact would also break pants-as-1 in that very gold)
  - dozen: 15 q, 82%, 7 always
  - oz/qt/gal/l: 26 q, 77%
  - cents/coins: 22 q, 75%
  - These are too broad: the trigger fires on many always-right questions.

Recall of the unused-number check on the 20 dropped-quantity items:
- It fires on at least 50% of the wrong attempts in only 6/20 items: dev 107 (28/29), 599, 129, 184*, 236*, 14* (* only with number words).
- In the 4 of those 6 items that have right attempts (dev 107, 129, 184, 236), it also fires on every right attempt (`R_unused_w = R`), because the solution legitimately does not re-print that number.
- Zero recall on 11/20 (verbal quantities, or a number that is used but in the wrong role).
- On the misread/plan/world items it fires on at most 3 items each.

## Implant assessment (single call, forward-only, weight-space)

Prior evidence to keep in mind:
- In-context hints get copied (unused-number hint: net 0; fact table: broke more than it fixed; units table: at most +0.1 pp).
- Only a weight-space forget mask or a restart actually changes the answer.
- So each implant below is framed as a **compiled detector that triggers a weight-space action** (a forget/restart of the segment, a steering vector, or a written token from an exact table), not as prompt text.

1. **Integer guard (compiled check on the `####` number → restart/forget the segment; optionally a ceil/floor head).**
   - **Break exposure: zero.** 0/23,144 right attempts and 0/900 golds are non-integer.
   - Fires on 7.5% of wrong attempts (70 items).
   - An exact ceil would directly rescue 53 attempts and floor 37, but the direction is item-dependent, so restart is the safe action and ceil is only safe when the ask is "how many {hours, days, trips, boxes, buses}".
   - In this slice it firmly fixes dev2 159 (ceil rescues 25/28 wrong attempts). It also touches dev 5 (1.5, 0.67), dev2 109 (0.14) and dev2 263, but restarts there would mostly reproduce the misreading.
   - Outside the slice: dev 216, dev2 41 (ceil) and dev 235, 413, 20 (floor) each have >=4 wrong attempts an exact ceil/floor would rescue.
   - Expected value: about +1 slice item, plus fewer than 5 vote-level flips elsewhere.
   - **Lowest risk of any candidate. Recommend building it first: it is cheap, and its trigger is a pure regex on the emitted `####` token stream.**
2. **Rare-word exact fact lookup** (horseshoe→4 per horse, tire→4 per car (+spare), couple(s) room→2 people, BC→larger year is older, born/younger→later year, "for every $N"→floor, round-trip phrase→×2).
   - Fires on 22 questions (2.4%), of which 2 are always-right.
   - Its 10 slice items are the plausible fixes: dev 29, 496, 136, 191, 533, 530, 423, 64; dev2 195, 262. Realistically 3-6 of them flip, given that the model ignored such facts in-context before.
   - Break exposure: at most about 12 non-slice questions with 43-63% accuracy. It must be a steering vector or forced-token write active only after the trigger, not an in-context table.
   - **Do not include the broad words** (week, year, pair, dozen, units). They fire on about 20% of questions, many of them always right. dev 432 shows pair=2 contradicts its own gold; a weekday/365 fact (dev 374) would hit 22 year questions to fix 1.
3. **Question-number register (each question number must be used before `####`).**
   - Precision 30-40%, firing on 3.2-8.6% of right attempts.
   - Recall on the dropped category is 6/20, and in 4 of those 6 it also fires on every right attempt.
   - At most 2-4 plausible slice fixes, with a known 0 net in-context result. **Not recommended.**
4. **Target register (re-read the ask noun at `####`).**
   - Only 1 wrong-target item in the slice (dev 273). The noun-absent trigger fires on 21% of right attempts.
   - The earlier goal-reminder probe gave +0.17 at +118% tokens. **Not worth it here.**
5. **Misreading (17) and plan (13)** have no lexical trigger: the phrases are diverse ("each bedroom", "in the cage", "3/4 times as many", "beats", "and product").
   - A per-phrase steering vector would be one-off.
   - Lever for the 9 cap-dominated plan items: a larger `acap` (320 → 448) or a compiled "drop LaTeX/algebra, restart" on cap hit. This is a structural knob, not an implant; it costs tokens only on the 0.7% of attempts that hit the cap.
6. **Gold/ambiguous (13): unreachable.** Six of these items have a wrong gold label; dev 6, 303, 488 and 519 should be excluded from any never-solved headline.

**Bottom line for this slice.**
- About 13/76 items are label or ambiguity problems, and about 30/76 are idiosyncratic misreadings or plans with no cheap trigger.
- The best single-call implants are:
  - the zero-exposure **integer guard**: about 1 sure item, 0 breaks;
  - a **rare-word exact fact/round-trip lookup**: 2.4% of questions, 3-6 plausible items out of 10 triggered.
- Together they give about 4-7 items, or ≤1 pp of the 900. Raising the attempt cap is the other cheap knob, for about 9 cap-dominated items.
