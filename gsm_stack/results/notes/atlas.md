# Lane 1 divergence atlas (rendered by atlas/report.py; hand labels atlas/data/hand_labels.tsv)

### Heuristic categories (all wrong attempts of mixed items; item = dominant category of its wrong attempts)

| category | dev att | dev item | dev2 att | dev2 item |
|---|---|---|---|---|
| operand | 395 (13%) | 35 (12%) | 253 (14%) | 16 (10%) |
| relation | 1496 (48%) | 145 (49%) | 859 (48%) | 78 (51%) |
| dropped | 565 (18%) | 62 (21%) | 354 (20%) | 34 (22%) |
| unit | 75 (2%) | 5 (2%) | 51 (3%) | 3 (2%) |
| target | 240 (8%) | 29 (10%) | 83 (5%) | 13 (8%) |
| plan | 150 (5%) | 8 (3%) | 117 (6%) | 6 (4%) |
| arithmetic | 23 (1%) | 5 (2%) | 20 (1%) | 1 (1%) |
| early_stop | 98 (3%) | 5 (2%) | 53 (3%) | 2 (1%) |
| format | 14 (0%) | 1 (0%) | 7 (0%) | 1 (1%) |
| other | 49 (2%) | 3 (1%) | 9 (0%) | 0 (0%) |

By n_right band (dev+dev2, attempt-level %, n = wrong attempts):

| category | 1-8 | 9-16 | 17-24 | 25-31 |
|---|---|---|---|---|
| operand | 14% | 13% | 12% | 11% |
| relation | 49% | 50% | 43% | 48% |
| dropped | 16% | 21% | 22% | 20% |
| unit | 3% | 2% | 3% | 1% |
| target | 4% | 4% | 10% | 12% |
| plan | 7% | 5% | 5% | 3% |
| arithmetic | 0% | 1% | 1% | 2% |
| early_stop | 5% | 1% | 3% | 1% |
| format | 0% | 0% | 0% | 2% |
| other | 2% | 1% | 1% | 0% |
| n | 2128 | 1082 | 1041 | 660 |

### Hand labels (stratified sample: 120 mixed items, 20 dev + 10 dev2 per band; the attractor-answer attempt)

| category | dev n | dev2 n | 1-8 | 9-16 | 17-24 | 25-31 | weighted % (all mixed) |
|---|---|---|---|---|---|---|---|
| operand | 4 | 1 | 0/30 | 2/30 | 0/30 | 3/30 | 6% |
| relation | 32 | 17 | 7/30 | 12/30 | 17/30 | 13/30 | 43% |
| dropped | 13 | 10 | 9/30 | 4/30 | 8/30 | 2/30 | 15% |
| unit | 10 | 4 | 5/30 | 4/30 | 2/30 | 3/30 | 11% |
| target | 6 | 2 | 2/30 | 1/30 | 2/30 | 3/30 | 8% |
| plan | 7 | 2 | 2/30 | 3/30 | 1/30 | 3/30 | 8% |
| arithmetic | 3 | 0 | 0/30 | 1/30 | 0/30 | 2/30 | 4% |
| early_stop | 2 | 3 | 4/30 | 1/30 | 0/30 | 0/30 | 3% |
| format | 3 | 1 | 1/30 | 2/30 | 0/30 | 1/30 | 3% |

Heuristic vs hand: exact agreement 62/120; per hand category (recall): operand 2/5, relation 30/49, dropped 10/23, unit 4/14, target 6/8, plan 2/9, arithmetic 0/3, early_stop 5/5, format 3/4

### Where the first divergent step sits

| set | wrong att with D | D = 1st step | D rel. position median [IQR] | D step index median | no D (all values right) |
|---|---|---|---|---|---|
| dev | 2581/3007 | 28% | 0.69 [0.43-0.91] | 1 | 426 |
| dev2 | 1567/1753 | 27% | 0.65 [0.46-0.88] | 1 | 186 |

Median relative position of D by category (dev+dev2): operand 0.57, relation 0.69, dropped 0.72, unit 0.58, plan 0.66, arithmetic 0.73, early_stop 0.53, format 0.92, other 0.97

### Wrong-answer structure (per mixed item)

| set | items | distinct wrong answers (median) | attractor share of wrong (median) | attractor >= 50% of wrong | attractor count > n_right |
|---|---|---|---|---|---|
| dev | 298 | 2 | 0.67 | 69% | 53 |
| dev2 | 154 | 3 | 0.67 | 67% | 33 |

By band (dev+dev2): distinct wrong / attractor share of wrong / attractor type

| band | items | distinct wrong | attr. share | R_canonical_intermediate | R_intermediate | R_rare_value | question_number | novel | none |
|---|---|---|---|---|---|---|---|---|---|
| 1-8 | 74 | 8 | 0.52 | 16% | 3% | 3% | 1% | 69% | 8% |
| 9-16 | 56 | 6 | 0.46 | 14% | 4% | 5% | 2% | 73% | 2% |
| 17-24 | 93 | 4 | 0.58 | 24% | 9% | 0% | 6% | 59% | 2% |
| 25-31 | 229 | 1 | 1.00 | 17% | 3% | 1% | 4% | 74% | 1% |
| all | 452 | 3 | 0.67 | 18% | 4% | 2% | 4% | 70% | 2% |

### Token branch points (wrong attempt vs right attempt with the longest common token prefix, same context group)

| set | wrong w/ partner | branch rel. pos median | p_w median | p_w<0.5 | p_w<0.1 | p_w<p_r | b in D-line | b before D-line | b after D |
|---|---|---|---|---|---|---|---|---|---|
| dev | 2683/3105 | 0.10 | 0.31 | 65% | 24% | 58% | 11% | 89% | 0% |
| dev baseline: all sampled tokens | 1294795 | | | 4% | 1% | | | | |
| dev2 | 1463/1806 | 0.10 | 0.32 | 65% | 22% | 58% | 10% | 90% | 0% |
| dev2 baseline: all sampled tokens | 686749 | | | 5% | 1% | | | | |

Branch token type (dev+dev2) and probabilities:

| subset | n | type: word / digit / punct+space+newline / operator | p_w median | p_r median | p_w<0.1 | p_w+p_r median |
|---|---|---|---|---|---|---|
| all branches | 4146 | 79% / 3% / 16% / 2% | 0.32 | 0.50 | 23% | 0.95 |
| in D-line ("critical token") | 444 | 70% / 9% / 18% / 4% | 0.38 | 0.44 | 17% | 0.95 |
| before D-line | 3651 | 80% / 2% / 16% / 2% | 0.32 | 0.50 | 24% | 0.95 |
| no D (answer = right value) | 524 | 78% / 1% / 19% / 2% | 0.27 | 0.55 | 27% | 0.96 |

Critical (in-D-line) branches by category: operand 81/529 (p_w med 0.27), relation 241/2001 (p_w med 0.47), dropped 64/783 (p_w med 0.41), unit 27/114 (p_w med 0.21), plan 19/220 (p_w med 0.21), arithmetic 5/40 (p_w med 0.29)

p_w histogram of critical tokens: >0.32 54%, 0.1-0.32 29%, 0.032-0.1 10%, 0.01-0.032 4%, <0.01 4%

### Selection gap: seed votes (N=8) that fail although an attempt of that seed is right

| set | gap cases (item x seed) | per seed | winner = 32-attempt attractor | winner count > right count | tie (=) | winner null |
|---|---|---|---|---|---|---|
| dev | 122 | 30.5 | 84% | 85% | 12% | 0 |
| dev2 | 97 | 24.2 | 65% | 76% | 20% | 0 |

Winner type / category (dev+dev2): novel 113, other-wrong 53, right_canonical_intermediate 40, right_intermediate 6, right_rare_value 6, question_number 1 | cat: relation 95, dropped 53, operand 37, target 14, plan 9, unit 6, other 5 | band: 1-8 128, 17-24 20, 9-16 71

Right count in the failing seed: 1: 116, 2: 54, 3: 33, 4: 15, 5: 1

### Examples (hand-labelled; W = wrong divergent line, R = closest right attempt)

**operand**

- dev 237 (one-fifth -> 0.02): W `She spends one-fifth of it on insurance, so the amount spent on insurance is 1600 x 0.02 = $32.` / R `She spends one-fifth of it on insurance, so the amount spent on insurance is 1600 x 0.2 = $320.`
- dev 326 (15 bills vs 25-15=10): W `He spent 20% of these special bills, so he spent 15 * 0.20 = 3 special bills.` / R `He has a 25th birthday, so he has received 25 - 15 = 10 special $1 bills.`
- dev 105 (one-quarter -> half): W `Half of the 12 lollipops is 12 / 2 = 6 lollipops.` / R `One-quarter of 12 is 12 / 4 = 3 lollipops.`
- dev 572 (5/12 -> 1/12 (240/12)): W `Then, calculate the number of students who read one novel. 5/12 of 240 is 240 / 12 = 20 students.` / R `Now, calculate the number of students who read one novel: 5/12 of 240 = (5/12) x 240 = 100 students.`
- dev2 160 (200/2 instead of 4/2 (wrong question number)): W `Mickey runs around the block half as many times as Johnny, so Mickey runs 200 / 2 = 100 times.` / R `Mickey runs around the block half as many times as Johnny, so Mickey runs 200 x 4 / 2 = 400 meters.`

**relation**

- dev 31 (double count of Friday 40x5+56): W `There are 5 weekdays (Monday to Friday).` / R `He borrows books from Monday to Thursday, which is 4 days.`
- dev 191 (wrong reference year 1970-20): W `To find out in what year her brother was born, subtract his current age from her current age: 1970 - 20 = 1950` / R `Therefore, her brother was born in 2010 - 20 = 1990.`
- dev 272 ("beat" needs strict +1): W `He wants to do more than last week, so he needs to do: 324 - 264 = 60 jumping jacks.` / R `This week, he has already done 264, so he needs to do 325 - 264 = 61 more jumping jacks.`
- dev 360 (240 "in all" read as per box): W `So, the number of pomelos shipped last week is 10 x 240 = 2400 pomelos.` / R `Last week, the farmer shipped 10 boxes with 240 pomelos in total. So, the number of pomelos per box is 240 / 1`
- dev 445 (wrong base: half of all birds incl. outside cage): W `After ten minutes, half of the birds in the cage are freed and fly away, so the number of birds that fly away ` / R `After ten minutes, half of the grey birds in the cage are freed, so 40 / 2 = 20 grey birds are freed.`
- dev 132 (2x10-5: cost not per poster): W `2 of them are large posters that sell for $10 each and cost $5 to make, so that's 2 x $10 - $5 = $15.` / R `2 of them are large posters that sell for $10 each and cost $5 to make, so her profit from these large posters`
- dev 211 (240x2 instead of /2): W `So the number of cases she needs is 240 x 2 = 480 bottles.` / R `240 days * 1/2 bottle/day = 120 bottles.`
- dev 216 (needs 90, strict beat): W `So the number of touchdowns Richard needs to have in the final two games is 89 - 84 = 5 touchdowns.` / R `The total number of touchdowns he needs is 89 + 1 (to beat the record) = 90 touchdowns.`

**dropped**

- dev 21 (90/10=9; forgot to add day 1): W `Starting with 100 chips, after the first day, she has 100 - 10 = 90 chips left.` / R `#### 10`
- dev 190 (Caleb's +11 gift): W `He gave 8 candies to Billy, 11 candies to Caleb, so the total candies given to Billy and Caleb is 8 + 11 = 19 ` / R `Billy initially had 6 candies. His father gave him 8 candies, so he now has 6 + 8 = 14 candies.`
- dev 257 (computes 350, never uses it): W `If he only goes 10 meters, he can lift 30% more, so 300 x 1.3 = 390 pounds per hand.` / R `350 x 1.30 = 455 pounds per hand.`
- dev 347 (every-other-lesson rule): W `Each lesson costs $5, so the total cost for the paid lessons is 19 x $5 = $95.` / R `In the first 10 paid lessons, he needs to pay for every other lesson. So he pays for 10 / 2 = 5 lessons.`
- dev 444 (friends' $5): W `#### 65` / R ``
- dev 522 (5 players x 5 rounds): W `Total blocks removed in 6 rounds = 1 (father's block) + 1 (Jess's block) = 2 blocks.` / R `So in 5 rounds, they remove 5 x 5 = 25 blocks.`
- dev 599 (peak hour ignored): W `First, we calculate the number of 15-minute intervals in 120 minutes. 120 / 15 = 8 intervals.` / R `First, we calculate the number of 15-minute periods in 60 minutes: 60 / 15 = 4 periods.`
- dev 39 (x2 bowls per 10): W `He rewarded 20 bowls, so the number of bowls remaining is 70 - 20 = 50 bowls.` / R `Now, calculate the total number of bowls rewarded: (200 / 10) x 2 = 20 x 2 = 40 bowls.`

**unit**

- dev 133 (rate period: per two weeks, missing x2): W `First, calculate her net savings per two-week period: $50 - $46 = $4.` / R `First, calculate her net savings per two weeks. She saves $50 each week, so in two weeks she saves 50 x 2 = $1`
- dev 217 (per day vs per school week, missing x5): W `So the total number of points earned per student per day is 200 points / 25 students / 10 days = 0.8 points pe` / R `So in 10 days, each student can earn 2 x 10 = 20 points.`
- dev 361 (per day + per week, missing x7): W `So per day, he eats 0.25 + 0.25 = 0.5 pounds of vegetables.` / R `So per week, he eats 7 x (1/4 + 1/4) = 7 x (1/2) = 3.5 pounds of asparagus and broccoli per week.`
- dev 281 (3 coaches vs 4 (counting)): W `Then, calculate the number of bottles filled by the coaches. There are 3 coaches (one for each team), so 3 * 2` / R `Each team coach also wanted 2 bottles each, and there are 4 coaches (2 for the football team and 2 for the soc`
- dev 377 (52 weeks in a month): W `The wife saves $100 every week. There are 52 weeks in a month, so in 4 months she saves 100 x 52 x 4 = 20800 d` / R `The wife saves $100 every week, so in 4 months (which has 4 x 4 = 16 weeks), she saves 100 x 16 = $1600.`
- dev 573 (6 days vs 5 school days): W `Osborn's old method took him an average of 3 minutes per day for the week. That means for 6 days, it took him ` / R `So in total, his old method took 3 x 5 = 15 minutes.`
- dev 598 (pots/hour = minutes, missing 60/6): W `The difference in the number of pots produced is 5 - 6 = -1.` / R `In the first hour, it takes 6 minutes to produce each pot. If there are 60 minutes in an hour, the number of p`
- dev 317 (16 oz per quart (is 32)): W `First, we know that each of the 2 bottles is 1.5 quarts. Since there are 4 quarts in a gallon, 1.5 quarts is e` / R `First, convert the 1.5 quarts to ounces. Since 1 quart is 32 ounces, 1.5 quarts is 1.5 x 32 = 48 ounces.`

**target**

- dev 175 (stops at 48, one step early): W `#### 48` / R ``
- dev 462 (per-week 15, missing x6 weeks): W `#### 15` / R ``
- dev 199 (stops at 3000 mg, no conversion to grams): W `#### 3000` / R ``
- dev 394 (stops at 88 pieces, missing /4): W `#### 88` / R ``
- dev 119 (total tissue cost, not per box): W `#### 6` / R ``
- dev 320 (stops at total daisies, missing /10): W `#### 70` / R ``
- dev2 296 (stops at 42 per bag, missing x3): W `The total number of jellybeans in one bag is 24 + 13 + 36 + 28 + 32 + 18 = 151 jellybeans.` / R `So the total number of red and white jelly beans Jessica originally thought would be 42 x 3 = 126.`
- dev2 53 (stops at 12 minutes, missing /4): W `#### 12` / R ``

**plan**

- dev 254 (invented $10 plate price): W `The Taco Grande Plate costs $10.` / R `So Mike's bill is x + 8 = 8 + 8 = 16 dollars.`
- dev 349 (assumes 200 hits): W `For every drum hit, she makes 2.5 cents, which is $0.025.` / R `So her net earnings from hitting the drums would be 5 - 7.5 = -2.5 dollars.`
- dev 53 (Andrena = 2+5): W `According to the problem, Andrena now has 2 more dolls than Christel. So Christel now has 19 - 2 = 17 dolls.` / R `Andrena has 2 more dolls than Christel, so Andrena now has 19 + 2 = 21 dolls.`
- dev 388 (one bank pennies, one dimes): W `#### 6` / R ``
- dev 424 (buys all bag types): W `In total, he can buy 300 + 300 + 400 = 1000 balloons.` / R `#### 400`
- dev 335 (invents continuing rate for remaining 40 min): W `In 40 minutes, she can complete 40 / 20 = 2 more 20-minute intervals.` / R `Therefore, the number of problems she has left to solve is 75 - 30 = 45 problems.`
- dev 449 (invents roses/lilies per place setting): W `1 linen tablecloth ($25 to rent)` / R `- 4 place settings: 4 x $10 = $40`
- dev2 19 (assumes same 180 miles at 70 mph): W `So the distance he needs to cover in the next 2 hours is 70 x 2 = 140 miles.` / R `He needs to average 70 mph over a total time of 3 + 2 = 5 hours to achieve this average.`

**arithmetic**

- dev 235 (sum 25.50 vs 25.00 (garbled calc token)): W `#### 25.50` / R `#### 25.00`
- dev 78 (chained "6 - 3 = 3 x 8 = 24"): W `Each 8 string guitar has 8 strings, so the total number of strings for the 8 string guitars is 24 x 8 = 192 st` / R `In total, John needs 12 + 36 + 24 = 72 strings.`
- dev 558 (10B/6 = 30 -> 5B = 30 algebra slip): W `So B = 30 / 5 = 6 inches.` / R `To find \( B \), multiply both sides by \( \frac{3}{5} \): \( B = 30 \times \frac{3}{5} = 18 \) inches.`

**early_stop**

- dev 28 (truncated LaTeX): W `` / R ``
- dev 106 (truncated LaTeX algebra): W `` / R ``
- dev2 199 (truncated algebra, no answer): W `Substituting \( x = -40d \), we get \( -40d + nd = 3(-40d) \), so \( nd + 40d = -120d \), thus \( nd = -160d \` / R `\[ x = 40 \]`
- dev2 298 (symbolic expression, no number): W `He then pays 10% of his weekly income as tax. So the tax amount is 0.10 x (12x + 8y).` / R `Therefore, the amount he earns every week after buying the ingredients is $126 - $70 = $56.`
- dev2 23 (truncated (also hours x daily wage)): W `- There are 6 employees, and they work 25 days a month, 8 hours a day, so the total number of hours worked in ` / R `Next, calculate the total wages for the managers: 2 employees x $20/hour x 25 days/month x 8 hours/day = 2 x 2`

**format**

- dev 225 (4100 cents vs $41): W `#### 4100` / R ``
- dev 300 (dollars vs cents): W `Each quadruple piece costs 4 cents, so Michael earned Q * $0.04.` / R `Each double piece costs 2 cents, so Thomas earned 45 x 2 = 90 cents.`
- dev 186 ($0.1 -> 100 cents): W `So, you save 10 x $0.1 = $1 on each apple.` / R `So you save $0.1 per apple from the first store.`
- dev2 159 (137.33 not rounded up to 138): W `Her total expenses are $1200 (rent) + $400 (groceries) + $200 (medical expenses) + $60 (utilities) = $1860` / R `Since Madeline can't work a fraction of an hour, she needs to work 138 hours this month.`

