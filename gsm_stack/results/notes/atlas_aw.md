# Failure atlas of the all-weights stack (todo `atlas-aw`, task [2].4)

Stack: SerialWD all-weights host plus the weight-space position reset (rung 1, WPR), with the **fixed** WPR code (`_wpr_ar` compaction fix, see implants/aw/README.md "ladder-r2").
Sets: dev600 and dev2 (300 items). 4 seeds (7/11/13/17), N=8 serial attempts, T=1. GSM8K test was not used.
Code is in atlas/*_aw.py, data in atlas/data_aw/, and the resume log in atlas/README_aw.md.
Rows are gen format from `aw.sh r1atl_{set}_s{seed} AW_SETS=set AW_SEEDS=seed WPR=1`, one process per (set, seed). They were produced by j23317, plus j23319 for dev seed 7.

Status: complete for rung 1 (dev and dev2, 4 seeds) and for rung 2 on dev (r2f_dev, fixed code, 4 seeds). Rung-2 dev2 (j23340) was still queued when this was written; rerun §2 on aw/out/r2f_dev2 when it lands (commands in atlas/README_aw.md).
Rung-1 dev seed 7 comes from aw/out/r1f_dev (j23319, fixed code). All other rung-1 rows come from aw/out/r1atl_{set}_s{seed} (j23317).

## 0. Headline

1. **The "residual WPR decay" is the `_wpr_ar` bug, not the model.**
   - The rung-1 fe tree (implants/wpr/fe/out/fe, `FAST_COMPACT=1`, old code) shows per-attempt accuracy decaying over k, along with growing no-answer garbage ("D D D D", "total total total"):
     - dev: 80.7 → 74.5 at k=6; no-answer 19 → 231 of 2400.
     - dev2: 79.5 → 67.4 at k=6.
   - The same runs with the fixed code (`_wpr_ar` `.view(1,-1)`, so fastroll compaction no longer permutes it) are **flat**:
     - dev: 80.8 / 81.1 / 80.1 / 81.0 / 80.4 / 82.0 / 80.5 / 80.9. No-answer is ≤1.1% at every k.
     - dev2: 79.5 … 78.2. No-answer ≤1.2%.
     - This is the same as the programmatic reset pr1 (dev 80.4 … 81.5).
   - Confirmation with dev seed 7: rows from a pre-fix process (aw/out/r1atl, whose log lists `_wpr_ar` among the compaction moves) reproduce the fe-tree decay exactly: identical per-k accuracies, and identical per-k no-answer counts (3, 6, 17, 23, 31, 41, 59).
   - So, compared with rung 0's 80.7 → 29.5, the fixed WPR flattens the decay **fully**. No position-decay lever is left at N=8.
   - The fixed code also moves the votes: dev seeds 11/13/17 are 88.83 / 88.50 / 87.50, versus 88.00 / 88.50 / 86.83 in the fe tree. dev2 is 86.33 / 86.33 / 85.67 / 86.00, mean 86.08 (fe tree 85.92).
   - Fixed rung-1 vote in these atlas rows: dev 88.67 / 88.83 / 88.50 / 87.50 = **88.38**; dev2 86.33 / 86.33 / 85.67 / 86.00 = **86.08**.
     - j23319's r1f_dev (4 seeds in one process, fixed code) gave 88.67 / 89.50 / 88.33 / 88.67 = 88.79.
     - Run-to-run nondeterminism under batching is therefore about ±0.5 per seed. LEAKCHECK found 2/600 rows that differ bitwise between a multi-seed process and a fresh one.
2. **WPR leaves the error profile unchanged.** The heuristic category shares of the fixed rung 1 match the old S3 atlas within about 4 points in every category (§3). The all-weights stack plus WPR has the same failure modes as S3. What remains is reasoning errors, not mechanics.
3. **Vote errors split evenly.**
   - In about half, some attempt of the stream was right: dev 4.79 / dev2 7.00 points.
   - In the other half, no attempt was right: dev 6.83 / dev2 6.92 points.
     - Never-solved items account for 4.0 / 4.0 of those points.
     - Of the never items, 11 are gold-ambiguous (S3 never labels), which is about 1.0 / 1.7 points that cannot be fixed.
   - The selection gap cannot be solved by reweighting.
     - The winner has more votes than the right answer in 80% (dev) and 67% (dev2) of cases.
     - The probe `lg` of the right answer beats the winner's in only 44% (dev) and 27% (dev2), i.e. chance or worse.
4. **Rung 2 (fixed code, dev) = 89.58**, oracle 94.54, oracle@8 93.75.
   - The extended attempts show no decay within a stream, and no-answer stays ≤1.6% at every k.
   - The +1.2 over rung 1 comes from fewer none-right streams (6.83 → 5.46). The selection gap is unchanged (4.79 → 4.96).
   - The old aw/out/r2_* rows, including seed 7, are bug-affected.

## 1. Classes, per-attempt accuracy by k, oracle, vote

Fixed rung 1, all 4 seeds (32 attempts per item). Source: atlas/data_aw/r1/analysis.md.

| | dev | dev2 |
|---|---|---|
| always / mixed / never (of 32) | 288 (48.0%) / 288 (48.0%) / 24 (4.0%) | 138 (46.0%) / 150 (50.0%) / 12 (4.0%) |
| vote (mean of 4 seeds) | 88.38 | 86.08 |
| oracle@8 | 93.17 | 93.08 |
| vote wrong, some attempt right (selection gap) | 4.79 | 7.00 |
| vote wrong, no attempt right | 6.83 | 6.92 |
| mean attempt accuracy | 80.86 | 78.90 |

Per seed (vote / oracle@8 / gap / none right):
- dev s7: 88.67 / 93.00 / 4.33 / 7.00
- dev s11: 88.83 / 93.33 / 4.50 / 6.67
- dev s13: 88.50 / 93.17 / 4.67 / 6.83
- dev s17: 87.50 / 93.17 / 5.67 / 6.83
- dev2 s7: 86.33 / 94.67 / 8.33 / 5.33
- dev2 s11: 86.33 / 92.33 / 6.00 / 7.67
- dev2 s13: 85.67 / 93.00 / 7.33 / 7.00
- dev2 s17: 86.00 / 92.33 / 6.33 / 7.67

Per-attempt accuracy by attempt index k, in %. "none" is the % of attempts with no `####`.

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| rung 0 (no reset), dev | 80.7 | 69.1 | 61.8 | 53.3 | 46.5 | 40.2 | 33.5 | 29.5 |
| rung 1 fe tree (old WPR code), dev | 80.7 | 81.0 | 79.2 | 79.5 | 77.7 | 77.5 | 74.5 | (n/a) |
| pr1 programmatic reset (fe tree), dev | 80.4 | 81.2 | 81.3 | 81.5 | 81.3 | 80.4 | 81.5 | (n/a) |
| **rung 1, fixed WPR, dev** | 80.8 | 81.1 | 80.1 | 81.0 | 80.4 | 82.0 | 80.5 | 80.9 |
| fixed dev, none % | 0.7 | 0.8 | 0.8 | 0.8 | 1.1 | 1.0 | 0.8 | 0.9 |
| **rung 1, fixed WPR, dev2** | 79.5 | 80.0 | 79.8 | 79.4 | 77.5 | 77.0 | 79.8 | 78.2 |
| fixed dev2, none % | 0.2 | 0.8 | 0.7 | 1.2 | 0.8 | 0.7 | 0.8 | 0.7 |
| rung 1 fe tree (old code), dev2 | 79.5 | 79.8 | 78.8 | 76.3 | 72.4 | 69.8 | 67.4 | (n/a) |

The fe-tree k=7 is not comparable: the quick fe parser mixes in the vote line.

Residual decay in the fixed rows: dev k0 → k7 is +0.1 (none); dev2 is -1.3 (k0-3 mean 79.7, k4-7 mean 78.1). The dev2 dip is within seed noise (1200 attempts per k, SE about 1.2). **There is no decay lever.**

Accuracy by attempt start position (stream token) still falls: dev 85.5 → 67.1 → 50.2 → 36.2% for starts at 0 / 1000 / 1500 / 2000.
- Per-k accuracy is flat, so this is item difficulty: only hard items produce long streams.
- No-answer only exceeds 5% for starts beyond 1500 tokens (about 3% of attempts). The 3500+ bin shows 71%, but n = 31.

In the bug-affected fe tree, the same bins showed far more no-answer. In the table below, "none" is the % of attempts with no answer.

| start | dev WPR (old) none / acc | dev pr1 none / acc | dev2 WPR (old) none / acc | dev2 pr1 none / acc |
|---|---|---|---|---|
| 0-499 | 0.9 / 84.6 | 0.2 / 85.0 | 0.8 / 83.3 | 0.3 / 83.7 |
| 500-999 | 4.8 / 76.9 | 0.4 / 80.2 | 9.4 / 71.3 | 0.5 / 77.4 |
| 1000-1499 | 22.0 / 47.1 | 2.5 / 60.9 | 33.5 / 37.9 | 1.1 / 59.0 |
| 1500-1999 | 36.6 / 31.2 | 9.4 / 45.1 | 66.9 / 12.9 | 5.0 / 41.8 |

Source: atlas/data_aw/fe/{pos,perk,perk_none}.txt.

## 2. Rung 2 (A+B+C+stop4, extended attempts up to k=16)

Rows: aw/out/r2f_dev (j23319, fixed code, SW_NS=16, adaptive A=0.7/16/4 S=4/0.5 B=3 C=400). Analysis: atlas/data_aw/r2f/{analysis,diverge}.md.

The earlier aw/out/r2_* rows are bug-affected and must not be used:
- seeds 11/13/17 collapse;
- seed 7 has 40-60% no-answer attempts at k12-15, which is entirely the `_wpr_ar` bug.

| dev (fixed) | s7 | s11 | s13 | s17 | mean |
|---|---|---|---|---|---|
| vote | 89.67 | 90.17 | 89.17 | 89.33 | **89.58** |
| oracle (all attempts) | 94.17 | 95.50 | 94.00 | 94.50 | 94.54 |
| oracle@8 | 93.17 | 95.17 | 93.67 | 93.00 | 93.75 |
| gap (some right, vote wrong) | 4.50 | 5.33 | 4.83 | 5.17 | 4.96 |
| none right | 5.83 | 4.50 | 6.00 | 5.50 | 5.46 |
| attempts / stream | 6.67 | 6.75 | 6.62 | 6.76 | 6.70 |

Over 4 seeds, the classes are always 313 / mixed 270 / never 17. There are fewer than 32 attempts per item because of adaptive stopping.

Per-k accuracy (k = 0..15): 81.8, 82.1, 81.8, 81.5 | 60.6, 59.7, 58.4, 58.5 | 38.5, 39.4, 39.5, 36.5 | 33.4, 32.9, 35.0, 35.4.
- No-answer is ≤1.6% at every k, versus 40-60% at k12-15 in the bug rows.
- The drops at k=4, 8 and 12 are adaptive selection: only hard streams get more attempts.
- Within streams of the same length, accuracy is **flat or rising** in k:
  - 16-attempt streams (n=308): k0-3 30.7, k4-7 30.0, k8-11 32.8, k12-15 34.5;
  - 8-attempt streams: k0-3 72.6, k4-7 77.7.
- So the extended attempts show no position decay either.

The extension past 8 attempts rescues the oracle in only 19 of 2400 streams (+0.79).
- Rung 2 gains +1.2 vote over rung 1 (89.58 vs 88.38).
- That gain comes mainly from the bias, stop and cap of A+B+C on k<8: none-right falls from 6.83 to 5.46, and oracle@8 rises from 93.17 to 93.75.
- The selection gap is unchanged: 4.96 vs 4.79.

Rung-2 selection gap (119 cases):
- winner count > right count in 110 (92%); ties 7.
- max lg(right) > max lg(winner) in 40%; median lg right-max 0.60 vs winner-max 0.88.
- So the adaptive stack makes wrong winners more confident; a voter needs a new signal.

Heuristic categories (rung-2 dev, 4065 wrong attempts of 252 mixed items): operand 12, relation 45, dropped 20, unit 4, target 7, plan 6, arithmetic 1, early_stop 2, format 1, other 1 (%). This is the same profile as rung 1 and S3.
- diverge.analyse_item raised StopIteration on 18 items, which were skipped.
- Rung-2 dev2 is pending (j23340): run `python3 atlas/parse.py --gen atlas/data_aw/r2f ...` after copying aw/out/r2f_dev2/*.gz (see README_aw.md).

## 3. Divergence categories of mixed items

### Heuristic

The heuristic table is from atlas/diverge_aw.py, which wraps atlas/diverge.analyse_item. It covers all wrong attempts of mixed items.

The heuristic's partner choice is order-dependent, so the shares move by about ±2 points between reruns.

| category | rung-1 dev att | dev item | rung-1 dev2 att | dev2 item | rung-2 dev att | old S3 heur att % | old S3 hand % |
|---|---|---|---|---|---|---|---|
| operand | 382 (13%) | 30 (10%) | 225 (14%) | 19 (13%) | 12% | 13 | 6 |
| relation | 1280 (44%) | 140 (49%) | 744 (45%) | 78 (52%) | 45% | 48 | 43 |
| dropped | 556 (19%) | 49 (17%) | 347 (21%) | 26 (17%) | 20% | 19 | 15 |
| unit | 78 (3%) | 6 (2%) | 40 (2%) | 2 (1%) | 4% | 2 | 11 |
| target | 266 (9%) | 36 (12%) | 78 (5%) | 11 (7%) | 7% | 7 | 8 |
| plan | 163 (6%) | 13 (5%) | 107 (7%) | 6 (4%) | 6% | 5 | 8 |
| arithmetic | 25 (1%) | 7 (2%) | 34 (2%) | 2 (1%) | 1% | 1 | 4 |
| early_stop | 126 (4%) | 6 (2%) | 51 (3%) | 4 (3%) | 2% | 3 | 3 |
| format | 14 (0%) | 1 (0%) | 7 (0%) | 2 (1%) | 1% | 0 | 3 |
| other | 16 (1%) | 0 | 9 (1%) | 0 | 1% | 1 | - |

Mixed items: dev 288 (2906 wrong attempts), dev2 150 (1642).
Top sub-categories on dev: relation/wrong_ref 811, relation/op_or_order 313, dropped/missing_operand 304, operand/wrong_q_number 281, target/missing_last_scale 187, dropped/missing_branch 144, early_stop/no_answer 126.

For contrast, the bug-affected dev seed 7 rows (data_aw/r1s7) had early_stop/no_answer = 28% of wrong attempts.

### Hand labels

There are 57 hand labels, written by reading each wrong attempt against the gold and the closest right attempt.
- Source: atlas/data_aw/hand_labels_aw.tsv, sampled by atlas/hand_sample_aw.py.
- The sample is dev seed 7, from the pre-fix j23311 rows; for answered attempts this does not matter.
- It is stratified by the stream's n_right (1-2 / 3-4 / 5-6 / 7 of 8). Each label is for the attractor-answer wrong attempt.
- The sample contains only answered attempts (no-answer attempts were excluded).

| category | hand n (%) | heuristic on the same 57 | old S3 hand (weighted) % |
|---|---|---|---|
| relation / direction / misread | 22 (39%) | 29 (51%) | 43 |
| dropped quantity / condition | 11 (19%) | 9 (16%) | 15 |
| wrong target / missing last scale | 9 (16%) | 7 (12%) | 8 |
| unit / world fact | 8 (14%) | 2 (4%) | 11 |
| operand (miscopy / wrong question number) | 3 (5%) | 8 (14%) | 6 |
| plan | 2 (4%) | 1 (2%) | 8 |
| arithmetic | 1 (2%) | 0 | 4 |
| early stop | 1 (2%) | 0 | 3 |

Exact agreement with the heuristic is 26/57 = 46%; the old atlas had 62/120 = 52%. With dropped/target/early_stop merged, agreement is 29/57 = 51%.

The main confusions (heuristic → hand):
- relation → dropped: 5
- relation → unit: 4
- operand → target: 3
- operand → dropped: 2
- dropped ↔ target: 2

The heuristic under-counts unit/world failures by about 3.5×. These show up as "relation" because the wrong line has a wrong operator or reference, but the cause is a missing fact:
- Monday–Friday = 5 days (the attempt used 6);
- a school week = 5 days (it used 7);
- 1 quart = 32 oz;
- 24 h per day (it wrote 2);
- 1 pound = 2 half-pounds;
- average per hour versus per minute;
- days → weeks.

Compared with the S3 hand shares:
- relation is a little lower (39 vs 43);
- target is higher (16 vs 8): "stops one scale short", e.g. ×$2, /3, /2, mg→g;
- dropped and unit are slightly higher.

The rung-1 hand sample excludes no-answer attempts, which are about 1% after the fix.

## 4. Selection gap ("some attempt right, vote wrong")

Source: atlas/data_aw/r1/analysis.md and the gap_{set}.json files.

| | rung-1 dev | rung-1 dev2 | rung-2 dev | old S3 dev / dev2 |
|---|---|---|---|---|
| gap cases (item × seed) | 115 (28.8 per seed) | 84 (21.0) | 119 (29.8) | 122 / 97 |
| winner count > right count | 92 (80%) | 56 (67%) | 110 (92%) | 85% / 76% |
| tie | 21 | 21 | 7 | 12% / 20% |
| right count > winner count | 2 | 7 | 2 | |
| right count in failing stream, 1 / 2 / 3 / 4+ | 60 / 31 / 21 / 3 | 40 / 23 / 12 / 9 | 46 / 32 / 19 / 22 | 116 / 54 / 33 / 16 (both sets) |
| max lg(right) > max lg(winner) | 44% | 27% | 40% | |
| median max lg, right vs winner | 0.69 vs 0.67 | 0.36 vs 0.74 | 0.60 vs 0.88 | |
| ties where mean lg(right) > mean lg(winner) | 2 / 21 | 0 / 21 | 0 / 7 | |

Reading:
- Most gap cases (80% / 67%) have the right answer out-voted. More than half have only 1 right attempt among the 8. lg cannot tell right from wrong (44% / 27%).
- Ties are lost because the lg weighting favours the wrong answer: mean lg(right) ≤ mean lg(winner) in 19 of the 21 dev ties and all 21 dev2 ties.
  - A tie-break by anything except lg would win about half of them: +0.4 dev / +0.9 dev2 (winning all 21 would be +0.9 / +1.75).
- Only an independent correctness signal attacks the remaining gap: a round-trip check, or a verifier probe trained on m12 attempts.

## 5. Lists for the mechanistic lanes

All lists are in atlas/data_aw/. They are built by atlas/lists_aw.py from the fixed rung-1 rows. Each item carries set, idx, question, gold, n_right / n_wrong / n_att, vote_ok per seed, and category counts.

| list | items (dev / dev2) |
|---|---|
| m12_mixed_aw.json | 253 (166 / 87) |
| m3_dropped_target_aw.json | 108 (74 / 34) |
| m4_unit_world_aw.json | 20 (14 / 6) |
| m5_attractor_aw.json | 154 (97 / 57); the attractor beats the right count in 64 items; 25 attractors in never-right items |

Also in atlas/data_aw/:
- hand_labels_aw.tsv: 57 hand labels with notes / facts.
- r1/{attempts,items,diverge}_{set}.jsonl.gz: full per-attempt tables.
- r1/{analysis,diverge,ceiling,never,lists}.md.
- r2f/ (rung 2 dev).
- fe/ (bug-tree position evidence).

- **m12_mixed_aw.json (M1 attention forensics / M2 steering).** Mixed items with ≥4 right and ≥4 wrong attempts. Every attempt carries:
  - `ok`, `ans`, `lg`;
  - `seg_start` (stream token offset of the attempt's final segment);
  - the first-divergence category, sub-category and detail;
  - `div_tok`: index of the divergent step's token in the final segment's `seg_tokens`, which start with the writer phrase;
  - `div_line_tok` and `div_line`;
  - `b_tok`, `b_partner`, `b_wtok` / `b_rtok`, `lp_w` / `lp_r`: the token branch point versus the right attempt with the longest common prefix, the wrong and right tokens, and their log-probs;
  - `right_line`.
  - Absolute stream offset = `seg_start + div_tok` (tokens).
- **m3_dropped_target_aw.json (M3).** Items whose wrong attempts are ≥50% dropped or target (and ≥2 of them), or hand-labelled as dropped/target. Each item has its question numbers and up to 6 example wrong attempts, each with its divergent line and the matching right line.
- **m4_unit_world_aw.json (M4 skill injection).** Items with ≥2 unit/world-fact wrong attempts, or a hand unit label with clean wrong attempts. Each item has:
  - `facts_from_question`: a keyword → fact table (dozen = 12, week = 7 days / 5 weekdays, day = 24 h, hour = 60 min, quart = 32 oz = 2 pints, pound = 16 oz, …);
  - `hand_fact`: the fact the hand label says was missing.
  - Add the S3 never-items labelled world_units: 7 dev + 1 dev2 rung-1 never items (atlas/never_aw.py).
- **m5_attractor_aw.json (M5 diversity).** Items with ≥4 wrong attempts. Each has the attractor answer and its share of the wrong answers, whether it beats the right count, the number of streams with ≥3 attractor attempts, the attractor type, and the attractor attempts' first lines.

## 6. What the remaining errors point to: weight-space implant ideas, ranked

Ceilings are in vote points at N=8 (dev / dev2). "Full" means the category is fixed completely in every vote-wrong stream where it is the dominant wrong category; "realistic" is my estimate.
- Basis: atlas/ceiling_aw.py, the vote-error split by the dominant category of the stream's wrong attempts (heuristic), corrected with the hand-label confusion.
- The total vote error is about 11.6 / 13.9 points, of which about 1.0 / 1.7 is gold-ambiguous.

Vote-error points by the dominant category of the stream's wrong attempts (atlas/data_aw/r1/ceiling.md):

| dominant category | dev gap | dev none-right | dev total | dev2 gap | dev2 none-right | dev2 total |
|---|---|---|---|---|---|---|
| never item | 0 | 4.00 | 4.00 | 0 | 4.00 | 4.00 |
| relation | 1.83 | 1.12 | 2.96 | 3.42 | 1.00 | 4.42 |
| dropped | 1.21 | 0.33 | 1.54 | 1.33 | 1.00 | 2.33 |
| operand | 0.62 | 0.67 | 1.29 | 1.42 | 0.17 | 1.58 |
| target | 0.54 | 0.12 | 0.67 | 0.08 | 0 | 0.08 |
| plan | 0.42 | 0.17 | 0.58 | 0.58 | 0.25 | 0.83 |
| early stop | 0 | 0.29 | 0.29 | 0 | 0.42 | 0.42 |
| unit | 0.12 | 0.12 | 0.25 | 0.08 | 0.08 | 0.17 |
| all | 4.79 | 6.83 | 11.62 | 7.00 | 6.92 | 13.92 |

Never items: dev 24, dev2 12. All were already ≤4/32 under S3; S3 labelled them (notes/never.md) as:
- dev: world_units 7, gold_ambig 6, dropped 4, misread 4, plan 3;
- dev2: gold_ambig 5, misread 2, dropped 2, plan 1, arith_format 1, world_units 1.

Gold-ambiguous points (unfixable): dev 6/600 = 1.0, dev2 5/300 = 1.7.

Hand-label correction: the heuristic labels about half of the hand dropped / unit / target cases as relation or operand. The unit / dropped / target ceilings below are therefore scaled up from the heuristic, and relation / operand scaled down.

| # | implant idea | what it targets | full ceiling | realistic |
|---|---|---|---|---|
| 1 | **Verifier-weighted vote**: a weight-space verifier register, e.g. a round-trip check that substitutes the answer back into the question's relations, or a per-attempt "consistency" probe trained on mixed items, replacing the plain count/lg vote | the selection gap. lg is at chance (44%/27% right>winner) and the winner out-counts the right answer in 80%/67%, so a new signal is needed, not a reweighting | 4.8 / 7.0 | +1.0 – 2.0 |
| 2 | **Relation / direction guard** (M1/M2): steer or patch at the divergence token (m12 list; the branch is usually before the divergent line, with p_w median 0.31 / 0.29, and p_w < 0.1 in 25%) | the largest category: ~40-45% of wrong attempts, dominant in 3.0 / 4.4 points of vote errors (hand labels: about 1/4 of these are really dropped / unit) | 2.3 / 3.3 | +0.5 – 1.0 |
| 3 | **Question-quantity coverage register** (M3): every number in the question is either used or explicitly dismissed before `####`; if not, trigger a repair | dropped quantity: 1.5 / 2.3 points, plus part of the gap. The hand labels show it is under-counted (relation → dropped confusions) | 1.5 / 2.5 | +0.5 – 1.0 |
| 4 | **Unit / world-fact skill injection** (M4): fact rows (5 weekdays, 7 days/week, 24 h, 60 min, 32 oz/quart, 12/dozen, 1000 mg/g, half-pound pricing) injected as a key-value memory into the MLP or the stack | unit/world: heuristic 0.25 / 0.17, but the hand labels show ~3.5× under-count. Plus 7 / 1 never items labelled world_units (1.2 / 0.3) | 2.0 / 1.5 | +0.5 – 1.0 |
| 5 | **"Answer what is asked" target check** (M3b): compare the question's target noun and unit with the last line, and add the missing final scale (×price, /count, unit conversion) | target / missing last scale: 0.7 / 0.1 (hand: 16% of answered wrong attempts) | 0.8 / 0.4 | +0.3 |
| 6 | **Attractor-breaking diversity** (M5): penalise re-deriving the previous attempts' answer path (attempt k sees the earlier answers), or force a different first step | attractor beats right count in 64 items; this overlaps the gap and none-right streams | within #1 + none-right | +0.3 – 0.7 |
| 7 | **Operand copy guard**: a copy-check of every number against the question | operand: 1.3 / 1.6 (the hand labels say the heuristic over-counts it about 2.5×) | 0.7 / 0.7 | +0.2 |
| 8 | ~~WPR residual-decay fix~~: **done** by the `_wpr_ar` fix (+0.4 dev / +0.2 dev2 over the fe-tree rung-1 numbers 87.96 / 85.92) | none | 0 | 0 |
| 9 | More attempts (rung-2 style extension) | oracle@8 → oracle@16: +0.79 on dev (19 / 2400 streams) | 0.8 | ≤ +0.3 |

Never items that no implant above can fix: 4.0 / 4.0 points, of which gold-ambiguous is 1.0 / 1.7. The rest are S3-labelled world_units, dropped, plan or misread, and need capability.
