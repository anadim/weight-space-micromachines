# Prior GSM8K failure taxonomies and implant experience (digest, todo prior-tax)

Sources (read-only): `gsm_push/{selection,reading,calc_v2,FINAL}.md`,
`selection/results/{hard_failures_dev.txt,rethink_proxy.json}`, `reading/debug/fail42_dump.txt`,
`reading/common_split.json`, `repo/gsm_push/results/calc_v2/r2/analysis.txt`, `repo/DETAILS.md` (base-host rethink).
The original failure analysis `(local notes)` (+ `dig/scripts/fail/classify.py`) is cited by all
four docs but is **not on this sandbox** (no `local notes`, not in `repo/`, not in `node_code/`); only its second-hand
numbers are given in §1e. Machine-readable labels: JSON block at the end (also `notes/prior_taxonomies.json`).

## 0. Numbering and splits (checked, read this first)
- **dev ids below = dev600 positions = `items_of('dev')` in our `gen/stk/stack4_hook.py`.** Verified locally: split-B gsm
  items in train order (md5%10 in {6,7}, shots removed: 1,462 items), `random.Random(0).shuffle`, `[:600]` reproduces
  `reading/common_split.json` `cdev` exactly, and `[600:900]` = `cdev2`. The `len(q_ids)+320<=544` filter removes
  nothing on split B (max question = 181 tokens with the Qwen3 tokenizer), so filtered and unfiltered orders coincide.
- `hard_failures_dev.txt` "item k" = dev600 position k: all 30 questions equal `train[shuffled_B[k]]` (0 mismatches).
- calc_v2.md §8 `#k` and selection.md `DEV #k` = the same dev600 positions (e.g. #6 Kimiko tiles, #29 horseshoes, #36
  Elvie appear identically in all three). rethink_proxy.json `j` = row position in the set (dev 0-599, dev2 0-299).
- **reading.md §8 `idx` is a GSM8K-train index, not a dev position.** Mapped via `cdev.index(idx)`: F0..F41 = dev
  1,5,6,21,28,29,30,31,36,39,53,59,64,66,76,78,88,90,107,116,117,120,121,129,133,136,140,141,145,164,166,167,170,175,177,
  190,191,193,196,199,205,211 (the first 42 greedy failures, in order). Used in the JSON as dev positions.
- **reading.md §1-§7 "dev/dev2/dev3" are a DIFFERENT split** (`reading/dev_split.json`: `Random(0).sample(B,900)`, dev3 =
  remaining 562). Only its lines marked "common dev600/dev2" use our sets. Its 74.83/77.00/80.60 baselines are on that split.
- dev2 ids = positions 0-299 within dev2 (= `items_of('dev2')`). No hand-labelled dev2 failure taxonomy exists.
- **Host/config of the labelled failures:** all labels are failures of the RL+calc host (a2_u300_s2) *greedy*, calc_s4
  only (no calc v2, no sanity, no serial tries): dev greedy 80.33 (B64) / 79.83 (calc_v2 r7 run) / 79.67 (r2 run), i.e.
  118-122 failures; different runs disagree on a few preds (e.g. dev 39: pred 50 vs -10). Many are already fixed in our
  S3 stack (e.g. 76, 193, 299, 53 by calc v2 / sanity+forget). Cross-reference with the atlas `class` before reuse.

## 1. Every failure category used before, with counts

### 1a. selection.md §8c: the "hard core" (dev600, hand-read)
DEV greedy failures 118; hard core = greedy wrong AND ≤1/16 T=1 samples right: 50 (42% of failures, 8.3% of dev).
30 read by hand (`hard_failures_dev.txt`; 7 of them have k16=1: 90 136 225 361 374 412 598).

| class | n | % of 30 | dev ids | candidate implant (verdict) |
|---|---|---|---|---|
| wrong plan / misread situation | 14 | 47% | 170 88 217 36 533 254 215 445 117 225 374 29 213 530 | none small (needs a better reader) |
| dropped stated quantity / multiplier | 5 | 17% | 107(x20 birds) 386(split w/ friend) 444 90 361 | unused-number checklist + hint: built (8b), +0 |
| unit / world knowledge | 4 | 13% | 317(quarts,oz) 136(couples=2) 432(pairs) 6(6 sq in tile) | unit-fact table: not built |
| relation / direction | 3 | 10% | 598 412 216 | all 3 flagged by sanity; sanity repair changes 0/3 |
| right steps, wrong reference in last step | 2 | 7% | 564(N+5 for "sum now") 240(70-28 not 56-28) | operand-source tracker: not built |
| gold or answer-unit ambiguity | 2 | 7% | 567($3.50 vs 350 cents) 488(gold 7 looks wrong; host 13) | none |

### 1b. reading.md §8: first 42 dev600 greedy failures, by primary side fix
| pattern | n | % of 42 | dev ids | exact trigger? precision | outcome |
|---|---|---|---|---|---|
| impossible final answer (neg/non-int) | 8 (doc: 9) | 19-21% | 1 5 64 66 90 133 141 145 | yes, 100% | compiled sanity (+forget) KEPT |
| calculator coverage / format | 4 | 10% | 76 78 164 193 | yes | calc v2 / chain2 |
| non-integer count intermediate (`= x.y <count noun>`) | 2 | 5% | 117 166 | yes, 83% (B) / 93% (A); only 4/14 beyond sanity | not built, ≤0.2 pp bound |
| zero answer | 1 | 2% | 205 | yes, 100% | built, 0 fixes, dropped |
| unused question number | 5 | 12% | 30 59 88 107 129 | yes, 33-35% | nudge 0 fixes; restart ~0: dropped |
| dangling computed result | 3 (+2 overlap: 1, 29) | 7% | 28 170 196 | yes, 27% beyond sanity | dropped offline |
| negative intermediate | 1 | 2% | 53 | yes, 50% beyond sanity | dropped offline |
| world-knowledge fact | 3 | 7% | 29(4 shoes/horse) 31(5 Fridays/week) 136(couples) | fact table: 4/12 "wrong" facts on wrong answers | would break more: dropped |
| plan / relation / double count / dropped event | 15 (doc: 14) | 33-36% | 6 21 36 39 116 120 121 140 167 175 177 190 191 199 211 | no | no small exact fix |
Per-row first-wrong-step text (fine labels used in the unified JSON): double count 1 21 167 177 199; relation 5 39 116
166 191; wrong earlier number/entity 28 145 190 196; dropped event 120 121; dropped quantity 30 59 107 129 211 (30, 107 =
stops a step early); rate period / per-unit target 66 133; algebra 36; tile wording 6.
Verdict: ~25% of failures have an exact small side fix (impossible 9, calc 4, non-int count 2, zero 1).

### 1c. calc_v2.md §8: first 50 dev greedy failures (calc_s4 only), 45 numeric rows (doc says 44)
By implant family: **A** exact chain arithmetic 5 (76 193 235 299, 288 partly; 9-11%); **B** answer sanity ~9 (39 53 64 66
133 159 216; secondary 5 145; ~20%); **C** unused-quantity ledger 9 (29 30 59 88 107 129 211 291 296; 20%); **D** operator
veto 2 (78 164; 4%); **E** semantic probe/steering ~12-13 (1 5 102 120 121 145 167 190 191 196 237 240; ~28%); **none**
(plan/world) 10 (6 31 36 116 117 136 140 170 207 219; 22%).
By error class: plan 9 (31 39 64 88 116 117 133 140 170); dropped quantity 7 (30 59 107 129 211 291 296); host chain
arithmetic 4 (76 193 235 299); relation direction 3 (5 53 191); dropped event 3 (120 121 190); units/world 3 (6 29 136);
double count 2 (1 167); algebra 2 (36 159); calc-induced continuation 2 (78 164); wrong earlier number 2 (102 240);
wrong entity 2 (145 196); 1 each: unit choice 66, dropped factor 207, rounding 216 (ceil), missing brackets 219 (`24 + 12 /
2`), number transcription 237 (one-fifth -> 0.02), bookkeeping + chain arith 288.

### 1d. Arithmetic-only scans of dev greedy traces (calc_v2 §1, r2 analysis.txt)
- dev 121 failures: calc_s4-written equations 1,838 with 0 wrong; host-written 4+-operand results 40 (9 wrong); brackets/
  comma/sign 32 (3 wrong). Failures containing a wrong host-written result: 10/121 (8%) dev, 6/66 dev2 (9%).
- r2 run (dev 79.67, 122 fails): host:chain 52 eq (21 wrong), host:paren 30 (19 wrong, mostly restatements), failures
  with a wrong host equation 18 (chain 15, paren 5, basic 2, comma 1, sign 1).
- => pure arithmetic is ≤8-10% of failures; calc v2 later fixed 5-6 dev, 3 dev2.

### 1e. Earlier (cited, not available here; test / split-A, NOT dev)
- "About 70% of the RL+calc host's GSM8K failures are reading errors with exact arithmetic: wrong plan, dropped quantity,
  reversed relation, units" (reading.md intro, from gsm8k_failures.md).
- "A question number never written": 18/380 test failures vs 4/939 correct answers (§4 there); dropped entities/events are
  usually words ("half", "both ways"), not printed numbers (§4.4).
- Stable misconception (§2d): 41/75 sanity rethinks repeated the same impossible answer; a single restart re-derives the
  same wrong reading 23/75 times.
- Base host: T=1 37.6 vs greedy 65.8 on train (§3a). Majority@8 on a 300-item test slice 72.0 -> 75.7.
- Offline trigger precisions on split-A dumps: unused number 33-35%; entity never named 24%; unit mismatch 60% precision /
  0.7% recall (6/806 wrong vs 4/3694 right); "answer != last computed value" 55% precision / 1-2% recall.

## 2. Implants tried, per category, and measured outcome
(dev = dev600, dev2 = 300; fixes/breaks vs the stage below; "rd-split" = reading.md's own split; 1462 = all split B)

| category targeted | implant | outcome | why |
|---|---|---|---|
| impossible final answer | T=1 restart(s) on sanity trigger (≤3) | +1.98 on 1462 (29 fixes) | restart = new call: no longer allowed |
| impossible final answer | reworded restart cascade (" We need to find" > re-read > T=1) | +2.53 on 1462, 37/0; FINAL ref on S1-look: dev 80.67->83.67, dev2 79.33->80.67, test 72.18->74.53 | not single-call |
| impossible final answer | in-context phrase (generic / typed / plan / reread) | +0.34..+0.62 on 1462 (≤9/0); test 62/80 fired rows still impossible | **in-context retry copies** the old attempt (17/65 commit same number) |
| impossible final answer | sanity + **weight-space forget** + " We need to find" (S2F) | common dev +2.17 (13/0), dev2 +1.67 (5/0); on calc v2 (S1->S2F) dev +2.50 (15/0), dev2 +1.67 (5/0), test +1.52 (20/0) | KEPT; ~70% of restart gain. Test: 21/79 fired end right, 28 still impossible |
| impossible (sel 8c relation items 598 412 216) | sanity repair | changes 0/3 | stable wrong reading |
| zero answer | zero-class fire | 0 fixes (dev, dev2, stack hook) | dropped |
| host chain arithmetic | calc v2 hook `verify_rsl` (rewinds) | dev +1.00 (6/0), dev2 +1.00 (3/0), scaled +7.0 (22/1) | not forward-only |
| host chain arithmetic | chain2 compiled {+,-} | dev +0.83 (5/0: 235 299 359 354 76), dev2 +1.00 (3/0) | KEPT |
| host chain arithmetic | calc v2 `fwd` in stack (S1) | dev +0.67 (4/0), dev2 +1.33 (4/0), test +0.61 (11/3) | KEPT; 3 test breaks (algebra term, ".00" veto, path change) |
| calc-induced continuation (78, 164) | operator veto after calc value | calc_s4: dev +0.17 (1/0), dev2 0, scaled -0.33 (2/3) | dropped |
| dropped / unused number | ledger nudge (quote unused clause / number) | rd-split dev 0/0, dev2 1/1; fire 4-5.5% | host re-derives same answer (copy) |
| dropped / unused number | ledger -> restart (re-read) | rd-split +0.67 (6/2) / +0.67 (2/0); dev3 +0.36 | not reproducible; restart not allowed |
| dropped / unused number | 8b unused-number hint in-context ("we have not used the N") | dev 2/2 (fix 30 362; break 56 472), dev2 0/0; precision 41%/33%; 23/29 & 17/18 unchanged | low precision + copy |
| plan / misreading | unconditional re-read (RE2, 3x, shots), plan-first, plan list | ±15% answers flip; dev/dev2 +1..+5 but dev3 -1.8..-2.5; T=1 pooled +0.8 n.s. | reshuffle = noise (one T=1 sample: +2.8/+2.3/-2.8) |
| wrong target | goal reminder at `####` | +0.17 (3/2), +118% tokens | dropped |
| wrong target | "answer != last computed value" rethink | +0.17, 1.5% fire | dropped |
| units | units table | offline bound ≤0.1 pp | dropped |
| world facts | fact table | offline: breaks > fixes | dropped |
| low-confidence (any) | L18-probe-or-sanity rethink in-context ("let me check") | dev 2/3 (fix 31 372; breaks e.g. 14 101 311), dev2 1/2 (fix 216; breaks 50 200); fire 12.8-13.7%, precision 0.71-0.76; 53/77 unchanged | detector good, repair copies |
| any | serial tries, no forget, N=4 | dev 80.33 (+0), 93.7% items all-same answer | copy; probe AUROC decays .835 -> .663 per attempt |
| any | serial "solve again using a different method" (T=1, N=4) | dev 78.33 (< greedy 80.33); all-same 88% | copy |
| any | probe-adaptive serial N≤2 | dev 79.67 = N1 at +14% tokens | copy |
| any | serial + forget + position register + probe vote (N=8, T=1) | dev 88.50 (59/10), dev2 87.00 (38/8) vs greedy; on S2F (S3): dev 87.50, dev2 88.33, test 80.44 (+7.05 vs S2F, 129/36) | KEPT = current stack; oracle 90-93 |
| any (vote selection) | K-sample vote (not single-call) | K=8 pv+san dev +5.8 (42/6), dev2 +7.4 (29/7); breaks = stable confident wrong reading (dev 20, dev2 29) | probe = good weight, poor judge (best-of-N select worse than vote) |
| base host (test, ref) | compiled "Hmm, wait. Let me rethink this." always | frozen base 52.01 -> 56.56 test; detector-gated 55.65 (AUROC 0.85) | on RL host same phrase only +0.41 (6/0) on 1462 |

Recurring lesson: **anything that lets the host see its failed attempt converts the repair into a copy** (restatement
rates 69-89% after "let me check", 93.7% all-same in serial without forget). Only removing the attempt from attention
(restart, independent samples, or the weight-space forget) changes answers. Also: on n=600, whole-prompt variants within
±3 pp are noise; a third slice reversed every "win".

Probe facts worth reusing: L18 logistic probe at `####` (split-C fit) AUROC 0.84 (C), within-question 0.77 (dev/dev2) vs
0.60 for mean logprob; 0.782 with calc v2 on; at logit < -0.042 fires 12.8-13.7% with 71-76% precision; per-attempt
AUROC 0.81-0.90 under forget+position register. Does not transfer to GSM-Hard (0.58).

## 3. Proposed but not built (category -> idea)
| category | idea | status / evidence |
|---|---|---|
| wrong reference in last step (564, 240; also 102 145 190 196) | operand-source tracker (which computed quantity feeds the final op) | not built (2/30 hard core; ~6/45 calc list) |
| unit / world knowledge (317 136 432 6 29 31) | exact unit-fact lookup table (like the calculator) | not built; only 2/4 hard-core are unit facts; fact table offline breaks>fixes |
| wrong entity / dropped event / relation (E: ~12/45) | semantic probe or steering implant (needs hidden-state detector) | not attempted ("selection agent's lane") |
| plan misreading (14/30 hard core) | activation-steering retry pushing attempt k away from attempt 1 | not built (argued useless vs copy path; moot now that forget exists — diversity steering inside serial is open) |
| plan / wrong target | latent write of "what is asked" (residual write fitted to target) | not attempted (text versions showed only churn) |
| non-integer count intermediate (117 166) | exact `= x.y <count noun>` trigger | not built: 83-93% precision but ≤0.2 pp bound |
| unused quantity (C, 9/45) | add as an extra flag class of the sanity trigger (needs a phrase) | suggested; only the in-context nudge/hint was measured (0) — never tested with forget |
| dangling computed result / negative intermediate / entity ledger | exact register triggers | dropped offline (27% / 50% / 24% precision) |
| number transcription (237) | fraction-word -> value reader | not built |
| arithmetic forms | `x% of y` operand rule; `x`/`÷` chains; restatement inside brackets; exact-literal substitution (2100 x 0.428571); compiled sign-only negative completion; let zero-only tokens pass after "value." | not built (0 dev fixes for x/÷ in hook study) |
| vote selection | in-graph number reader + in-graph probe dot; better verifiers | reader/probe in-graph not built; multi-layer probe, calc re-check, cross-attempt consistency never tried |
| gold errors (488, 567) | none | — |

## 4. Caveats
- All labels are on greedy failures of an older stage (S0-like host greedy). Our atlas classifies items under S3-serial
  (calc v2 + sanity/forget + 8 tries at T=1); 76 193 299 354 359 235 (calc) and the 13 sanity+forget fixes are likely
  "always/mixed" now. Use the JSON to cross-reference by dev id, not to assume the label still applies.
- Labels come from three readers with different schemes; the same item can carry several (e.g. 29: plan (sel) / world +
  dangling (rd) / unused-quantity C (c2); 88: plan vs unused number; 90: dropped multiplier vs impossible; 216: relation vs
  rounding). The `unified.*` keys are my union mapping to the BRIEF atlas categories; they are multi-label.
- Counts in source docs are slightly off their own tables (8c "29 of them" = 30 rows; rd §8 sanity 9 / plan 14 vs table
  8 / 15; calc §8 "44" = 45 rows). Numbers above follow the tables.
- `out.*` keys are measured implant outcomes (fix/break ids); `*_examples` lists are partial (only ids printed in docs).
- reading.md §1-§7 numbers (dev 74.83 etc.) are on reading's own split, not dev600; don't compare them to our dev600.

## 5. Machine-readable labels (dev = dev600 positions, dev2 = dev2 positions)
```json
{
 "dev": {
  "sel8c.plan_misread": [29, 36, 88, 117, 170, 213, 215, 217, 225, 254, 374, 445, 530, 533],
  "sel8c.dropped_quantity_or_multiplier": [90, 107, 361, 386, 444],
  "sel8c.wrong_reference_last_step": [240, 564],
  "sel8c.unit_world_knowledge": [6, 136, 317, 432],
  "sel8c.relation_direction": [216, 412, 598],
  "sel8c.gold_or_answer_unit_ambiguity": [488, 567],
  "rd8.impossible_final_answer": [1, 5, 64, 66, 90, 133, 141, 145],
  "rd8.calc_coverage_format": [76, 78, 164, 193],
  "rd8.nonint_count_intermediate": [117, 166],
  "rd8.zero_answer": [205],
  "rd8.unused_question_number": [30, 59, 88, 107, 129],
  "rd8.dangling_computed_result": [28, 170, 196],
  "rd8.dangling_computed_result_secondary": [1, 29],
  "rd8.negative_intermediate": [53],
  "rd8.world_knowledge_fact": [29, 31, 136],
  "rd8.no_small_fix_plan_relation_doublecount_event": [6, 21, 36, 39, 116, 120, 121, 140, 167, 175, 177, 190, 191, 199, 211],
  "c2_8.A_exact_chain_arith": [76, 193, 235, 288, 299],
  "c2_8.B_answer_sanity": [39, 53, 64, 66, 133, 159, 216],
  "c2_8.B_answer_sanity_secondary": [5, 145],
  "c2_8.C_unused_quantity_ledger": [29, 30, 59, 88, 107, 129, 211, 291, 296],
  "c2_8.D_operator_veto": [78, 164],
  "c2_8.E_semantic_probe_steer": [1, 5, 102, 120, 121, 145, 167, 190, 191, 196, 237, 240],
  "c2_8.none": [6, 31, 36, 116, 117, 136, 140, 170, 207, 219],
  "c2_8cls.double_count": [1, 167],
  "c2_8cls.relation_direction": [5, 53, 191],
  "c2_8cls.units_world": [6, 29, 136],
  "c2_8cls.dropped_quantity": [30, 59, 107, 129, 211, 291, 296],
  "c2_8cls.plan": [31, 39, 64, 88, 116, 117, 133, 140, 170],
  "c2_8cls.algebra": [36, 159],
  "c2_8cls.unit_choice": [66],
  "c2_8cls.host_chain_arith": [76, 193, 235, 299],
  "c2_8cls.calc_induced_continuation": [78, 164],
  "c2_8cls.wrong_earlier_number": [102, 240],
  "c2_8cls.dropped_event": [120, 121, 190],
  "c2_8cls.wrong_entity": [145, 196],
  "c2_8cls.dropped_factor": [207],
  "c2_8cls.rounding": [216],
  "c2_8cls.missing_brackets": [219],
  "c2_8cls.number_transcription": [237],
  "c2_8cls.bookkeeping_plus_chain_arith": [288],
  "out.calc_v2_hook_fix": [76, 78, 193, 299, 354, 359],
  "out.chain2_fix": [76, 235, 299, 354, 359],
  "out.calc_v2_r2_ext_fix": [76, 193, 299, 354, 405],
  "out.sel8b_probe_check_fix": [31, 372],
  "out.sel8b_probe_check_break_examples": [14, 101, 311],
  "out.sel8b_unused_hint_fix": [30, 362],
  "out.sel8b_unused_hint_break": [56, 472],
  "out.sel8b_sanity_typed_fix": [372],
  "out.kvote_fix_examples": [53, 76],
  "out.kvote_break_examples": [20],
  "out.forget_mask_fix_example": [53],
  "hard_core_listed_30": [6, 29, 36, 88, 90, 107, 117, 136, 170, 213, 215, 216, 217, 225, 240, 254, 317, 361, 374, 386, 412, 432, 444, 445, 488, 530, 533, 564, 567, 598],
  "hard_core_k16_eq1": [90, 136, 225, 361, 374, 412, 598],
  "unified.plan": [21, 29, 31, 36, 39, 64, 88, 116, 117, 133, 140, 141, 159, 170, 175, 213, 215, 217, 225, 254, 374, 445, 530, 533],
  "unified.dropped_quantity_or_event": [30, 59, 88, 90, 107, 120, 121, 129, 190, 207, 211, 291, 296, 361, 386, 444],
  "unified.early_stop": [30, 107],
  "unified.wrong_operand_reference": [28, 102, 145, 170, 190, 196, 240, 564],
  "unified.relation_direction": [5, 39, 53, 116, 166, 191, 216, 412, 598],
  "unified.unit_world_knowledge": [6, 29, 31, 133, 136, 317, 432],
  "unified.wrong_target": [66, 217],
  "unified.double_count": [1, 21, 167, 177, 199],
  "unified.arithmetic_or_calc_format": [76, 78, 164, 193, 219, 235, 237, 288, 299, 354, 359, 405],
  "unified.gold_or_ambiguity": [6, 488, 567],
  "unified.impossible_final_answer_flag": [1, 5, 39, 53, 64, 66, 90, 133, 141, 145, 159, 216, 254, 374, 412, 598]
 },
 "dev2": {
  "out.sel8b_probe_check_fix": [216],
  "out.sel8b_probe_check_break": [50, 200],
  "out.kvote_fix_examples": [24],
  "out.kvote_break_examples": [29]
 }
}
```
