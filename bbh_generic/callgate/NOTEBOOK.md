# Call gate: fix BBH tool over-calling, in weights (Weight-Space Micromachines)

Host = runs/ft_int_v2 (Qwen3-0.6B-Base, full FT on 432 rows; the integration v2 host). Tools = v2 Attached
(A/B call form: track,order,interval,walk,truth + Team C PASSIVE expr/calc/bool). Pod: an A40 machine (GPU 1),
plus an H100 machine for parallel BBH runs.
Code: code/ (cg_*.py). NEVER trained on BBH; A_oos2/B_oos/C_ood untouched by the probe.

## Problem (from TABLES_v1_vs_v2.md)
v2 FT+tools on real BBH: 27-mean 48.36, 12 tool tasks 84.33, but 15 no-tool tasks 19.58 (base 39.15) because the host
writes `<tool_call>` on 59.2% of no-tool questions; A/B implants answer empty and the host loops to the token cap.
In v2 Team C is PASSIVE (fires at the host's own `=`, never emits `<tool_call>`), so ALL spurious `<tool_call>` openers
are A/B (track 6442, interval 3939, order 3676, walk 1584, truth 536 on the 15 no-tool tasks). The gate only needs to
veto spurious A/B openers; it is neutral to the 3 Team C tasks (boolean_expressions, multistep_arithmetic_two, object_counting).

## Design
`<tool_call>` is a SINGLE token (id 151657) -> one clean decision point. Two variants, both in weights:

- **GATE** (cg_gate_gen.py mode=gate): a linear probe reads the host's post-final-norm last hidden state h (the vector
  fed to the LM head) at the step the host is about to emit `<tool_call>`. Score s = w·h + b. If the host's argmax is
  `<tool_call>` and s<0 (probe: no tool here), block that token and latch the block for the rest of the answer, so the
  host answers itself. ReLU threshold unit realised as a masked readout through the frozen head (exactly the IFEval v2
  enforcer mechanism). Probe (w,b) = one 1024-vector + bias attached to the model.
- **BREAKER** (mode=breaker, cheaper, no training): when a tool returns an EMPTY reply (`<tool_response>` immediately
  followed by `</tool_response>`), latch a block on `<tool_call>` for the rest of the answer.

Not in weights (disclosed, same as IFEval): the greedy decode loop, the hook on/off switch, the block mask / threshold.

## Probe training data (cg_collect_states.py, cg_neg_prompts.py) — OWN data only
- POSITIVES (label 1, tool fits): Team A test (track/order, own synthetic) + Team B test (interval/walk/truth, own
  synthetic), with their eval CoT prefixes. [sizes below]
- NEGATIVES (label 0, no tool): our own no-tool questions = integration team's 100+9 nocall prompts + 133 new ones we
  wrote (general why/how, definitions, advice, MC commonsense, sarcasm, adjective order, movie/sports/colored-object
  DISTRACTORS that resemble BBH no-tool genres but need no tool). Each shown zero-shot AND behind 2 random A/B CoT
  prefixes to induce spurious opens.
- Hidden state = host post-norm last hidden at the FIRST `<tool_call>` step (tools-off; writes are 0 before any
  response region, so the opener state is identical with tools on). 50/50 train/test split BY QUESTION (held-out AUROC).

## FROZEN configuration (2026-11-17, BEFORE any BBH run) — own data only, no BBH tuning
Per coordinator hostile-review: the over-calling was measured on real BBH, so nothing about the gate may be chosen on
BBH. These are fixed here and not changed afterward; real BBH is then one single look per variant.
- **Probe layer (fixed a priori, NOT swept on BBH):** the host's post-final-norm last hidden state h (d=1024) — the
  exact vector fed to the LM head at the step the host is about to emit `<tool_call>`. One linear reader w·h+b.
- **Threshold rule (own data):** θ = 2.5th percentile of the positive (tool-question) scores on the TRAIN split, i.e.
  keep ≥97.5% of legitimate calls. Folded into (w,b); gate vetoes iff w·h+b < 0.
- **Negatives (own only):** 109 integration nocall general + 133 our general/distractor (incl. movie/sports/colored/
  adjective-order/sarcasm distractors) + ~40 PUZZLE/MC/word-problem items we wrote (riddles, analogies, odd-one-out,
  commonsense MC, syllogisms, reading-comp MC, non-numeric patterns) that deliberately AVOID the 5 A/B tool genres, so
  the probe cannot learn "puzzle-format ⇒ call". Each shown zero-shot AND behind 2 random A/B CoT prefixes.
- **Positives (own only):** Team A test (track/order) + Team B test (interval/walk/truth) synthetic questions, eval CoT
  prefixes. 50/50 train/test split by question md5 → AUROC on held-out own questions.
- **Independent non-BBH check:** ARC-Easy (570 val) in a 3-shot CoT format; never used for training.
- **Untouched:** A_oos2, B_oos, C_ood, and all 27 BBH tasks are never seen by the probe.

### Collection 2 (tools ON, 1329 items) and probe v0 — own data
- Opened `<tool_call>` within 120 tokens: positives 271/480 (A/order3 50/50, track3 33/50, order5 8/50, **track5/track7/
  order7 0/50**: long setup text before the call), B 180/180. Negatives only 46/849: general 15/726 (2%), puzzle 31/123 (25%).
- Probe v0 (2.5th-pct rule): AUROC train 1.000 / held-out test 1.000 (183 states: 159 pos, 24 neg); test pos-pass 0.962,
  neg-veto 1.000. Classes separated: folded score pos min −0.6 / 5th pct 0.3; neg max −3.0.
- Two weaknesses found on OWN data: (a) the probe never saw the 5/7-object A positives; (b) only 22 training negatives.

### AMENDMENT (2026-11-18 ~04:25 UTC, own data only, BEFORE the gate's BBH run; the breaker BBH run had already started
— it has no parameters)
1. Collection 3 (cg_collect_states3.py): re-collect A track5/track7/order5/order7 positives with max_new 450, and add 300
   programmatic OWN puzzle-format negatives (cg_gen_puz.py: 10 generic genres — dictionary order, longest word,
   odd-one-out, category membership MC, capital MC, nonsense-word syllogisms, spell backwards, short-story recall MC,
   rhyme MC, antonym MC; own word lists), each in 3 contexts: zero-shot, behind 1 random A/B CoT prefix, and behind a
   3-shot prefix of the SAME genre (4 held-aside exemplars) — mimicking a task-specific few-shot prompt without any BBH item.
2. Threshold rule changed from 2.5th to **0.5th percentile** of TRAIN positive scores, decided now from collection-2
   evidence (classes separated by a ~2.4 margin; the 2.5% rule falsely vetoed 3.8% of held-out tool questions, and a
   false veto kills a legit call for the whole answer). Probe v1 = collections 2+3 merged, this rule. Layer unchanged.

### Collection 3 + FINAL probe v1 (FROZEN 2026-11-18 ~04:50 UTC, before the gate touches BBH)
- Collection 3: A track5/track7/order5/order7 all 200/200 opened at max_new 450; generated puzzle negatives opened
  173/900 (0-shot 53/300, A/B-prefix 76/300, same-genre 3-shot 44/300).
- Probe v1 = collections 2+3 (682 opener states: 463 pos, 219 neg; train 296 / held-out 386), L2 logistic regression
  (λ=1e-2) on standardized post-norm h, folded into one (w ∈ R^1024, b); θ = 0.5th pct of train positives (4.239 raw).
- **Held-out own data: AUROC 1.000 (268 pos / 118 neg); pos-pass 0.996 (1 false veto, B/lies: 34/35); neg-veto 1.000**
  (incl. negatives behind A/B tool prefixes and behind same-genre 3-shot prefixes → not a prefix/format detector).
- Leave-one-genre-out (train without a generated genre, test on it): veto 1.000 for all 8 genres that produced opens
  (alpha 20, story 50, syllogism 75, rhyme 15, odd-one-out 7, reverse 4, longest 1, member 1); tightest margin =
  syllogisms (max score 0.5 vs θ 4.6).
- Caveat: AUROC 1.000 on own data says the own-data classes are easy to separate; it does not say BBH no-tool openers
  look like our negatives. BBH is the out-of-distribution test.
- File: results/probe_v1.pt. (Superseded: see amendment 2.)

### Probe v1 on own HELD-OUT sets (gate, v2 tools) — FAILED the "does not kill legit calls" check
| set | v2+tools (integration) | v2+gate v1 | call% | gate-blocked% |
|---|---|---|---|---|
| A_oos2 (n=180) | 47.8 | 46.1 (53.3/36.7/60.0/60.0/30.0/36.7) | 65.0 (was 100) | 57.2 |
| B_oos (n=150) | 70.7 | **29.3** (meeting 8.0, robot 30.0, knights 50.0) | 25.3 (was 100) | **86.0** |
| C_ood (n=180) | 97.2 | 98.9 (100/100/96.7) | 0.0 | 2.2 |
Reading: AUROC 1.000 measured separability WITHIN the A_test/B_test prompt styles; out-of-style legit calls (zero-shot,
new settings) fall below θ. The probe partly learned "this looks like the A_test/B_test prompt".

### AMENDMENT 2 (2026-11-18 ~05:15 UTC; BBH still untouched by any gate)
Disclosure: A_oos2/B_oos were looked at once (above) and that look caused this change, so their re-run with v2 is a
SECOND look, not a clean held-out number. C_ood unaffected. Change = more varied positives, all OWN data, none from
A_oos2/B_oos/C_ood: (a) the teams' own TRAINING rows with a tool call from the integration union v2 (A bbh/alt/mix/code,
B hiker/lab/rumor/lies/navigate/temporal incl. recovery rows, multi M1–M5; prompt as trained, host writes the call);
(b) 20 more A_test questions/task and 30 more B_test questions/task shown ZERO-SHOT. Same layer, same 0.5th-pct rule.
Own-data out-of-style estimate added: leave-one-STYLE-out over positive groups (cg_loso.py).

### Collection 4 + probe v2 + own-data selection → FINAL probe v3 (FROZEN 2026-11-18 ~06:25 UTC, before BBH)
- Collection 4: 495 style-diverse positives, all opened (train-union A 102, B 100, multi 30; A_test zero-shot 120,
  B_test zero-shot 90). Total opener states (coll. 2+3+4): 1124 = 905 pos + 219 neg; train 524 / held-out 600.
- Probe v2 (λ=0.01, p0.5): held-out AUROC 1.000, pos-pass 0.990, neg-veto 1.000; but leave-one-STYLE-out positive
  pass: A_test 0.996, A zero-shot 1.000, B_test 0.994, **B zero-shot 0.878, train/A 0.931, train/B 0.760, multi 0.533**.
- Own-data model selection (cg_select.py; criterion set before reading it: max LOSO positive pass s.t. leave-one-
  GENRE-out negative veto ≥ 0.98):
  | λ | rule | LOSO pos-pass (min group) | LOGO neg-veto (min) |
  |---|---|---|---|
  | 0.01 | p0.5 | 0.936 (multi 0.53) | 1.000 |
  | 0.01 | mid | 0.996 | 0.927 |
  | 0.01 | q1 | 1.000 | 0.763 |
  | **0.1** | **p0.5** | **0.962 (multi 0.67)** | **1.000** |
  | 0.1 | mid | 0.987 | 0.959 |
  | 0.1 | q1 | 0.997 | 0.895 |
  | 1.0 | p0.5 | 0.981 | 0.918 |
  | 1.0 | q1 | 0.970 | 0.950 |
  (mid = midpoint between pos 0.5th pct and neg max; q1 = neg max + ¼ gap). λ=1.0/mid row not run (cannot qualify).
- **FINAL probe v3 = λ 0.1, θ = 0.5th pct of train positives (2.417 raw), coll. 2+3+4. Held-out own data: AUROC 0.9999
  (482 pos / 118 neg), pos-pass 0.992, neg-veto 1.000.** Weak spots (held-out): multi M2_order_bool 2/4, B rumor 8/9,
  B zero-shot temporal 22/23. File results/probe_v3.pt. This is the only probe that touches BBH.
- BBH gate run split by alternate tasks to fit the time: even tasks on the A40 machine GPU1 (bs 32), odd tasks on the H100 machine GPU1
  (bs 16; batch size does not change greedy outputs except through bf16 batch numerics). Breaker on the H100 machine GPU0 (bs 16).

## Results
### Run log (collection agent, 2026-11-18 UTC)
- BBH breaker on the H100 machine GPU0 was slow (shared GPU, OOM retries): 15 tasks done by 09:34 (causal..object_counting).
  At 08:40 the A40 machine GPU1 was idle (q_duo done), so the remaining 12 tasks (penguins..word_sorting) ran there as a second
  breaker run, same script/config/bs 16 (results/bbh_breaker_b.json). The h100 breaker was stopped at 09:35 right after
  object_counting was saved in its .partial (it had just started penguins); no task is scored twice. Breaker 27 =
  bbh_breaker.json.partial (15) + bbh_breaker_b.json (12).
- ARC breaker ran twice: the A40 machine GPU1 (results/arc_breaker.json; same GPU/pod as arc_none/arc_gate, used below) and by
  q_h100b on the H100 machine (arc_breaker_h100.json: acc 70.2, call 11.8, blk 7.2). Same conclusion.
- All queues finished: q_duo.done 08:37, q_h100.done 10:26, q_h100b.done 11:52. All planned runs finished (only gap: no cg-harness none-mode BBH run, see silence check).
- Silence check script: code/cg_silence.py (CPU, saved JSONs only) -> results/silence.txt; report -> results/cg_report.txt.

### probe AUROC (held-out own data)
Final probe v3 (λ 0.1, θ = 0.5th pct of train positives, 2.417 raw): held-out own data AUROC 0.9999 (482 pos / 118 neg),
pos-pass 0.992, neg-veto 1.000; LOSO min positive pass 0.962 (multi 0.67); LOGO neg-veto 1.000 (see above). Own data only.

### silence check (gate vs none on tool sets)
Rows where the gate/breaker never blocked, compared token-for-token with the no-gate run:
| comparison | not blocked | identical |
|---|---|---|
| own held-out sets, gate vs none (the H100 machine bs16) | 396 / 510 | **396 (100%)** |
| own held-out sets, breaker vs none (the H100 machine bs16) | 416 / 510 | **416 (100%)** |
| ARC-Easy gate vs none (the A40 machine) | 352 / 400 | **352 (100%)** |
| ARC-Easy breaker vs none (the A40 machine) | 369 / 400 | **369 (100%)** |
| BBH gate-odd vs breaker, both the H100 machine bs16, neither blocked (7 tasks) | 721 | **721 (100%)** |
| BBH gate even (the A40 machine bs32) vs PUBLISHED v2+tools transcripts | 2348 | 2104 (89.6%) |
| BBH gate odd (the H100 machine bs16) vs published | 1762 | 1612 (91.5%) |
| BBH breaker vs published | 4990 | 4340 (87.0%) |
Reading: in the same harness the hooks are exactly silent when they do not fire. The 9-13% mismatch vs the published
BBH transcripts is a run/harness difference (published = run27_integ on another run; gate-even even has the same bs 32),
mostly on no-tool tasks (word_sorting, movie_recommendation, ruin_names, snarks); for gate even, 111 of the 244 differing
rows change the prediction, net +8 correct. INCOMPLETE: there is no cg-harness mode=none BBH-27 run, so the BBH silence
check is only against the published transcripts plus the cross-arm check above.

### BBH 27-task table (none/published vs gate vs breaker; 27/12/15 means + call rate)
Single look per variant, probe frozen before BBH. Baseline = published v2+tools transcripts (merged).
| arm | 27 mean | 12 tool | 15 no-tool | 15 call% | 15 no-answer% | 15 calls/out | 15 blk% | 12-tool blk% |
|---|---|---|---|---|---|---|---|---|
| v2+tools (baseline) | 48.36 | 84.33 | 19.58 | 59.2 | 35.7 | 5.2 | - | - |
| v2 + GATE (probe v3) | **54.58** | 85.63 | 29.74 | 3.8 | 7.3 | 0.0 | 58.4 | 11.7 |
| v2 + BREAKER | 53.30 | 85.00 | 27.94 | 59.5 | 11.9 | 0.7 | 35.7 | 8.9 |
Per task (baseline / gate / breaker): boolean_expressions* 80.0/84.4/81.2; causal_judgement 32.1/34.2/41.7;
date_understanding 14.4/44.0/38.0; disambiguation_qa 24.0/32.0/30.4; dyck_languages 1.2/0.4/0.8; formal_fallacies
15.6/45.6/46.0; geometric_shapes 3.2/22.0/8.4; hyperbaton 16.4/44.4/39.6; logical_deduction_five* 69.6/74.0/74.4;
logical_deduction_seven* 69.2/72.0/72.0; logical_deduction_three* 73.6/76.4/74.8; movie_recommendation 26.0/26.8/27.6;
multistep_arithmetic_two* 100/100/100; navigate* 72.8/73.6/70.0; object_counting* 57.2/57.6/58.0; penguins_in_a_table
18.5/26.0/27.4; reasoning_about_colored_objects 24.8/35.2/30.8; ruin_names 19.2/22.4/26.4; salient_translation
20.8/30.0/24.0; snarks 27.5/32.6/27.5; sports_understanding 46.8/47.2/47.2; temporal_sequences* 99.6/99.6/99.6;
tracking_five* 95.2/95.2/95.2; tracking_seven* 96.4/96.4/96.4; tracking_three* 98.4/98.4/98.4; web_of_lies* 100/100/100;
word_sorting 3.2/3.2/3.2 (* = tool task). Gate split: even 14 tasks the A40 machine bs32 (mean 54.86), odd 13 the H100 machine bs16 (54.28).
(Breaker "15 call%" counts the one spurious call that the breaker lets through before latching.)

### own held-out sets (A_oos2, B_oos, C_ood) none vs gate vs breaker
acc % (call% / blocked%); all three arms in the same harness on the H100 machine bs16. For gate, A_oos2/B_oos are a SECOND look
(amendment 2), C_ood a first look.
| set | none (v2+tools) | GATE (probe v3) | BREAKER |
|---|---|---|---|
| A_oos2 (n=180) | 47.8 (100 / 0) | 52.2 (97.8 / 36.1) | **55.0** (100 / 37.2) |
| B_oos (n=150) | 71.3 (100 / 0) | **61.3** (86.0 / 30.0) | 71.3 (100 / 18.0) |
| C_ood (n=180) | 97.2 (2.2 / 0) | 98.9 (0.0 / 2.2) | 97.2 (2.2 / 0) |
Per task none / gate / breaker: A track3 70.0/60.0/66.7, track5 50.0/50.0/53.3, track7 60.0/66.7/70.0, order3
63.3/70.0/70.0, order5 23.3/30.0/33.3, order7 20.0/36.7/36.7; B meeting 78.0/**48.0**/78.0, robot 44.0/44.0/44.0,
knights 92.0/92.0/92.0; C num_list 95.0/100/95.0, num_shop 100/100/100, bool_code 96.7/96.7/96.7.
Reading: the gate still vetoes legitimate zero-shot meeting calls (call 58%, blk 44%: -30 points on meeting); the
breaker costs nothing on own sets (B unchanged, A +7.2 because it stops empty-reply loops on order5/order7).
(probe v1 for reference: A_oos2 46.1, B_oos 29.3, C_ood 98.9.)

ARC-Easy (400 val, 3-shot CoT, never used for training):
| arm | acc | call% | blocked% | calls/answer | no-answer |
|---|---|---|---|---|---|
| none | 68.25 | 12.0 | 0 | 0.93 | 48 |
| GATE | 70.0 | 0.0 | 12.0 | 0.00 | 37 |
| BREAKER | 70.5 | 12.0 | 7.75 | 0.18 | 28 |
Gate blocks exactly the 48 rows that call under none (none got 8 of them right, gate 15); breaker latches on 31 (none
1 right, breaker 10). All non-blocked ARC rows are identical to none.

### no-answer rate per arm; 15 no-tool mean vs BOTH base 39.15 and FT-tools-off 30.01
| arm | 15 no-tool mean | no-answer % (15 no-tool) | no-answer % (12 tool) |
|---|---|---|---|
| base, no tools | 39.15 | - | - |
| v2 FT, tools off | 30.01 | 5.6 | - |
| v2 FT + tools (published) | 19.58 | 35.7 | - |
| v2 FT + tools + GATE | 29.74 | 7.3 | 1.0 |
| v2 FT + tools + BREAKER | 27.94 | 11.9 | 1.4 |
Reading: both hooks remove most of the empty-reply looping (no-answer 35.7% -> 7.3 / 11.9%) and recover the 15 no-tool
mean to the FT-tools-off level (29.74 vs 30.01), not to the base (39.15). The remaining 9.4-point gap to base is the
fine-tune's own cost on no-tool tasks, which no call gate can fix. 12 tool tasks are not hurt (85.63 / 85.00 vs 84.33).
Net 27-mean: +6.22 (gate) and +4.94 (breaker) over v2+tools 48.36.

### what is NOT in weights
Same as IFEval: the greedy decode loop, the hook on/off switch, the block mask (setting the `<tool_call>` logit to
-1e9) and the latch flag. The threshold θ is folded into the probe bias (one 1024-vector + bias on the model), and the
decision is a ReLU unit read through the frozen LM head; but the "block for the rest of the answer" latch is a Python
boolean per row. The breaker has no learned parts; its empty-reply detector (`<tool_response>` followed directly by
`</tool_response>`) is a token comparison in the loop, not in weights.
