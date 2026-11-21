# BLOG_CONTROLS (the stack lane) — task [10](2), [11], [12]: controls, red-team experiments, compliant reruns

> Release note: working notes as written during the runs (2026-11-20/21), with cluster paths mapped to this
> repository (`gsm_stack/...`) and host names replaced by GPU types. "task [n]" refers to the item numbers of the
> original task list; `j23xxx` are job-queue ids (labels only). Labeling slips found at release are listed in
> `gsm_stack/COMPLIANCE.md` (last section).

Rule for every number: weight-space implants, one generation call, nothing programmatic beyond the decode loop
(sampling, token budget, stop rule). Deliberately programmatic baselines are labeled **(programmatic baseline)**.
CIs: 95% bootstrap (paired vs the named reference when items match). RL-stack results: RL_RESULTS.md.

Status (2026-11-21 13:55Z): everything below is DONE, except three runs still in progress: m1m (masked adapters,
mixed task), LIB/LIBscr s1, and BBH-27 with no call gate on the same rebuilt host (lg_cg_bbhnone a/b/c, ETA ~17:30Z).
Owners are in brackets. Tables: bc/gsmevals/out/tables.md, bc/latgate/tables_{lw,cg}.md, bc/rlctl/out/ANALYSIS.md.

## [12] Priority 1 — make the GSM8K numbers compliant

### Item 2: S3-serial all-weights (no position reset), test once — DONE (= task [8])
Host a2_u300_s2, chain2 + calc_s4 + compiled sanity + weight-space forget + compiled register / probe voter / final
writer, T = 1, seed 7, 320 tokens per attempt.

| config | GSM8K test [CI] | tok/q | GSM-Hard [CI] |
|---|---|---|---|
| S3-serial all-weights N=4 | 72.02 [69.52, 74.45] | 586 | 55.65 [53.07, 58.30] |
| S3-serial all-weights N=8 | 71.87 [69.37, 74.15] | 1094 | 57.54 [54.89, 60.12] |
| *reference, NOT compliant: N=8 + programmatic position reset* | *80.97 [78.77, 83.09]* | *1274* | *62.09 [59.44, 64.67]* |
| item 2b: S3-serial all-weights on the RL-stack host (A1 u300), N=4 | 79.23 [77.03, 81.35] | 614 | 62.09 [59.51, 64.75] |
| item 2b: same, N=8 | **79.76 [77.56, 81.88]** | 1223 | **63.15 [60.50, 65.81]** |

Without the reset, N tries add nothing at u0 (later tries copy earlier ones: attempt accuracy 72.9 → 22.7 by try 8); the
80.4 of the old S3 number needed the reset. RL with the stack attached (A1) recovers it in weights (+7.88 [5.91, 9.86]).

### Item 1: S2F all-weights (chain2 + compiled sanity + forget + compiled missing-answer END trigger) — DONE [the stack lane]
- Compiled trigger vs the old post-sampling swap: greedy token-identical on 600/600 dev and 300/300 dev2.
- Test, T=1 seed 7:
  - u0 **72.63** [70.13, 75.06] (hook 72.25);
  - B u300 **74.22** [71.80, 76.42] (hook 73.84).
- GSM-Hard: u0 56.03; B u300 57.47.
- All Δ vs hook are within ±0.9, n.s. Details: RL_RESULTS.md §6.

## [12] Priority 2
### Item 4: latent write, forward-only (4 writer-fit seeds) — DONE [bc/latgate]
Setup: Qwen3-0.6B-Base, frozen; n=300 per cell; layers L0–L14. 4 tasks: muladd mid/hard, mulsub mid, word problem
(pens lost). On muladd mid the writer was fit with seeds 1, 2, 3 and 11 (the original's seed). Table:
bc/latgate/tables_lw.md.
- **The original reproduces.** Plain 14.3 (published 15.0). The non-causal write (`nc`, rewrites the operand positions)
  scores 98.3 / 98.0 / 96.0 at L0 / L8 / L12, equal to the published numbers on every task.
- **Forward-only, original ridge recipe: no gain.** On muladd mid, writing at the closing token (`cl`) scores
  14.3 → 0.3 (L0 → L12; 4-seed sd ≤ 1.0). Writing at the model's own ` =` (`eq`) scores 7–8, which is the echo-format
  plain baseline (7.3). The other tasks show the same pattern.
- **Forward-only, writer trained on the answer loss (best case for a one-position write).** On muladd mid it scores
  51.8 ± 1.2 (L0) to 59.3 ± 2.6 (L14) over the 4 seeds. That is +37 to +52 over plain, but **−37 to −53 vs `nc`** at every
  layer where `nc` works (L0–L12). The other tasks agree:
  - mulsub mid: 48–69 vs `nc` 84–100;
  - word problem: 38–58 vs `nc` 89–98;
  - muladd hard: 15–54 vs `nc` 95.
- **Value use is partial.**
  - A writer that always writes the mean value scores 10–23.
  - When a wrong P′ is written, the answer follows it in ≤ 6% of rows on the mid tasks (≤ 16% on hard L14).
  - So the trained write carries a value-like signal, but the model does not read it as "P = …".
- Circuits fire exactly: 0 position or P mismatches vs the Python parse in every cell.

Reading: **negative for the forward-only claim.** A causal one-position latent write recovers at most about half of what
the non-causal (prefill-rewrite) write gives, even with a writer trained end to end. The blog should say that latent
arithmetic needs the non-causal path.
### Item 5: call gate compiled into weights, BBH-27 once — gate DONE; same-host no-gate reference running [bc/latgate]
**The compiled gate equals the Python gate.**
- Own held-out sets, full (510 rows): output text identical in 510/510, 201 of them gated. Accuracy 59.4 = 59.4.
- On BBH-27 (6,511 rows, 854,284 decode steps), with the Python gate replayed on the same logits: 0 latch mismatches and
  0 token mismatches.
- Smallest probe score at a call step: 1.3e-4. Small, but no step flips.

**BBH-27, one look** (bs 32, 600 tokens, 3-shot CoT):

| arm | 27 mean | 12 tool tasks | 15 no-tool tasks | call % on the 15 |
|---|---:|---:|---:|---:|
| published: v2 FT + tools | 48.36 | 84.33 | 19.58 | 59.2 |
| published: + Python call gate | 54.58 | 85.63 | 29.74 | 3.8 |
| rebuilt host + compiled call gate (B200) | 45.38 | 64.93 | **29.74** | 0.5 |

- On the 15 no-tool tasks the compiled gate reproduces the published gate exactly (29.74, call rate 0.5%).
- The 12 tool tasks are 20 points lower. Almost all of the gap is the three tracking_shuffled_objects tasks:
  17.6 / 16.0 / 40.4 vs 95.2 / 96.4 / 98.4 published. Logical deduction is −7 to −19.
- The gate replays the Python gate token for token, so this gap comes from the **rebuilt host**, not from the gate. The
  rebuilt host is a fine-tuned 0.6B that is not bit-identical; sha256s in cg/out/setup.txt.
- The same-host no-gate BBH-27 runs (lg_cg_bbhnone a/b/c, j23543–5, started 13:30Z, about 4 h) will give the paired
  gate effect on this host. Until then, quote only "compiled = Python gate (0 mismatches)" and the no-tool-task row.

## [12] Priority 3 / [11](b)
### Mixed-task RL: fair plain-RL arm from a worked-solution fine-tune; m1 with masked adapters; plain RL seeds 1-2 — aNL, m3 DONE; m1m running [bc/rlctl]
### Broken calculator arm R to u300 ×3; W3d s2; LIBscr s1 (+ LIB s1) — aR ×3, W3d s2 DONE; LIB/LIBscr s1 running [bc/rlctl]
Smoke u0 dev, new vs original:
- m1m: 44.0 vs 44.5;
- LIB: 48.75 vs 50.1;
- aR: 6.75 vs 7.0;
- W3d: 51.0 vs 49.1;
- aNL (new arm): 33.0.

The rebuilt SFT host is not bit-identical, so the m1m and m3 numbers are fresh. **(g) blocked**: the a1/a2 300-update
3-seed checkpoints are not on the B200 node, and regenerating them costs 6–9 GPU-h. Not queued.

**Results (13:40Z; bc/rlctl/out/ANALYSIS.md from `python3.11 analysis.py`). Still running: m1m_s0/s1 (b) and LIB_s1/LIBscr_s1 (d).**
- **(a) vs (c), mixed task, test 400, u150, 2 seeds each:**
  - aNL (plain RL from the worked-solution SFT host): **45.0** [40.4, 49.8]; gsm 27.3, bbh 71.5, combo 18.0, ifeval 44.6.
  - m3 (plain RL from the tagged SFT host, rebuilt): 33.9 [29.5, 38.1]. The original m3 s0 scored 36.3.
  - Paired difference aNL − m3: **+11.1 [+6.6, +15.7]**, almost all from BBH (+24.8). GSM is equal (+0.5).
  - Seed SD is 0.7 for aNL and 0.2 for m3. Dev at u150: aNL 40.5 / 41.0, m3 31.0 / 30.8, original m1 (tools) 65.5.
  - Reading: the fairer plain-RL arm is about 11 points stronger than m3. It is still about 23 points below the
    original m1 tool arm (68.2, one seed, original host). The paired tools − fair-plain number waits for m1m.
- **(e) broken calculator, arm R (random digits), 3 seeds, u300:**
  - Test **65.5** [63.4, 67.7] with tools on, 66.4 host-only. Hard 39.2 with tools on, 39.8 host-only. Seed SD 0.5.
  - vs a2 (working calculator, 2 seeds): −4.2 [−5.9, −2.5] test, **−16.1 [−18.2, −14.0] Hard**.
  - vs a1 (plain RL, 2 seeds): −3.7 [−5.3, −2.0] test, −3.8 [−5.4, −2.2] Hard.
  - Tools on vs host-only: −0.8 (test) and −0.6 (Hard). The host learns to route around the broken tool.
  - Dev 800 at u300: 56.6 / 54.8 / 56.2, flattening from u250.
  - Reading (3 seeds): the a2 Hard gain needs a *working* calculator. A random-digit calculator costs about 4 points vs
    plain RL. It is not a "more tokens / more RL" effect.
- **(f) W3d seed 2** (links-only distillation init, RL 300 u):
  - Dev u300 69.5 with links, 57.5 with links reset, so the links effect is **+12.0**. The originals were s0 +11.3 and
    s1 +11.5; the original s2 died at u250 (OOM, +10.6). The effect replicates.
  - Test: full 69.7, host-only 70.1, diff −0.4 [−2.0, +1.3].
  - Hard: full 54.7, host-only 43.1, diff **+11.7 [+9.4, +13.9]**.
- (g) is still blocked (see above).

## [11] red-team experiments
### (a) Python-tool baseline + tokens/s vs the in-weight calculator — DONE [bc/gsmevals]
pytool **(programmatic baseline)**: the same decode loop, but Python computes the result at the implant's trigger and
forces its characters. Host Qwen3-0.6B-Base, repo protocol (8 shots, greedy, 320 tokens), one B200. The three modes run
one after another in one job. Full tables: bc/gsmevals/out/tables.md.

| set | plain | calc implant | pytool | calc − plain | pytool − calc | identical text pytool / calc |
|---|---|---|---|---|---|---|
| test (bs 32) | 51.55 | 53.22 | 53.15 | +1.67 [+0.68, +2.65] | −0.08 [−0.23, 0.00] | 1304/1319 |
| GSM-Hard (bs 32) | 26.38 | 34.19 | 34.27 | +7.81 [+6.22, +9.48] | +0.08 [0.00, +0.23] | 1302/1319 |
| GSM-Hard 201+ (bs 32) | 25.29 | 32.17 | 32.26 | +6.88 [+5.18, +8.58] | +0.09 | 1106/1119 |
| test (bs 128) | 51.48 | 52.99 | 52.92 | +1.52 [+0.68, +2.35] | −0.08 | 1303/1319 |
| GSM-Hard (bs 128) | 26.23 | 33.97 | 34.04 | +7.73 [+6.14, +9.40] | +0.08 | 1300/1319 |

Throughput is the per-batch speed relative to plain, on the same batches (mean of plain/mode time ratios [95% CI]):

| bs | set | calc implant | pytool (programmatic) |
|---|---|---|---|
| 32 | test | 0.909 [0.862, 0.965] | 1.030 [0.975, 1.093] |
| 32 | Hard | 0.890 [0.844, 0.940] | 1.008 [0.955, 1.066] |
| 128 | test | 0.949 [0.937, 0.958] | 1.001 [0.999, 1.004] |
| 128 | Hard | 0.950 [0.940, 0.956] | 0.998 [0.998, 0.998] |

- **Accuracy:** the in-weight calculator equals the Python tool (|Δ| ≤ 0.09, text identical on ~99%). The ~15 differing
  rows per split are the known emulation gaps (a fraction digit after `.`, rare spaced negatives).
- **Speed:** in this repo engine the implant costs **5% at bs 128 and 9–11% at bs 32** vs plain. pytool costs nothing
  measurable, since the forced tokens replace decode steps one for one.
  So the blog **may not claim** that the in-weight calculator is faster than a Python tool in this engine. It may claim
  equal accuracy, no host round-trip, and a 5% overhead at batch.
- The calc_s3 implant here is the repo's unfused PyTorch implant on the repo engine. It is not the fused TK/megakernel
  path of the RL engine (engine/), where the implants are a flat ~2.2 ms/step. No fused-path matched comparison has
  been run.
- The plain/calc rows reproduce the shipped repo rows within 0.9 points; bf16 batch effects give 60–65% identical text.
### (c) / [12] item 12: mistake detector vs random trigger at the same fire rate — DONE [bc/gsmevals]
Rethink v2 (hybrid, cap 300), Qwen3-0.6B-Base, greedy, bs 32, 700 tokens. In the random arm, the fitted detector is
replaced by per-problem Bernoulli coins **(programmatic control)**, calibrated to the gated implant's firing rate.
Achieved fire rates are within 0.03 of the gated arm. Random = 3 seeds; Δ is paired [95% CI].

| split / mode | none | always | gated (implant) | random (3 seeds, mean ± sd) | gated − random | extra tok/q gated / random |
|---|---|---|---|---|---|---|
| test / plain | 51.55 | 56.03 | **55.27** | 53.73 ± 0.42 | **+1.54 [+0.45, +2.63]** | 109 / 110 |
| test / calc | 53.30 | 57.85 | **57.09** | 55.32 ± 0.36 | **+1.77 [+0.63, +2.88]** | 113 / 113 |
| Hard / plain | 26.38 | 28.73 | **28.35** | 27.29 ± 0.35 | **+1.06 [+0.13, +2.00]** | 179 / 174 |
| Hard / calc | 34.27 | 37.07 | **35.63** | 34.80 ± 0.20 | +0.83 [−0.23, +1.87] | 196 / 193 |

- The detector carries signal. At the same fire rate and the same extra tokens, it beats random firing by +1.5 to +1.8
  on test and about +1 on Hard (3 of 4 cells significant).
- The detector accounts for 40–60% of the gated gain over none (test 41% / 47%, Hard 54% / 61%). The rest comes from
  firing at all: random firing alone gives +2.2 (test plain), +2.0 (test calc), +0.9 and +0.5 (Hard).
- "Always rethink" is as good as gated (+0.4 to +1.4) but costs 26–35% more extra tokens. The gate saves tokens; it is
  not an accuracy gain over always-on.
- The none/always/gated rows reproduce the shipped repo summary within 0.8 points (other engine, bf16 batch effects).
### (d) eyes: CNN digit pasted as text — ALL arms DONE (n=8 and n=32 × eyes / text / textnc / plain, 4 seeds each) [bc/eyesctl]
- Text arm **(programmatic tool baseline)**: Python runs the same CNN and pastes `(CNN reads 7)` after each image. The
  calculator implant stays; the training recipe is identical.
- n=8, seeds 0–3:
  - eyes (in-weight): 75.3 [67.7, 82.0] (repo 78.5);
  - text (programmatic): 74.3 [67.6, 80.0], diff vs eyes −1.0 [−9.8, +7.8] (n.s.);
  - plain fine-tune, no implants: 4.6 [3.1, 6.2] (repo 4.0), diff vs eyes −70.7 [−77.7, −62.7].
- n=32, seeds 0–3 (j23401 analysis, 06:00Z; bc/eyesctl/out/results.md):
  - eyes (in-weight): 92.6 [89.6, 95.3] (repo 94.7), 83 tok/q;
  - text (programmatic): **96.0** [93.9, 97.8], diff vs eyes **+3.4 [+1.1, +6.1]**, 62 tok/q.
  - So at n=32 the programmatic text tool beats the in-weight eyes by a small, significant margin. The CNN ceiling is
    97.4% of questions with every digit read right.
- No-calculator text arm (textnc: CNN digit pasted as text, no implants at all) and plain at n=32 (j23423/j23425/j23427, analysis j23435 08:10Z):
  - textnc n=8: 67.7 [60.0, 74.7], vs eyes −7.6 [−17.4, +2.0] (n.s.); textnc n=32: 84.2 [79.7, 88.5], vs eyes **−8.4 [−13.2, −3.8]**.
  - plain n=32: 4.7 [3.3, 6.4] (repo 3.7), vs eyes −87.9.
  - So the in-weight calculator is worth +8.4 (n=32) on top of the text digits (text 96.0 vs textnc 84.2 is +11.8, unpaired).
    The eyes + calc arm beats a programmatic vision tool that has no calculator, and loses to one that keeps the calculator implant.
- The CNN was rebuilt with seeded shifts (99.09% digits; all digits right in 97.4% of questions).
- At n=8 the misses are arithmetic structure, not misread digits.
- **The in-weight eyes do not beat the text-tool baseline.** They tie at n=8 and lose by 3.4 points at n=32.
  The text arm also uses ~25% fewer tokens. The blog may claim that the eyes are a working in-weight replacement
  for a vision tool call. It may not claim they are better than it. Full table: bc/eyesctl/out/results.md
  (analyze.py).
### (e) Qwen3-0.6B-Base GSM8K with the Qwen3 report protocol — DONE [bc/gsmevals j23369]
The report (arXiv 2505.09388, Table 8) gives 59.59, 4-shot CoT. Its prompt is unpublished, so this uses OpenCompass
gsm8k_gen_17d0dc (512 tokens, greedy). Results on test (n=1319):
- plain: **56.48** [53.83, 59.14];
- calc implant: 56.63 [53.90, 59.29], calc − plain = +0.15 [−0.83, +1.14] (n.s.).

This reproduces the report within 3 points. In our zero-shot chat format the same base scores 28.05, so the low
zero-shot base is a format effect, not a weak model. Report gains over base against the 4-shot ~56 as well.

## [10](2) / [13b] / [14]6 matched controls for the serial stack (TEST once, seed 7; dev 3-seed versions: ctl_s7/s11/s13 all done, table at the end of this section)

### Same-token self-consistency (SC) vs the one-call serial stack
SC = N separate generations, each with the one-attempt S2F stack, then plain majority vote. **(programmatic baseline: N calls +
majority outside the model.)** Serial = the one-call all-weights stack (tries, forget, register, compiled voter, final writer).
Δ is paired [95% CI].

| set | host | serial N=8, one call | SC ×8 | Δ SC − serial | serial N=4 | SC ×4 | Δ |
|---|---|---|---|---|---|---|---|
| test | u0 (no RL) | 71.87 (1094 tok/q) | 79.68 (1215) | **+7.81 [5.84, 9.86]** | 72.02 (586) | 76.57 (606) | +4.55 [2.65, 6.60] |
| test | A1 u300 (RL with stack) | **79.76** (1223) | 77.56 (1204) | **−2.20 [−4.02, −0.53]** | **79.23** (614) | 75.44 (600) | −3.79 [−5.53, −2.05] |
| GSM-Hard | u0 | 57.54 (1237) | 60.05 (1395) | +2.50 [0.53, 4.47] | | | |
| GSM-Hard | A1 u300 | 63.15 (1440) | 63.38 (1360) | +0.23 [−1.67, 2.05] | | | |

- Before RL, one call with N tries is much worse than N separate calls (later tries copy earlier ones).
- After RL with the stack attached, one call **matches** 8 separate calls of the untrained host:
  A1 serial 79.76 vs SC×8 on u0 79.68, Δ +0.08 [−1.74, 1.90] at equal tokens.
- It also beats 8 separate calls of its own host on GSM8K, and ties them on GSM-Hard.
- So RL is what makes single-call serial as good as N separate calls.

### Forget mask off (prompt-only retry: later tries see the earlier ones)
| host | serial N=8 | forget off | Δ |
|---|---|---|---|
| u0 | 71.87 | 72.10 (1022 tok/q) | +0.23 [−1.90, 2.27] |
| A1 u300 | 79.76 | 71.57 (978 tok/q) | **−8.19 [−10.16, −6.29]** |

The RL'd policy relies on the weight-space forget. Without it, seeing its earlier tries costs 8 points. At u0 the
forget makes no difference, because its tries mostly copy anyway.

### Dev replication over 3 sampling seeds (ctl_s7 / ctl_s11 / ctl_s13; dev 600, T=1; qdrive j23510/j23513 for s11/s13)
Host u0 = a2_u300_s2; A1 = A1 u300. Per-seed accuracy, mean (sd over seeds). Δ is seed-pooled paired [95% CI over items] (eval/results_tools/pbootm.py).

| config | u0 | A1 u300 |
|---|---|---|
| host only (H) / + calc (C4) / + chain2 (D) | 77.17 (1.59) / 77.56 (0.35) / 77.56 (0.25) | |
| one-attempt S2F stack (B config) | 79.78 (0.67) | |
| serial N=4, one call | 82.00 (0.76) | 85.50 (0.60) |
| serial N=8, one call | 80.72 (1.17) | **86.06** (0.35) |
| SC ×4 (programmatic) | 85.17 (0.29) | 84.61 (0.63) |
| SC ×8 (programmatic) | 86.11 (0.10) | 86.00 (0.33) |
| serial N=8, forget off | 79.11 (1.25) | 81.33 (0.93) |

- SC×8 − serial N=8: u0 **+5.39 [3.56, 7.28]**; A1 −0.06 [−1.67, 1.50].
- SC×4 − serial N=4: u0 +3.17 [1.44, 4.83]; A1 −0.89 [−2.44, 0.72].
- A1 serial N=8 − u0 SC×8: −0.06 [−1.78, 1.72] at equal tokens (1163 vs 1161 tok/q).
- Forget off − on: A1 **−4.72 [−6.72, −2.83]**; u0 −1.61 [−3.50, 0.22].

The test conclusions replicate on dev over 3 seeds:
- Before RL, N separate calls beat one call with N tries by a wide margin.
- After RL with the stack attached, one call matches N separate calls of both the untrained host and its own host.
  On test, serial beats its own SC×8 by 2.2. On dev it is a tie.
- The RL'd policy needs the weight-space forget; the u0 host barely does.

Figure: figures/controls_dev.png. It shows serial / SC×8 / forget-off for u0 and A1, with seed dots (made by figures/make_figs.py).

### [14]5 Implant ablation at test (paired vs full config; full table + reading in RL_RESULTS.md §8)
After RL, the weight-space tools barely matter on GSM8K test:
- A1: both tools off −2.50 [−4.25, −0.76].
- C and B: n.s.

The stack mechanics do matter:
- forget off: A1 −8.19, C −9.17.
- compiled voter → plain majority: A1 −1.36 [−2.65, −0.15].

On GSM-Hard the calculator is essential: A1 calc off −15.16 [−17.51, −12.74]; B tools off −17.21 [−19.64, −14.71];
chain2 off −0.61 (n.s.).

SC×4 on C = 71.19 (−3.41 [−5.61, −1.21] vs C serial N=4). SC×4 on B = 77.10 (+3.26 vs B one call; 4× tokens).

### Other controls
Self-consistency at matched tokens (N independent single-attempt samples + plain majority) vs the serial voter;
prompt-only retry (forget mask off); voter off (last attempt); plain majority vs probe voter; calc-only vs full stack.

### Voter controls, offline from the same generations (GSM8K test n=1319, T=1 seed 7; eval/results_tools/offctl.py)
The "stack" column is the compliant in-weight answer. The other columns re-read the same attempt registers after
generation, so they are **(programmatic baseline, offline)**. They isolate what the compiled voter adds over simple rules.
Δ is paired vs stack [95% CI].

| host / N | stack (compiled voter) | last attempt only (voter off) | first attempt only | plain majority | probe argmax only |
|---|---|---|---|---|---|
| u0 N=4 | 72.02 | 45.79 (−26.23) | 71.49 (−0.53 [−1.97, 0.91]) | 72.71 (+0.68 [−0.61, 1.97]) | 69.75 (−2.27) |
| u0 N=8 | 71.87 | 22.74 (−49.13) | 72.93 (+1.06 [−0.53, 2.65]) | 74.00 (+2.12 [0.83, 3.41]) | 67.25 (−4.62) |
| A1 u300 N=4 | 79.23 | 70.58 (−8.64) | 73.16 (−6.07) | 77.26 (−1.97 [−3.11, −0.83]) | 78.54 (−0.68 [−1.82, 0.45]) |
| A1 u300 N=8 | 79.76 | 65.73 (−14.03) | 73.77 (−5.99) | 78.01 (−1.74 [−2.81, −0.68]) | 77.33 (−2.43 [−3.79, −1.06]) |
| *u0 N=8 + reset (non-compliant ref)* | *80.97* | *71.72* | *73.92* | *79.61 (−1.36)* | *78.47* |

Reading:
- After RL with the stack attached (A1), the compiled voter beats plain majority by ~1.8-2.0 points, and the full serial
  stack beats a single attempt by ~6.
- At u0 without the reset, attempts after the first are mostly copies. The voter then does worse than plain majority
  (−2.1 at N=8), and the try-1 answer alone matches the stack, so N tries add nothing.
- GSM-Hard runs used plain majority (no sanity filter) by design. There, voting adds +1.4 (u0) and +4.6 (A1) over the
  first attempt.
