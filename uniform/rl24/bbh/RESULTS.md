# rl24/bbh RESULTS (task [24]/[24b] → [26] → [29])

FINAL (23:25Z 2026-11-21). Full table incl. per-task: `table_final.md` (= w/out/table_final.md from `w/final_ana.sh`,
job label j23805; the release rebuilds it from the reduced rows in `res/`); figure data: `rows.json` ([26] schema).

## Setup
- **Host.** Rows 1–4 all use **ft_int_v2** = Qwen/Qwen3-0.6B-Base after the §6.1 call fine-tune:
  - `teamA/ft.py` on the teams' own `train_union_v2` (432 rows), lr 2e-5, 6 epochs, tok_budget 8000, seed 0;
  - no BBH test data;
  - sha256 5e17c7c4…; provenance in the call-FT setup log (not in the release).

  **The call-FT changes the weights of the whole host.** It is not a frozen add-on. It is shared by rows 1–4 and counted
  as part of the implant install. Row 0 (plain Base) and row 2' (RL from plain Base) are references.
- **Implants (rows 3, 4).**
  - The 8 BBH tools (teams A/B/C, compiled integer ReLU circuits, attached as in `micromachines.machine.attach`):
    track, order, interval, turtle, truth, expr, calc, bool.
  - The call gate, refit on ft_int_v2 with the original recipe (`repo/bbh_generic/callgate`), on the teams' own data only:
    - probe at `$RL24_DATA/bbh/cg/probe_ft2.pt`;
    - held-out AUROC 0.9999;
    - compiled the same way as cg_compiled: linear probe, big-M ReLU, and a 1-float latch register per row;
    - the Python gate and the compiled gate agree exactly: 0 mismatches over 63,312 steps on the teams' own held-out sets.
- **RL (rows 2, 2', 4).**
  - GRPO with Dr.GRPO advantages (group mean, no std), kl 0, clip 0.2, 300 updates × 64 prompts × G 8, lr 1e-6 (warmup 5),
    AdamW (0.9, 0.95), grad clip 1, T 1.0, max_new 600, seeds 0/1/2.
  - Train data: the teams' generators for the 12 BBH tool families at a fresh seed 200 (11,981 items). Never test.
  - Dev: seed 400, 408 items, greedy, every 25 updates.
  - Reward: the exact answer under the team harness norms (envs.normA/B/C).
  - The tools stay attached during RL in row 4. This follows the reference library RL convention (`train_library`
    train_step):
    - a tool writes into the residual stream just before the final norm, rather than writing tokens;
    - every generated token is trained;
    - the trainer re-emits the frozen tool writes teacher-forced, so the logits at every trained position include exactly
      the write that was present at decode time;
    - tokens masked by the gate (`<tool_call>` after the latch) and the early-eos mask are renormalised in the log-prob.
- **Eval.** The BBH-27 harness of `bbh_generic/callgate` (cg_gate_gen, not-in-release driver job_cg.sh), run in the fast engine. The settings are the same for
  every row:
  - 3-shot CoT prompts, greedy, max_new 600, the '\n\nQ' stop rule;
  - 6,511 items;
  - mean over the 27 task accuracies; "tool12" is the 12 algorithmic tasks the tools target.

  CIs are paired, from 10,000 item bootstrap resamples stratified by task; for multi-seed rows the per-item accuracy is
  averaged over seeds.

## Main table (BBH-27 test, 6,511 items; greedy, max_new 600; mean over seeds; CI = paired item bootstrap)
| row | config | seeds | BBH-27 mean [95% CI] | 12 tool tasks [95% CI] | 15 other | tok/q | per-seed mean27 |
|---|---|---|---|---|---|---|---|
| 0 | plain Qwen3-0.6B-Base (reference) | – | 41.19 [40.08, 42.28] | 43.47 [41.83, 45.07] | 39.37 | 222 | 41.19 |
| **1** | host = ft_int_v2 (Base + call-FT), tools off | – | 32.59 [31.52, 33.65] | 36.57 [35.07, 38.17] | 29.40 | 135 | 32.59 |
| **2** | host + RL, tools off (F0) | 3 | 42.27 [41.41, 43.12] | 56.34 [55.23, 57.47] | 31.01 | 144 | 41.14, 43.15, 42.52 |
| **3** | host + 8 tools + refit gate, no RL | – | 54.99 [54.05, 55.92] | 86.80 [85.73, 87.87] | 29.54 | 134 | 54.99 |
| **4** | host + 8 tools + gate + RL (I0) | 3 | **60.24 [59.50, 60.97]** | **97.64 [97.21, 98.06]** | 30.32 | 140 | 60.31, 60.11, 60.29 |
| 2' | plain Base + RL (P0) | 3 | 53.63 [52.64, 54.61] | 70.44 [69.14, 71.74] | 40.18 | 201 | 53.91, 54.09, 52.90 |
| ctl | row-2 weights + 8 tools + gate attached at eval | 3 | 57.62 [56.85, 58.39] | 91.09 [90.32, 91.84] | 30.84 | 139 | 57.08, 58.37, 57.41 |
| ctl | row-2 weights + 8 tools, no gate, at eval | 3 | 57.37 [56.61, 58.15] | 91.11 [90.34, 91.87] | 30.38 | 149 | 56.77, 57.81, 57.54 |
| ctl | host + 8 tools, no gate (published rebuilt-host bbhnone 52.94) | – | 52.66 [51.70, 53.59] | 85.47 [84.33, 86.60] | 26.42 | 162 | 52.66 |

Rows 0, 1 and 3 are greedy evaluations of a fixed model, so there are no seeds.

| paired difference | BBH-27 mean [95% CI] | 12 tool tasks | 15 other |
|---|---|---|---|
| 2 − 1 (RL, no implants) | +9.68 [+8.64, +10.70] | +19.78 [+18.18, +21.32] | +1.60 [+0.20, +2.97] |
| 3 − 1 (implants, no RL) | +22.40 [+21.45, +23.37] | +50.23 [+48.53, +51.90] | +0.14 [−0.93, +1.21] |
| 4 − 2 (implants, both with RL) | +17.97 [+17.19, +18.75] | +41.30 [+40.19, +42.41] | −0.69 [−1.74, +0.36] |
| 4 − 3 (RL, both with implants) | +5.25 [+4.38, +6.11] | +10.84 [+9.79, +11.89] | +0.78 [−0.54, +2.07] |
| 4 − ctl (RL with implants on vs RL plain and implants added afterwards) | +2.62 [+1.99, +3.24] | +6.56 [+5.90, +7.23] | −0.53 [−1.51, +0.46] |
| 1 − 0 (the call-FT itself, tools off) | −8.61 [−10.01, −7.20] | −6.90 [−9.00, −4.73] | −9.97 [−11.77, −8.12] |
| 2' − 0 (RL from plain Base) | +12.44 [+11.38, +13.50] | +26.98 [+25.26, +28.70] | +0.81 [−0.55, +2.19] |
| 4 − 2' | +6.61 [+5.59, +7.63] | +27.20 [+25.89, +28.51] | −9.86 [−11.39, −8.36] |

**Reading.**
- **The implants carry most of the tool-task gain, and RL on top adds +5.3 mean27 / +10.8 tool12.** Row 4 is at
  97.6 on the 12 tool tasks. Logical deduction goes from 71–77 to 95–97, navigate from 78 to 99.5, and object_counting
  from 56 to 85 (per-task table in `table_final.md`).
- **RL with the implants attached beats RL without them and attaching them afterwards.** The gap is +2.62 [+1.99, +3.24]
  mean27 and +6.6 tool12.
- **The call-FT host is worse than plain Base without tools.** Row 1 − row 0 is −8.6, and the other 15 tasks lose
  ~10 points. Without tools it emits `<tool_call>` in ~47% of rows and gets no reply. RL from that host with tools off
  (row 2) raises tool12 to 56, but the 3 tracking_shuffled tasks stay at 10–32: RL did not unlearn the calls in 300
  updates.
- **RL from plain Base (row 2') is the better no-implant baseline, at 53.6.** Row 4 still beats it by +6.6 overall
  (+27.2 tool12). Row 4 loses −9.9 on the 15 non-tool tasks, all of it inherited from the call-FT.
- **RL does not transfer to the 15 non-tool tasks** (−0.7 to +1.6 in every arm), as expected from training only on the
  12 tool families.
- **Dev curves (408 look-alike dev items, greedy), u0 → u300:**
  - I0 85.5 → 96.3 / 95.3 / 97.1;
  - F0 38.2 → 51.5 / 56.1 / 57.6;
  - P0 36.3 → 67.9 / 67.9 / 64.7.

## Speed: reference vs fast stack (one B200, 1 GPU)
| | before (reference) | after (rl24/bbh fast stack) |
|---|---|---|
| RL s/update, Base host, no tools (64×8) | – | 6.5 s (rollout 4.6 s @ 23.7k tok/s, train 1.9 s) |
| RL s/update, ft_int_v2, tools off (64×8, F0) | – | 4.3 s (rollout 3.4 s @ 22k tok/s, train 0.9 s) |
| RL s/update, ft_int_v2 + 8 tools + gate (64×8) | 42–72 s (reference lib engine with the same tool library: reference-library run smk_LIB_s1 (not in the release) on one B200; mixed task) | 12.0 s (rollout 11.1 s @ 6.5k tok/s, train 0.9 s; 14–19 s early, when more groups have non-zero advantage) |
| greedy generation, ft_int_v2 + tools | 319 tok/s (HF harness, batch 64, dev 408 items) | 3,498 tok/s (BBH-27, 6,511 items in 302 s, B 128) |

Packing ([31]): the last 8 evals ran as 3 concurrent processes per GPU (`w/pack.sh`, `--mem_frac 0.30`, `--B_max 64`), with
mean SM busy 95.0% (j23773) and 92.4% (j23793) in `nvidia-smi dmon`. One eval alone at B 128 takes ~2.5 min and is
prefill-bound.

The reference RL engine was not ported to the BBH task, so "before" is the reference lib engine on its own task, with
the same 8 BBH tools attached, plus the HF harness used for the host check.

## Exactness evidence (fast engine vs the references)
On 64 dev rows, B200:
- **Greedy vs the HF harness.**
  - ft_int_v2 + tools: 62/64 identical texts, accuracy equal.
  - ft_int_v2 + tools + gate: 62/64 identical, accuracy 87.5 = 87.5.
  - Base: 42/64 identical. Every divergence is a bf16 near-tie, with a top-2 gap ≤ 0.375.
- **Continuous batching (32 slots) vs all rows at once:** 64/64 identical in every configuration.
- **Teacher-forced ratio** on a full 512-row RL rollout: 1.0000–1.0002 (mean |Δlogp| 8.6e-4 with tools; 0.015 for Base).
  - u1 of every run: 0.9998–1.0001.
  - Tool write flags, teacher-forced vs decode: 0 mismatches (3,204 + 3,078 writes in the checks; 0 at u1 of each I0 run).
- **Tool fuzz.** The reference Micromachine `prime_sym`/`step` was compared with the engine's batched step and with the
  trainer's `emit_tf`: 0 mismatches over 37,912, 56,792 and 30,736 steps (ft2+tools, Base+tools, ft2+tools+gate).
- **Gradient vs HF fp32 autograd:**
  - trainer cos 0.984 (Base) and 0.9994 (ft2 + tools);
  - HF bf16 vs fp32 gives 0.969 and 0.9991, so the trainer is within bf16 noise.
- **BBH-27 reproduction:** ft_int_v2 + tools with no gate scores 52.66 in the engine, against the published rebuilt-host
  `bbhnone` 52.94.

## Compliance
- **All implants live in weight space.** The tools are compiled ReLU circuits on the residual stream. The gate is a fixed
  linear probe, big-M ReLU arithmetic, and a 1-float latch register that is carried like KV cache and reset per sequence.
  It is identical to the Python gate.
- **Host-side Python** handles only:
  - sampling (greedy or T 1);
  - the token budget (max_new 600);
  - stopping (eos, and the harness '\n\nQ' stop, checked at sync points);
  - continuous-batching admission and compaction, including per-slot reset of the tool and latch registers.

  There is no per-token Python decision about tools or calls.
- **The call-FT is a whole-host weight change** (§6.1 recipe, the teams' own data). It is disclosed and shared by rows 1–4.
