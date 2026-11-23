# rl28/bbh RESULTS: task [28]/[29]/[34](b), the BBH 4-row protocol on Qwen3-1.7B-Base

FINAL (05:55Z 2026-11-22). Full table incl. per-task: `table_final.md` (= w/out/table_final.md from `w/final_ana.sh`,
analysis job label j23917; the release rebuilds it from the reduced rows in `res/`). Figure data: `rows.json` ([26] schema, = rl24/uniform.json bbh panel layout). Log: `NOTES.md`.
Same protocol, harness, data, tools and RL recipe as the 0.6B study (uniform/rl24/bbh/RESULTS.md); only the host family changes.

## Setup (differences from 0.6B only)
- **Host.** Rows 1–4 use **ft17** = Qwen/Qwen3-1.7B-Base after the same §6.1 call fine-tune recipe as ft_int_v2
  (`teamA/ft.py` on the teams' own `train_union_v2`, lr 2e-5, 6 ep, tok_budget 8000, seed 0; 100 s on one B200; no BBH test
  data). Path `$RL28_DATA/bbh/ft17` (weights not in the release; rebuild with `setup.sh`), sha in `w/out/ft17_sha.txt` (not shipped).
  **The call-FT is a whole-host weight change shared by all rows 1–4** (not a frozen add-on; counted as part of the implant
  install). Row 0 (plain 1.7B-Base) and row 2' (RL from plain 1.7B-Base) are references.
- **Implants (rows 3, 4).** The same 8 compiled BBH tools as 0.6B (integ.TOOLS_V2: track, order, interval, turtle, truth, expr,
  calc, bool). Host-fitted parts (readers from the host embedding, write directions on the final norm + LM head) are solved at
  attach time on the 1.7B host; calc veto = `veto_q17.pt` (`results/gsm_bigger/veto_q17.pt`) (1.7B-Base, same solve_veto.py). Nothing new was climbed.
  Call gate refit on ft17 with the original callgate recipe (`cg/job_refit.sh`), teams' own data only: `probe_ft17.pt`,
  AUROC train/test 1.0000/1.0000 (n 441/522), test pos-pass 0.996, neg-veto 1.000; re-derivation gives an identical probe.
  Python gate vs compiled gate on the teams' held-out sets A_oos2/B_oos/C_ood: 0 latch / 0 token mismatches over 61,774 steps.
- **RL (rows 2, 2', 4).** Identical recipe: GRPO, Dr.GRPO, kl 0, clip 0.2, 300 u × 64 prompts × G 8, lr 1e-6 (warmup 5),
  T 1.0, max_new 600, seeds 0/1/2; train = teams' generators for the 12 BBH tool families at seed 200 (same file as 0.6B),
  dev = seed 400 (408 items). Tools stay attached during RL in row 4 (teacher-forced tool writes in the trainer, as on 0.6B).
- **Eval.** BBH-27 harness of `bbh_generic/callgate` (cg_gate_gen; the original driver job_cg.sh is not in the release) in the fast engine: 3-shot CoT, greedy, max_new 600, '\n\nQ' stop,
  6,511 items, mean over the 27 task accuracies; tool12 = the 12 algorithmic tasks. CIs: paired item bootstrap (10,000,
  stratified by task), multi-seed rows averaged per item over seeds.

## Main table (BBH-27 test, 6,511 items; greedy; mean over seeds; CI = paired item bootstrap)
| row | config | seeds | BBH-27 mean [95% CI] | 12 tool tasks [95% CI] | 15 other | tok/q | per-seed mean27 | 0.6B mean27 |
|---|---|---|---|---|---|---|---|---|
| 0 | plain Qwen3-1.7B-Base (reference) | – | 51.74 [50.59, 52.85] | 55.37 [53.73, 56.97] | 48.84 | 206 | 51.74 | 41.19 |
| **1** | host = ft17 (Base + call-FT), tools off | – | 42.26 [41.16, 43.37] | 41.90 [40.40, 43.43] | 42.54 | 167 | 42.26 | 32.59 |
| **2** | host + RL, tools off (F17) | 3 | 51.58 [50.60, 52.54] | 60.88 [59.66, 62.13] | 44.14 | 152 | 51.21, 51.84, 51.68 | 42.27 |
| **3** | host + 8 tools + refit gate, no RL | – | 66.33 [65.39, 67.28] | 95.73 [95.03, 96.43] | 42.82 | 161 | 66.33 | 54.99 |
| **4** | host + 8 tools + gate + RL (I17) | 3 | **68.29 [67.48, 69.12]** | **99.29 [99.02, 99.53]** | 43.49 | 158 | 68.29, 68.25, 68.34 | 60.24 |
| 2' | plain 1.7B-Base + RL (P17) | 3 | 66.95 [66.06, 67.84] | 86.17 [85.26, 87.07] | 51.58 | 195 | 67.20, 67.33, 66.33 | 53.63 |
| ctl | row-2 weights + 8 tools + gate attached at eval | 3 | 67.08 [66.26, 67.92] | 95.82 [95.22, 96.40] | 44.09 | 152 | 66.72, 67.41, 67.11 | 57.62 |
| ctl | row-2 weights + 8 tools, no gate, at eval | 3 | 66.99 [66.17, 67.82] | 95.92 [95.33, 96.49] | 43.85 | 153 | 66.49, 67.37, 67.12 | 57.37 |
| ctl | host + 8 tools, no gate | – | 66.04 [65.11, 66.97] | 96.23 [95.57, 96.87] | 41.88 | 164 | 66.04 | 52.66 |

| paired difference | BBH-27 mean [95% CI] | 12 tool tasks | 15 other | 0.6B mean27 |
|---|---|---|---|---|
| 2 − 1 (RL, no implants) | +9.32 [+8.32, +10.32] | +18.98 [+17.33, +20.61] | +1.60 [+0.38, +2.82] | +9.68 |
| 3 − 1 (implants, no RL) | +24.08 [+23.29, +24.86] | +53.83 [+52.17, +55.47] | +0.27 [−0.30, +0.86] | +22.40 |
| 4 − 2 (implants, both with RL) | +16.71 [+15.96, +17.48] | +38.41 [+37.16, +39.66] | −0.64 [−1.56, +0.30] | +17.97 |
| 4 − 3 (RL, both with implants) | +1.96 [+1.25, +2.65] | +3.56 [+2.89, +4.26] | +0.68 [−0.45, +1.81] | +5.25 |
| 4 − ctl (RL with implants on vs RL plain + implants attached afterwards) | +1.21 [+0.65, +1.79] | +3.47 [+2.93, +4.01] | −0.60 [−1.51, +0.34] | +2.62 |
| 1 − 0 (the call-FT itself, tools off) | −9.49 [−10.89, −8.04] | −13.47 [−15.57, −11.27] | −6.30 [−8.18, −4.40] | −8.61 |
| 2' − 0 (RL from plain Base) | +15.21 [+14.13, +16.32] | +30.80 [+29.26, +32.41] | +2.74 [+1.28, +4.21] | +12.44 |
| 4 − 2' | +1.34 [+0.41, +2.28] | +13.12 [+12.21, +14.03] | −8.09 [−9.59, −6.57] | +6.61 |

**Reading.**
- **Same ordering as 0.6B: 4 > 3 > 2 > 1, and 4 > 2'.** Row 4 reaches 99.3 on the 12 tool tasks (0.6B: 97.6). The implants
  carry the tool-task gain (3 − 1 = +24.1 mean27 / +53.8 tool12); RL on top adds +2.0 / +3.6 (smaller than on 0.6B, +5.3 /
  +10.8, because row 3 is already at 95.7 tool12 on 1.7B: little headroom left).
- **RL with implants on beats RL plain + implants attached afterwards** by +1.21 [+0.65, +1.79] mean27 (+3.5 tool12).
- **The call-FT costs −9.5 without tools** (−6.3 on the 15 other tasks); without tools the FT host emits `<tool_call>` in
  ~43% of rows. RL from that host with tools off (row 2) does not repair tracking_shuffled (12–32), as on 0.6B.
- **RL from plain Base (row 2') is the strong no-implant baseline on 1.7B: 66.95**, i.e. almost the level of row 3. Row 4
  beats it by only +1.34 [+0.41, +2.28] overall: +13.1 on the 12 tool tasks, but −8.1 on the 15 other tasks, all of it
  inherited from the call-FT (1 − 0 on the other 15 = −6.3; 2' keeps Base's other-15 level and adds +2.7).
- **RL does not transfer to the 15 non-tool tasks** in the FT-host arms (−0.6 to +1.6), as expected from training on the 12
  tool families only.
- **Dev curves** (408 look-alike dev items, greedy), u0 → u300: I17 90.0 → 98.8 / 98.8 / 98.0; F17 39.2 → 60.8 / 58.1 / 57.6;
  P17 46.3 → 80.4 / 78.4 / 78.4 (0.6B: I0 85.5 → 96–97, F0 38.2 → 52–58, P0 36.3 → 65–68).

## [32](e) per-task check: tasks where row 4 < row 1 by > 3 points
`w/pertask.py` (paired item bootstrap per task, row 4 averaged over 3 seeds). Full per-task table in `table_final.md`.
Result: **2 tasks, both non-tool; none of the 12 tool tasks** (tool tasks: 4 − 1 between +11.2 and +86.8, all CIs > 0).
| task | 1 host | 2 +RL | 3 +tools | 4 +tools+RL (seeds) | 4 − 1 [95% CI] | 4 − 2 | 4 − 3 |
|---|---|---|---|---|---|---|---|
| geometric_shapes | 28.0 | 27.9 | 28.4 | 21.9 (21.2, 23.2, 21.2) | −6.1 [−9.5, −3.1] | −6.0 [−9.1, −3.2] | −6.5 [−9.9, −3.3] |
| formal_fallacies | 48.8 | 48.3 | 46.4 | 45.1 (45.2, 43.2, 46.8) | −3.7 [−8.0, +0.5] | −3.2 [−6.8, +0.4] | −1.3 [−5.3, +2.7] |
- **geometric_shapes** (SVG-path multiple choice, ~7–11 options; Base 35.2, every FT-host row 26–29): in all 3 I17 evals
  **no tool writes, no gate latch and no `<tool_call>`** on any of the 250 items, so the implants are idle there and (by idle
  bit-identity) the outputs are those of the I17 RL weights alone. The drop is weight drift of the RL run with tools on, on a
  task outside the RL training families; the attach-after-RL controls (F17 weights + tools) stay at 26–29. Consistent across
  seeds (21.2/23.2/21.2). Not caused by the tools.
- **formal_fallacies** (binary valid/invalid, near chance for every row; Base 52.0): CI includes 0. Tools write on 23–31
  items per seed (accuracy on those 36–54%) and the gate latches on ~80% of rows; on the no-write items row 4 is also
  44–46 vs 48.8, so it is within the noise of a chance-level binary task (F17 seeds alone span 45.6–50.4).
- (0.6B, for comparison: formal_fallacies 43.2 → 38.4, geometric_shapes 19.6 → 20.4.)

## Speed and compute (one B200 node; several processes packed per GPU, [31])
| arm | s/update (median over 300) | packing |
|---|---|---|
| P17 (Base, no tools) | 19.1–19.4 / 15.9 | 2 RL procs per GPU / RL + F17_s2 |
| F17 (ft17, tools off) | 8.3–13.1 | 2 RL procs (+ eval worker) per GPU |
| I17 (ft17 + 8 tools + gate) | 43.4 (s0, s1) / 15.7 (s2) | 2 RL procs at B_max 64 / 1 RL at B_max 128 + 2 eval workers |
- Mean SM busy per pack job (dmon): 87.7–98.9% (P17a 98.4, F17a 67.4 before the save fix, S2a 87.7, I17a 95.6, E1 95.5,
  E2 97.2, I17b 92.7, setup2 97.4). The F17a stall (synchronous resume save, ~97 s every 10 updates) was fixed in
  `w/rl_bbh.py` (pinned buffers, background write, skip-if-in-flight; I/O only, update math unchanged) from E1 on.
- One BBH-27 eval of a 1.7B config: ~8–30 min depending on packing (I17_s2 alone 8 min).
- u1 teacher-forced ratio of every run: 0.9998–1.0002.

## Exactness evidence (1.7B engine; 128 dev rows unless stated)
- Greedy vs HF: base17 83/128 identical (divergences = bf16 near-ties, gap ≤ 0.25; acc 43.75 vs 41.41); ft17 + tools 124/128
  (acc 89.06 vs 88.28), + gate 125/128. Continuous batching (32 slots) vs all at once: 128/128 identical everywhere.
- TF ratio 0.99990 (base17) / 0.999995 (ft17 + tools); tool write flags TF vs decode 0 mismatches / 1,515 writes; tool fuzz
  0 / 57,312 steps (engine and trainer).
- Idle bit-identity on 408 dev rows: tools and tools + gate vs detached: idle rows 6/6 identical, acting rows 402/402
  identical up to the first write/latch; 0 violations.
- Gradient vs HF fp32: within bf16 noise (base17 cos 0.961 vs HF bf16 0.966).

## Compliance
- All implants live in weight space (compiled ReLU circuits on the residual stream; the gate = fixed linear probe + big-M
  ReLU + a 1-float latch register carried like KV cache, reset per sequence; identical to the Python gate).
- Host-side Python handles only sampling, the token budget (max_new 600), stopping (eos, '\n\nQ'), and continuous-batching
  admission/compaction incl. per-slot reset of tool and latch registers. No per-token Python decision about tools or calls.
- **The call-FT is a whole-host weight change** (§6.1 recipe, teams' own data), disclosed and shared by rows 1–4.

## Jobs (labels only; all on one B200 node)
setup j23812 (ft17 + gate refit), setup2 j23836 (rows 0, 1, 3, ctl, checks), P17a j23813 (P17_s0/s1), F17a j23847
(F17_s0/s1), S2a j23853 (P17_s2, F17_s2, P17_s0/s1 evals), I17a j23848 (I17_s0/s1 u0–~285), E1 j23875 (I17_s2; F17_s0/s1
u300 evals + _t1/_t1g ctl), E2 j23886 (P17_s2, F17_s2 evals + ctl), I17b j23908 (I17_s0/s1 resume → u300 + evals),
I17c j23914 (I17_s2 eval), analysis j23917. Checkpoints kept: runs/{P17,F17,I17}_s{0,1,2}/u300.pt (3.4 GB bf16 each), ft17,
cg/probe_ft17.pt; all resume states deleted at DONE.
