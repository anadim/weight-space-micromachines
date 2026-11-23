# rl24/ifeval RESULTS (task [26] 4-row table, [24b] fast-stack port, [29] ft_int_v2)

Final. The table below is written by `python3 analyze.py --out RESULTS_TABLE.md` from the score files of the u0 eval and the
6 chains (P_s0..2, I_s0..2) in res/ (the runs write out/; job labels below are labels only).

## Setup (all rows)
- Host Qwen/Qwen3-0.6B-Base (../HOST.md). Prompt format: Qwen3 chat template, thinking off (empty think block), identical for all rows.
- Eval: IFEval 541, greedy, max_new 2048, lm_eval IFEval checkers (seeded: langdetect seed 0, random.seed(key)), prompt-level strict
  (main), prompt loose, instruction-level strict; tok/q = mean generated tokens per prompt (stop token excluded).
- Implant (rows 3, 4): the full14 compiled 14-rule IFEval circuit (repo/ifeval/full14_compiled) with the frozen Qwen3-0.6B LM-head
  snapshot (sd tag `inst`), attached to Base; chosen on the IFEval-like DEV set only (host check: code/hostchk/, see ../HOST.md).
- RL (rows 2, 4): GRPO / Dr.GRPO, 300 updates, 64 prompts x G8, lr 1e-6, T 1.0, kl 0, training max_new 1024 (Dr.GRPO norm 1024),
  seeds 0/1/2. Data: fresh Team-D gen_data IFEval-like prompts (train seed 200, 18219 rows; dev seed 201, 415 rows), both filtered
  against the 541 (exact + near-duplicate). Reward = lm_eval prompt-level strict. Row 4 trains with the implant attached
  (its writes reproduced in the teacher-forced trainer forward).

## 4-row table
FINAL (3 seeds for rows 2, 4, 2b; 2026-11-21 23:58Z). Regenerate: `python3 analyze.py --out RESULTS_TABLE.md` (also writes rows.json).
The CIs come from a 10000-resample bootstrap over the 541 prompts, applied to the per-item score averaged over seeds. Seed
spread: row 2 is 40.3-46.8 and row 4 is 47.7-49.2. Idle bit-identity (implant attached, prompts where the implant never fires,
output identical to the run without the implant): P_s0 364/364, P_s1 377/377, P_s2 374/374, I_s0 338/338, I_s1 341/341,
I_s2 344/344.

## IFEval 541 (541 prompts)
| row | prompt strict % [95% CI] | prompt loose % | inst strict % [95% CI] | tok/q | per seed prompt strict (n) |
|---|---|---|---|---|---|
| 1 plain host | 12.94 [10.17, 15.90] | 13.12 | 23.98 [21.25, 26.72] | 1889 | 12.94 (1) |
| 2 host + RL (no implant) | 43.56 [39.86, 47.44] | 45.96 | 56.59 [53.73, 59.46] | 573 | 43.62 / 46.77 / 40.30 (3) |
| 3 host + 14-rule implant | 34.75 [30.68, 38.82] | 36.78 | 47.48 [44.12, 50.85] | 1388 | 34.75 (1) |
| 4 host + implant + RL (implant attached) | 48.37 [44.49, 52.19] | 52.87 | 60.35 [57.45, 63.28] | 1364 | 49.17 / 48.24 / 47.69 (3) |
| 2b RL'd plain host + implant attached at eval | 52.68 [48.92, 56.44] | 55.51 | 64.51 [61.80, 67.24] | 546 | 54.16 / 54.71 / 49.17 (3) |

### paired deltas (prompt strict, pp, 95% item bootstrap CI)
- 2 host + RL (no implant)  minus  1 plain host: +30.62 [+26.68, +34.50]
- 3 host + 14-rule implant  minus  1 plain host: +21.81 [+18.30, +25.51]
- 4 host + implant + RL (implant attached)  minus  1 plain host: +35.43 [+31.55, +39.25]
- 4 host + implant + RL (implant attached)  minus  3 host + 14-rule implant: +13.62 [+10.41, +16.94]
- 4 host + implant + RL (implant attached)  minus  2 host + RL (no implant): +4.81 [+1.11, +8.56]
- 2b RL'd plain host + implant attached at eval  minus  2 host + RL (no implant): +9.12 [+6.84, +11.52]
- 2b RL'd plain host + implant attached at eval  minus  4 host + implant + RL (implant attached): +4.31 [+1.17, +7.33]

Reading: RL raises both hosts. The implant still helps after RL (row 4 - row 2 is +4.8 pp, and the CI excludes 0). But the best
model is row 2b: RL the plain host, then attach the implant at eval. It beats row 4 by +4.3 pp [+1.2, +7.3]. So training with
the implant attached makes the host lean on it and learn less on its own.

## Speed (before -> after)
| path | config | tok/s | s/update |
|---|---|---|---|
| reference lib engine (reference RL library, HF + circuit_g) | Qwen3-0.6B, mixed task, max_new 512 | - | 57 |
| reference HF + f14 Executor, RL shape 512 rows, T 1, max_new 1024 | Base + implant (refit / snapshot) | 1134 / 1089 | - |
| fast stack, same RL shape (chk3, 2 calls) | Base + snapshot implant / no implant | 13.4k-14.7k / 27.1k-30.1k | - |
| fast stack, row 2 arm (P_s0, 300 u, measured) | Base, no implant | 33.9k rollout (median) | 4.5 median (1.6-12.0; rollout 3.8, train 0.58) |
| fast stack, row 4 arm (smk2) | Base + implant | ~19k rollout | ~17.6 (len ~470) |
| fast stack 541 eval | Base + implant / plain | 9.9k / 24k | - |

## Exactness evidence (fast stack vs reference)
- Implant fuzz (GExec vs f14 Executor on recorded streams): 0 register / memory / write-index mismatches (chk1, chk2).
- Fused Triton vocab op vs Executor: 0 mismatches over 303,808 rows (chk2).
- Greedy vs HF reference, 128 dev prompts, max_new 1024. With Base + refit implant (chk2), 86/128 are token-identical and all
  42 divergences sit at bf16 near-ties (top-2 logit gap <= 0.25 in the host or the implant readout). With Base + snapshot implant,
  the variant used here (chk3, j23723), 82/128 are identical. 45 of the 46 divergences are near-ties <= 0.25. The remaining one
  (item 101, position 1) has an implant-readout gap of 0.3125, i.e. 2.5 bf16 ulps at that logit scale. Fuzz: 0 mismatches.
  Fused vocab: 0 mismatches over 279,232 rows.
- Trainer teacher-forced ratio at u1: 1.00014 (implant arm), 0.99974 (plain arm) (smk2); P_s0 ratio median 1.0000 over 300 u.
- Idle bit-identity (implant attached but silent == host alone, token-exact on the 541 prompts on which it never fires):
  Base u0 291/291; P_s0 u300 (refit implant) 364/364; ft_int_v2 277/277; the other chains: see their summaries.

## Compliance
The implant runs entirely in weight space inside the CUDA-graph decode step (GExec: fixed tensors; linear, relu, gated products,
attention, one-hot lookups, rmsnorm, residual/logit writes; recurrent registers reset per slot at admission). There are no
decode-loop hooks and no token swaps. Host-side Python is limited to the continuous-batching scheduler (admission, compaction),
sampling and stop handling, and the reward checker (lm_eval IFEval) used in training.

## [29] ft_int_v2 (BBH call-FT host) vs Base, rows 1 and 3
Host ft_int_v2 = Qwen3-0.6B-Base + §6.1 tool-call full FT ($FT2_HOST = bbh_generic/integrate/runs/ft_int_v2, model.safetensors sha256 5e17c7c4...,
same as the §6 call-gate setup). Same format (chat, thinking off), greedy, max_new 2048, seeded lm_eval scorer, same
snapshot implant (sd_inst, no refit: it fires on 264/541 prompts vs 250 on Base). The Base rows are the u0 rows of the main table
(j23721); ft_int_v2 is from j23731. Paired item-level bootstrap over the 541 prompts (10000 resamples). Script: `python3 analyze29.py`.

| host | row | prompt-strict | prompt-loose | inst-strict | inst-loose | tok/q | fired | idle identical |
|---|---|---|---|---|---|---|---|---|
| Base | 1 plain | 12.94 | 13.12 | 23.98 | 24.34 | 1889 |  |  |
| Base | 3 + implant | 34.75 | 36.78 | 47.48 | 49.64 | 1388 | 250 | 291/291 |
| ft_int_v2 | 1 plain | 16.82 | 21.07 | 28.78 | 33.81 | 777 |  |  |
| ft_int_v2 | 3 + implant | 35.67 | 39.74 | 49.64 | 53.60 | 660 | 264 | 277/277 |

| row | delta prompt-strict (ft_int_v2 - Base) | 95% paired CI | > 2 points? |
|---|---|---|---|
| 1 plain | +3.88 | [+0.37, +7.39] | yes |
| 3 + implant | +0.92 | [-3.14, +4.99] | no |

- Row 1 (plain): ft_int_v2 is +3.9 points over Base (CI excludes 0, above the 2-point bar). Most of it comes from stopping:
  tok/q drops from 1889 to 777, and Base often repeats until the 2048 cap in this format. Loose gains more (+7.9).
- Row 3 (+ implant): +0.9 points, CI [-3.1, +5.0], below the 2-point bar. With the implant attached, the two hosts are equivalent on IFEval.
- Idle bit-identity holds on ft_int_v2 as well: 277/277 prompts on which the implant never fires are token-identical to the host alone.

## [31] GPU use and the IFEval reward (CPU checkers)
- (a) Packing: 1-GPU IFEval arms used the GPU poorly (dmon SM %, 21:36-21:46Z: I_s0 38.9, I_s1 46.3, P_s1 34.6). All remaining
  work runs packed: job pkA = P_s2 + I_s2 (two RL processes on one GPU, B_max 256, CUDA memory cap 0.35 each) + a side lane
  (degen measurement, then the row-2b re-eval of P_s0), memory cap 0.22. B_max (rollout batch slots: 256 instead of 512) only
  changes the engine's batch scheduling, not the recipe (64 prompts x 8, same sampler, same trainer). Seeds 0/1 keep B_max 512.
- (b)(i) Scoring speed: the reward already runs in a 16-process pool. On P_s0 (300 u) reward scoring took a median 0.19 / 0.14 /
  0.11 s per update for u1-50 / 51-150 / 151-300, i.e. 4.0 / 2.7 / 3.9 % of the update time (I arms: 0.15-0.3 s of ~10.5 s).
  Overlapping the scoring of batch k with the rollout of batch k+1 is not possible on-policy (rollout k+1 needs the update
  that uses batch k's rewards), and any overlap saves at most these ~3 %. The running chains were left unchanged.
- (b)(ii) Early reward 0 for degenerate outputs (ifreward.degenerate: no letters, or a short unit repeated to the end;
  code/eng/degen.py, res/pkA/degen.summary.json). Over 6830 outputs (3584 sampled RL rollouts from u0/u100/u300 checkpoints with and
  without the implant + 3246 greedy-541 outputs) the rule flags 1014 outputs. 196 of these get an official reward > 0, mostly
  from checkers that a degenerate text passes: no_comma, forbidden_words, number_sentences, startend:quotation,
  letter_frequency. So the reward would change for 2.9 % of all outputs: 0.75 % of the sampled rollouts, 5.2 % of the greedy
  outputs, and 12.4 % of the greedy u0 outputs. Unflagged outputs score the same either way.
  Decision: no shortcut. The rule does not preserve the reward, and every arm was already past u25 with the official reward.
  The shortcut would also save at most the ~3 % scoring time from (b)(i).
- (c) The real idle source was the resume save, not the reward. See the time split and fix below.
- (c) Time split and fix. Per update, other = wall - (roll + reward + train + dev); old code from steps.jsonl and the train
  wall, new code from its own wall records:
  | arm | code | wall s/u | other s/u | GPU SM % (dmon) |
  |---|---|---|---|---|
  | P_s0 (1 GPU) | old, blocking save every 2 u | 12.17 | 6.98 | - |
  | P_s1 (1 GPU) | old | 18.11 | 12.54 | 34.6-38 |
  | I_s0 (1 GPU) | old | 15.98 | 3.92 | 38.9 |
  | I_s1 (1 GPU) | old | 18.87 | 7.75 | 46-58 |
  | P_s2 (packed with I_s2) | new | 9.08 | 0.44 | 99 (packed) |
  | I_s2 (u240-300, alone on the GPU) | new | 11.38 | 0.48 | 81.5 |
  Cause: every 2 updates, resume_save() joined the previous background torch.save of the 7.2 GB state (fp32 params + AdamW;
  10-37 s per file on /work), so the loop stalled. Fix in code/eng/rl_if.py: the snapshot goes to pinned CPU buffers, a
  background thread writes the file (.tmp + rename), a save is skipped (not waited for) while the previous write still runs,
  and rl_chain.sh uses resume_every 10. On the new code, save_block_s totals 6.9 s (P_s2) and 7.3 s (I_s2) over 160-190
  updates, and no save was skipped. The recipe, rewards and RNG streams are unchanged. The whole pkA job (j23782) averaged
  87.1 % SM over 56 min. That includes startup, the evals and the I_s2-alone tail; the old packed job j23767 averaged 96.2 %.
  The remaining I_s2-alone dips (about 30 s every ~10 updates) line up with the resume snapshot plus the GIL held by
  torch.save.
