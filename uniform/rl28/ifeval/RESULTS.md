# rl28/ifeval RESULTS — task [28]: the IFEval 4-row table on Qwen3-1.7B-Base (next to 0.6B)

Status: FINAL (2026-11-22T03:45Z). All 6 RL arms (P_s0..2, I_s0..2) done at 300 u and evaluated. Regenerate the table with
`python3 analyze.py --out RESULTS_TABLE.md` (also writes rows.json; pure python on res/, the reduced score files). 0.6B numbers come from
../../rl24/ifeval/rows.json and, for row 4p only, read-only from ../../rl24/ifeval/res/I_s*/eval_u300.plain.score.json.gz.

## Setup (identical to the 0.6B [26] protocol except the host and the refit implant readers/writers)
- Host Qwen/Qwen3-1.7B-Base. Its vocab and chat template (thinking off) are identical to Qwen3-0.6B. Format: Qwen3 chat template with
  an empty think block, the same for all rows.
- Eval: IFEval 541, greedy, max_new 2048, seeded lm_eval IFEval checkers. Main metric is prompt-level strict; prompt loose and
  instruction-level strict are also reported. tok/q = mean generated tokens per prompt (stop token excluded).
- Implant (rows 3, 4, 2b): the same full14 compiled 14-rule circuit. Only the host-tied parts are rebuilt for d=2048:
  - force_fix re-solved from Qwen3-1.7B's own plain dev answers (420 answers; 1190 rows fixed, the same count as the repo's
    vecs_q17; sha ae7cb299...251a);
  - LM-head copy E and final-norm g are exported from the host.
  Two exports were compared on the IFEval-like DEV set only (gen dev seed 201, 415 prompts, greedy 1024):
  - none: 12.29
  - snap17 (Qwen3-1.7B chat readout): 43.61
  - **base17 (1.7B-Base readout): 50.84, chosen**
  On 0.6B the snapshot had won (41.7 vs 34.2). On the 541, snap17 would have given 34.94 for row 3 (base17: 35.12).
- RL (rows 2, 4): GRPO / Dr.GRPO, 300 updates, 64 prompts x G8, lr 1e-6, T 1.0, kl 0, training max_new 1024, seeds 0/1/2.
  - Data: the same fresh gen_data train set as 0.6B (seed 200, filtered against the 541).
  - Reward: lm_eval prompt-level strict (official checker; no degenerate shortcut, per [31](b)).
  - Row 4 trains with the implant attached; the teacher-forced trainer reproduces its writes. The u1 TF ratio was
    P 0.99994 / 1.00031 / 0.99960 and I 1.00013 / 1.00023 / 1.00009; the median over 300 u was 1.0000 for every arm.
  - The engine is rl24's fast stack with the async resume save (resume_every 10). B_max is 256, with 2 arms per GPU.

## 4-row table (Qwen3-1.7B-Base)
The CIs come from a 10000-resample bootstrap over the 541 prompts, applied to the per-item score averaged over seeds.
- Row 4p (secondary) = the row-4 hosts, i.e. trained with the implant, evaluated with the implant detached.
- Idle bit-identity at u300: the implant is attached but never fires on the prompt, and the output is token-identical to the run
  without the implant:
  - P_s0 370/370, P_s1 372/372, P_s2 370/370;
  - I_s0 366/366, I_s1 366/366, I_s2 369/369;
  - Base u0 355/355.

## IFEval 541 (541 prompts)
| row | prompt strict % [95% CI] | prompt loose % | inst strict % [95% CI] | tok/q | per seed prompt strict (n) |
|---|---|---|---|---|---|
| 1 plain host | 17.19 [14.05, 20.52] | 18.48 | 31.29 [28.31, 34.28] | 1467 | 17.19 (1) |
| 2 host + RL (no implant) | 62.97 [59.33, 66.54] | 65.99 | 73.30 [70.87, 75.75] | 390 | 61.92 / 62.48 / 64.51 (3) |
| 3 host + 14-rule implant | 35.12 [31.05, 39.19] | 37.71 | 48.92 [45.44, 52.39] | 1440 | 35.12 (1) |
| 4 host + implant + RL (implant attached) | 74.68 [71.35, 78.00] | 76.96 | 82.21 [79.91, 84.46] | 595 | 74.86 / 73.57 / 75.60 (3) |
| 2b RL'd plain host + implant attached at eval | 72.34 [69.01, 75.72] | 75.11 | 80.42 [78.12, 82.68] | 434 | 71.90 / 71.53 / 73.57 (3) |
| 4p implant-RL'd host, implant detached at eval | 59.58 [55.82, 63.34] | 61.18 | 70.14 [67.46, 72.76] | 579 | 59.33 / 59.15 / 60.26 (3) |

### paired deltas (prompt strict, pp, 95% item bootstrap CI)
- 2 host + RL (no implant)  minus  1 plain host: +45.78 [+41.34, +50.28]
- 3 host + 14-rule implant  minus  1 plain host: +17.93 [+14.60, +21.44]
- 4 host + implant + RL (implant attached)  minus  1 plain host: +57.49 [+53.05, +61.80]
- 4 host + implant + RL (implant attached)  minus  3 host + 14-rule implant: +39.56 [+35.06, +44.05]
- 4 host + implant + RL (implant attached)  minus  2 host + RL (no implant): +11.71 [+8.19, +15.16]
- 2b RL'd plain host + implant attached at eval  minus  2 host + RL (no implant): +9.37 [+7.02, +11.77]
- 2b RL'd plain host + implant attached at eval  minus  4 host + implant + RL (implant attached): -2.34 [-5.18, +0.43]
- 4p implant-RL'd host, implant detached at eval  minus  2 host + RL (no implant): -3.39 [-6.65, -0.18]
- 4 host + implant + RL (implant attached)  minus  4p implant-RL'd host, implant detached at eval: +15.10 [+12.32, +18.05]

## 0.6B next to 1.7B (IFEval 541 prompt strict %, mean over seeds [95% item-bootstrap CI]; per seed)

| row | Qwen3-0.6B-Base | Qwen3-1.7B-Base |
|---|---|---|
| 1 | 12.94, tok/q 1889 | 17.19, tok/q 1467 |
| 2 | 43.56 [39.86, 47.44] (43.62 / 46.77 / 40.30), tok/q 573 | 62.97 [59.33, 66.54] (61.92 / 62.48 / 64.51), tok/q 390 |
| 3 | 34.75, tok/q 1388 | 35.12, tok/q 1440 |
| 4 | 48.37 [44.49, 52.19] (49.17 / 48.24 / 47.69), tok/q 1364 | 74.68 [71.35, 78.00] (74.86 / 73.57 / 75.60), tok/q 595 |
| 2b | 52.68 [48.92, 56.44] (54.16 / 54.71 / 49.17), tok/q 546 | 72.34 [69.01, 75.72] (71.90 / 71.53 / 73.57), tok/q 434 |
| 4p | 30.31 [26.86, 33.89] (31.79 / 30.31 / 28.84), tok/q 1582 | 59.58 [55.82, 63.34] (59.33 / 59.15 / 60.26), tok/q 579 |


## Reading
- RL helps more on 1.7B: row 2 is +45.8 pp over row 1 (0.6B: +30.6). The implant alone helps about as much as on 0.6B
  (row 3 - row 1: +17.9 vs +21.8). Row 3 is about the same absolute level on both sizes (35.1 vs 34.8).
- The implant still adds on top of RL. Row 4 - row 2 is +11.7 pp [+8.2, +15.2] (0.6B: +4.8 [+1.1, +8.6]).
- The 0.6B finding "best = RL the plain host, attach the implant at eval (2b)" does not carry over. On 1.7B, row 4 (trained
  with the implant) is nominally the best at 74.68, against 72.34 for 2b. The 2b - 4 difference is -2.3 pp [-5.2, +0.4], so
  the two are tied within the CI. On 0.6B, 2b beat 4 by +4.3 pp [+1.2, +7.3].
- Row 4p is consistent with this. On 1.7B, the host trained with the implant stays almost as good on its own: 4p is 59.6, 3.4 pp
  [0.2, 6.7] below row 2. The 0.6B host became dependent on the implant: 4p was 30.3, 13.3 pp below its row 2, with tok/q
  1582. So the larger host keeps learning the rules itself while it trains with the implant, instead of leaning on it
  (an interpretation; 4p is a secondary row).
- Seed spread is small on 1.7B: row 2 is 61.9-64.5, row 4 is 73.6-75.6, row 2b is 71.5-73.6.

## Speed and GPU use (1.7B, fast stack)
| item | value |
|---|---|
| reference HF + f14 Executor (check j23838) | greedy 64 dev: 436 tok/s; RL shape 512 rows, T 1: 722 tok/s |
| fast stack (check j23838) | greedy 64 dev: 5.1-5.3k tok/s; RL shape: 12.3-12.7k tok/s (implant on), 25.9-27.3k (implant off) |
| P arms, packed 2/GPU (P_s0, P_s1) | 13.2 / 12.8 s/u wall (roll 10.4 / 10.1, train 2.2 / 2.1, reward 0.18, dev 0.55, save block 0.07) |
| P_s2 (packed with I_s2) | 12.9 s/u wall (roll 10.2, train 2.0) |
| I arms, packed 2/GPU (I_s0, I_s1) | 29.4 / 29.3 s/u wall (roll 22.0 / 21.7, train 6.2, reward 0.24-0.29, dev 1.1) |
| I_s2 (packed with P_s2 until 02:27Z, alone after) | 21.9 s/u wall (34-38 packed early; ~12.5 alone at the end) |
| 541 eval at u300 (B_max 128) | implant 3.5-3.7k tok/s (44-93 s), plain 6.1-7.2k tok/s (18-52 s) |
| memory | 83 GB per P arm, 85.4 GB per I arm (B_max 256, cap 0.46); 160-168 GB per GPU |
| dmon SM % per job | setup j23834 84.1; check j23838 76.3; pkA j23828 (P_s0+P_s1) **97.2**; pkB j23840 (I_s0+I_s1) **97.4**; pkC j23842 (P_s2+I_s2) **89.9** |

- Resume: async save of the 20 GB state takes 98-199 s on /work for 1.7B, longer than 10 updates. So 4-10 due saves per arm
  were skipped, never waited for: save block was 0.07 s/u. A crash could therefore lose up to ~20 u. Resume states are deleted
  at DONE (rl_if.py). Kept per arm: step_000300 (bf16, 3.4 GB).
- GPU time, 1 GPU each: pkA 1h10, pkB 2h33, pkC 1h54, setup j23834 0h15, check j23838 0h07. In addition, the hung first setup
  j23827 used 0h34, 23 min of it idle (see NOTES.md).

## Exactness evidence (1.7B, check j23838: fast engine vs HF + reference Executor, base17, 64 dev prompts, greedy 1024)
- 47/64 outputs are token-identical. All 17 divergences are bf16 near-ties (top-2 gap <= 0.25 logit; 16 of them on the host
  gap). For comparison, 0.6B had 82/128 identical and 45/46 near-ties.
- Implant fuzz: 0 state / memory / write-index mismatches (1059 steps, 1024 writes). Unpadded vs padded: 0 / 22539 live-write
  mismatches. Fused Triton vocab op vs Executor: 0 / 205056 rows.
- Idle bit-identity: see above (all u0 and u300 hosts, 100%).

## Compliance
Unchanged from rl24/ifeval. The implant runs entirely in weight space inside the CUDA-graph decode step: fixed tensors; linear,
relu, gated products, attention, one-hot lookups, rmsnorm, residual/logit writes; recurrent registers reset per slot at
admission. There are no decode-loop hooks and no token swaps. Host-side Python is limited to the continuous-batching scheduler
(admission, compaction), sampling/stop handling, and the lm_eval reward checker used in training. The 1.7B port changed only
shapes (fastqwen.py: HOST_MODEL, 2048/6144) and the refit host-tied tensors (E, g, force_fix rows).
