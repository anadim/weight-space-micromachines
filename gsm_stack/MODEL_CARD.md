# Model and data card: the GSM8K implant stack

## Models

No model weights are released. Every host is either a public checkpoint or produced by a documented RL command.

| host | what | how to get it | used by |
|---|---|---|---|
| base | `Qwen/Qwen3-0.6B-Base` (Apache-2.0) | downloaded from the Hugging Face Hub by the scripts | base rows, arms C / CW / Bb / Pb |
| u0 = `a2_u300_s2` | the base host after 300 GRPO updates with the calculator implant attached (arm 2, seed 2) | `rl/gsm_grpo/train2.py --arm 2 --seed 2 ...` (`REPRODUCE.md` §16.1), then copy `step_000300/` to `$WSM_DATA/ckpt_a2s2` | the ladder (rungs 0-3), arms A / A1 / A1W / B / Ar; and the implant source (`--eval_ckpt`) of every arm |
| arm hosts (`A1`, `CW_s0`, `Pb_s1`, ...) | u0 or base after 300 more GRPO updates with the stack attached (or with no implants, arm P) | `GPU=0 bash gsm_stack/engine/run_arm.sh ARM` (arm table in its header) | RL rows |

RL is not bit-reproducible across GPU types or batch sizes (bf16 numerics depend on both). A rebuilt host should give
the published numbers within seed noise (no host was rebuilt for this release); the 3-seed spreads are in `results/RL_RESULTS.md` (e.g. A1 N = 8 test sd 0.59,
CW N = 4 sd 1.3).

**What is not released:** fine-tuned / RL'd host checkpoints, the resume states (fp32 + AdamW, ~7 GB each), the
u0 host's own training records (`steps.jsonl` of `a2_u300_s2`; the seed-0 / seed-1 runs of the same recipe are in
`rl/results/gsm_grpo/H100/long/`). The two small fitted gadgets for rungs 3/4 (`vf_w.pt` 1.5 KB, `sn_w.pt` 35 KB) ship in `gsm_stack/gadgets/`.

**Shipped gadgets** (small, fitted, slow to rebuild): `gsm_push/final/verifier_probe_L18.pt` (6 KB, the layer-18
verifier probe of the serial voter), `gsm_push/calc_v2/veto_op_qwen3.pt` (6 KB). The calculator veto
`runs/veto_qwen3.pt` and the sanity-DFA tables are rebuilt by `scripts/solve_veto.py` and on first use.

## Data

| set | source | size | use |
|---|---|---|---|
| GSM8K train, split A (md5(question) % 10 in 0-5) | `openai/gsm8k` via `data/prepare_gsm.py` (+ the seeded scaled-number copy `train_scaled.jsonl`) | | RL prompts |
| split B (% 10 in {6, 7}) | same | dev = 600 items, dev2 = 300 (shuffled with `Random(0)`) | every design decision, arm selection, implant gates |
| split C (% 10 in {8, 9}) | same | "trainC", 1,499 items × 16 attempts | fitting the probes, the voters and the side-net |
| GSM8K test | `openai/gsm8k` | 1,319 | each config run once (T = 1, seed 7) |
| GSM-Hard | `reasoning-machines/gsm-hard` | 1,319 | each config run once |

Benchmarks are downloaded, not vendored. Result files keep per-item records (gold, prediction, correctness, slot
registers, token counts, hashes), never question text. The released per-arm training curves
(`results/curves/*.steps.jsonl`) hold dev evals and per-update scalars only.

**Prompt.** Zero-shot chat format, 320 tokens per attempt, T = 1 (the arms' format). The base model is much weaker in
this format (28.05 test at T = 1, 52.54 greedy) than under the Qwen3 report's protocol (OpenCompass 4-shot CoT, greedy,
512 tokens: 56.48 here, 59.59 in the report). Report base gains against the 4-shot number as well.

## Intended use and limits

Research code for studying implants in a small frozen model. The stack is tuned to GSM8K-style word problems: the
rung-3 answer-group voter, fit on GSM8K-train rollouts, loses 2.2 points on GSM-Hard, and the rung-2 thresholds were
tuned on the u0 host's dev rollouts and not retuned for the RL hosts.
