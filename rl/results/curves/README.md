# RL dev curves used by the blog figures

Per-update dev records (`steps.jsonl`, one JSON object per logged update) copied from the runs listed in
`rl/README.md`, in the layout the figure scripts in `figures/blog/` (`rl_curves.py`, `rl_long.py`, `rl_arith.py`) read.
The same records also sit under `rl/results/gsm_grpo/<GPU>/...` and `rl/results/wiring/<GPU>/...`; this folder is
only a flat copy for the figures (the per-run `log` files are left out).

| path | run |
|---|---|
| `a1_lr1e-6_s{0,1}/`, `a2_lr1e-6_s{0,1}/`, `aR_lr1e-6_s{0,1}/` | 64-update ablation, A100 (arm 1 no implant, arm 2 calc_s4, arm R scrambled digits) |
| `long_a2/`, `a2_u500_s0.jsonl` | arm 2, 500 updates, H100 |
| `h300/*.jsonl` | 300-update comparison, H100 (arms 1 and 2, seeds 0 and 1) |
| `a4_s{0,1}.jsonl` | arm 4 (calc + compiled gated rethink) |
| `wiring/*.jsonl` | wiring arms W2 / W3e, 500 updates |
