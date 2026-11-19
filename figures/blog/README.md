# Blog figures

Scripts that draw the figures of the blog post, and the rendered files (`figs/`, `figs/anim/`). Run `bash make_blog_figs.sh`
(CPU, matplotlib 3.10; the animations also need `rsvg-convert`, `ffmpeg`, `gifsicle`). The scripts use the Avenir Next
font if `figures/blog/fonts/*.ttf` exists (not redistributed; matplotlib falls back to its default font otherwise). With the Avenir Next fonts in `fonts/`,
`make_blog_figs.sh` reproduces every committed static PNG from a fresh clone pixel-for-pixel (25_rl_curves: 32 pixels differ; checked 2026-11-19).
Most bar/line charts keep their numbers as constants in the script, copied from the result files listed below; the
source of each constant is named in the script's docstring.

| blog figure | script | numbers come from | output |
|---|---|---|---|
| The calculator implant lighting up while Qwen generates | `make_anim_hero_v2.py` (imports `make_anim.py`) | drawing (real `calc_s4` example 2595 + 3075) | `figs/anim/hero_v2.gif` |
| One model, many implants | `make_implants_figure_v2.py` | implant sizes: `results/implant_sizes.json`, component READMEs | `figs/0_implants_v2.png` |
| Inside the calculator | `make_anim_zoom.py` | drawing of `micromachines/circuits` (calc) | `figs/anim/zoom.gif` |
| Accuracy as the numbers get longer | `make_figs2.py: calc_digits()` | `evidence/item1/*.json` (reads them) | `figs/9_calc_digits.png` |
| Or fine-tune Qwen to make the calls | `make_figs2.py: finetune()` | `finetuned_phase/results/reported_numbers.json` (5 seeds) | `figs/8_finetune.png` |
| Giving Qwen eyes (animation) | `make_anim_eyes.py` | MNIST test images (`MNIST_DIR`) | `figs/anim/eyes.gif` |
| Giving Qwen eyes | `make_figs.py: eyes()` | `evidence/eyes_sweep/results/` (8/32), `finetuned_phase/results/` (4k/16k) | `figs/6_eyes.png` |
| How Claude builds an implant | `make_figs2.py: agent_loop()` | diagram | `figs/13_agent_loop.png` |
| Rule implants on IFEval | `make_figs3.py: ifeval_bars()` | IFEval held-out scoring (`ifeval/heldout/final_score.sh`; transfer runs under `ifeval/`) | `figs/17_ifeval_bars.png` |
| Each rule implant, one by one | `make_figs3.py: ifeval_rules()` | same, per rule | `figs/18_ifeval_rules.png` |
| Add one rule implant at a time | `ifeval_curve.py` | `ifeval_curves.json` (from the add-one-rule curve scores, IFEval transfer folder) | `figs/22_ifeval_curve.png` |
| Eight implants on BIG-Bench Hard | `make_figs3.py: bbh_bars()` | `bbh_generic/integrate/` (integration v2 tables) | `figs/19_bbh_bars.png` |
| All 27 BIG-Bench Hard tasks | `bbh27.py` (dot plot, the blog's; `make_figs3.bbh_tasks()` is an older bar version that writes the same file name, not run by `make_blog_figs.sh`) | `bbh_generic/integrate/` | `figs/16_bbh27.png` |
| Add one implant at a time, all 27 BBH tasks | `bbh_curve27.py` | BBH add-one-tool curve (BBH transfer folder, `curve_table.md`) | `figs/23_bbh_curve27.png` |
| Same implants, other models | `transfer_fig.py` | `results/gsm_bigger/`, IFEval and BBH transfer folders | `figs/24_transfer.png` |
| The mistake detector in action | `make_anim_detector.py v2` | `figs/anim/detector_v2_trace.json` (scores recomputed by `figs/anim/anim_scores_job_v2.py`) | `figs/anim/detector_v2.gif` |
| Two different fixes, and they compose | `make_figs.py: gsm()` | `results/rethink_v2/summary.json`, `results/rethink_v2/final_eval/` | `figs/3_gsm_stack.png` |
| The topic blocker in action | `make_anim_blocker.py` | `figs/anim/blocker_trace.json` | `figs/anim/blocker.gif` |
| Block a topic by reading the model's state | `make_figs3.py: topic_blocker()` | `topic_fence/results/q17_{math,ml}_table.txt` (same for the in-weights version, `topic_fence/inweights/`) | `figs/21_topic_blocker.png` |
| The model treats the written value as its own | `make_figs2.py: latent_inw()` | `latent_write/results/{muladd,lost}_mid_Qwen3-0.6B-Base.json` (reads them) | `figs/15_latent_inweights.png` |
| Dev accuracy on normal and big-number problems during RL, with and without the calculator | `rl_arith.py` | `RL_CURVES/h300/*.jsonl` (= `rl/results/curves/h300/`, 300-update H100 runs, 2 seeds; reads them) | `figs/28_rl_arith.png` |
| RL with and without the calculator | `rl/interp/make_fig.py` (v1, two panels) / `rl/interp/make_fig_v2.py` (right panel) | 64-update A100 dev traces (**not released**; numbers in `rl/interp/stats.json`, `stats_aR.json`) | `rl/interp/fig_rl_calc_use.png`, `figs/fig_rl_calc_use_v2.png` |
| How often the RL-trained in-wire vetoes a calculator call, by operand size | `cw_veto.py` | `rl/results/wiring/H100/L300_CW_s0_kl001evalall0/steps.jsonl` (reads it; u25-250 plotted) | `figs/27_cw_veto.png` |
| (not in the current blog) | `rl_curves.py`, `rl_long.py` | `rl/results/curves/` | `figs/25_rl_curves.png`, `figs/26_rl_long.png` |

Also kept: `make_figs2.gsm_calc()` -> `figs/14_gsm_calc.png` (calculator on GSM8K / GSM-Hard, three Qwen3 sizes, from
`results/gsm_full_implant/` and `results/gsm_bigger/`).

Not reproducible from the repo alone: `figs/anim/anim_scores_job*.py` (recompute the detector / blocker scores for the
animations on a GPU) read `gen_ids*.json`, token ids of the full GSM generations, which are not released (the released
result files keep only a SHA-256 of each generation); their outputs are already in `figs/anim/*_trace.json`.
`make_figs2.latent_results()` (figure 12, not in the blog) reads latent-write-on-BBH results that are not in this repo.
