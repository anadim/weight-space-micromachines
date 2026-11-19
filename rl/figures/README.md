# RL figures of the blog post

Every RL figure of the blog section "Can RL optimize how a model uses implants and how they're wired in?", with the
script that draws it and the data it reads. All scripts run on CPU in seconds (matplotlib; the Avenir Next font is used
when installed, otherwise matplotlib's default). Outputs are written next to the scripts.

| blog figure (alt text) | file | script | data |
|---|---|---|---|
| Test accuracy after 300 RL updates and dev accuracy during RL, with and without the calculator | `rl1_calc_after_rl.{png,svg}` | `make_blog_figs_rl.py` | test numbers: constants in the script (3 seeds; seeds 0/1 = `../results/gsm_grpo/H100/final/`, seed 2 = `../wiring/NOTEBOOK.md` "Headline seed 2"); dev curves `data/a{1,2}_s{0,1}.jsonl` (= `../results/curves/h300/`), `data/a{1,2}_s2.out` (seed-2 run logs) |
| Good calculator: fewer unit words. Broken calculator: RL stops triggering it | `rl2_writing.{png,svg}` | `make_blog_figs_rl.py` | `../interp/stats_aR.json`, `../interp/NOTEBOOK.md` Q2 (64-update runs) |
| RL learns where the implant is right; calling it blindly hurts | `rl3_competence.{png,svg}` | `make_blog_figs_rl.py` | `../results/wiring/figdata/CWinOx_s{0,1,2}.jsonl` |
| RL shuts off the broken implants; the implants carry the big numbers | `rl4_toolbox.{png,svg}` | `make_rl45.py` | `../results/library/figdata/LIB*_steps.jsonl` |
| Implants on win on every task type; RL drops the costly wrong calls | `rl5_mixed.{png,svg}` | `make_rl45.py` | `../results/library/figdata/mix_m{1,2,4}_*` , `u0_*.jsonl` |
| RL skips the calculator on small numbers; learned wiring ends about level with hand wiring | `rl6_wiring.{png,svg}` | `make_rl6.py` | `../results/wiring/figdata/{CW_s0,W3e_s0,REF_a2long}.jsonl` |
| RL strengthens a calculator link the model already uses (animation) | `anim_w3d_links.{gif,mp4}` | `../wiring/make_anim.py` | `../results/wiring/figdata/W3d_s0.jsonl` |
| (extra) competence map over training, all seeds and ablations | `fig_competence_map.{png,svg}`, `anim_competence_map.{gif,mp4}` | `../wiring/make_fig_competence.py`, `../wiring/make_anim.py` | `../results/wiring/figdata/CWinO*.jsonl`; numbers with call counts in `../results/wiring/figdata/fig_competence_map.json` |

`blog_figs_rl.json` is the sidecar `make_blog_figs_rl.py` writes: the exact plotted values (per-op pass rates with call
counts, per-seed test numbers).

```bash
cd rl/figures
python make_blog_figs_rl.py && python make_rl45.py && python make_rl6.py
python ../wiring/make_fig_competence.py && python ../wiring/make_anim.py     # animations need ffmpeg for the .mp4
```

Notes
* rl1: bars = mean of 3 seeds, whiskers = min-max. Seeds 0/1 ran on an H100, seed 2 on an A100 (bf16 kernels differ by
  GPU by ~1 pp). The seed-2 per-question test records are not in the repo (see `../README.md`, "Not released").
* rl2 uses the 64-update runs (one seed per arm, arm R seed 1 shown as a dot), not the 300-update runs of rl1.
* rl3 and the "five runs" sentence of the blog: the figure shows the three CWinOx seeds; the two lr-only runs
  (`CWinOlr_s{0,1}`) are in `fig_competence_map.json`.
* rl4/rl5 are single-seed runs (LIB s0 / LIBscr s0; mixed m1 / m2 / m4).
