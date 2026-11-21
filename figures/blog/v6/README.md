# figures/blog/v6: the blog's v6 re-renders and the GSM8K stack figures

All scripts run on a CPU (matplotlib, numpy). The blog used the Avenir Next fonts in `src/fonts/` (not shipped: a
commercial font; the scripts fall back to the matplotlib default, so a re-render differs in glyphs but not in data).

| figure | script | data |
|---|---|---|
| `gsm8k/gsm8k_bars.*`, `gsm8k_one_by_one.*`, `gsm8k_curve.*` (GSM8K section) | `python3 gsm8k/make_gsm8k_figs.py` (`make_gsm8k_figs_v1.py`: first version) | constants in the script, from `gsm_stack/results/RL_RESULTS.md` §4 (ladder) and §8 (ablation); see `gsm_stack/COMPLIANCE.md` for the GSM-Hard "+ calculators" value |
| `diagram/serial_implants.*` (one call, many attempts) | `cd diagram && python3 make_serial_diagram.py` | none (schematic) |
| `0_implants_v2_v6.*` | `python3 src/make_implants_figure_v6.py` (SVG; PNG converted) | none |
| `bbh_bars_v2_v6.*` | `cd src/rf && python3 bbh_bars_v6.py` | constants (call-gate table, `bbh_generic/callgate/README.md`) |
| `16_bbh27_v6`, `18_ifeval_rules_v2_v6`, `23_bbh_curve27_v6`, `6_eyes_v2_v6` | `python3 src/wsm_static_v6.py` (layout-only re-runs of the `figures/blog/*.py` scripts) | as the source scripts |
| `rl2_writing_v6`, `rl5_mixed_v3_v6`, `rl6_wiring_v6`, `rl8_train_implant_v6` | v6 layout of `rl/figures/v3` (`make_rerender_v3.py`, `make_new_v3.py`; numbers `v3_*_numbers.json`) | `rl/figures/v3/data`, `rl/results/*/figdata` |
| hero / eyes / detector / blocker animations | `src/make_anim_hero_v6.py` (hero frames to `_hero/`, not tracked); the GIFs themselves are not shipped | |
