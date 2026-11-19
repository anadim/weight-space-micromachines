#!/bin/bash
# Rebuild every static blog figure into figures/blog/figs/ (CPU only, matplotlib).
# Most panels hold their numbers as constants copied from the result files named in each script's docstring / README.md;
# the scripts that read files: make_figs2.calc_digits (evidence/item1), make_figs2.latent_inw (latent_write/results),
# ifeval_curve.py (ifeval_curves.json), rl_curves.py / rl_long.py / rl_arith.py (RL_CURVES, default rl/results/curves), cw_veto.py (rl/results/wiring).
set -e
cd "$(dirname "$0")"
python make_implants_figure_v2.py                 # 0_implants_v2.svg/.png (needs rsvg-convert)
python -c "import make_figs as m; m.eyes(); m.gsm()"                                     # 6_eyes, 3_gsm_stack
python -c "import make_figs2 as m; m.calc_digits(); m.finetune(); m.agent_loop(); m.latent_inw(); m.gsm_calc()"
python -c "import make_figs3 as m; m.ifeval_bars(); m.ifeval_rules(); m.bbh_bars(); m.topic_blocker()"
python bbh27.py; python bbh_curve27.py; python ifeval_curve.py; python transfer_fig.py
python rl_curves.py; python rl_long.py; python rl_arith.py; python cw_veto.py
# animations (need rsvg-convert, ffmpeg, gifsicle):
# python make_anim_hero_v2.py; python make_anim_zoom.py; python make_anim_detector.py v2; python make_anim_blocker.py
# MNIST_DIR=<dir with t10k-*-ubyte.gz> python make_anim_eyes.py
