#!/bin/bash
# MEM_GB=40
# LIBnoIF s0 (library without the IFEval circuit) on a A40 host A40. A40: FLEX_SMALL=1, non-persistent calculator (fused 0),
# tok_budget 2048 (memory only). ~2.7x slower than H100 -> dev every 50 (not 25), deadline guard 13:45 UTC cuts prompts/update
# at u5 if needed (never updates). Poller pins the GPU as device 0.
cd "$(dirname "$0")/.."
export WSM_INTEG_DIR=$PWD/../../bbh_generic/integrate/code RLW_DIR=$PWD/../wiring FLEX_SMALL=1
python train_library.py --arm LIB --seed 0 --scramble 0 --ifeval 0 --fused 0 --run_dir runs/LIBnoIF_s0 --updates 300 --prompts 64 --G 8 --T 1.0 \
  --lr 1e-6 --kl 0 --loss drgrpo --dev_every 50 --dev_extra 25 --dev_n 800 --B_max 128 --tok_budget 2048 --ckpt_every 300 \
  --iw_eps 0.05 --link_lr 1e-3 --link_eps 1e-8 --deadline 13:45 > runs/LIBnoIF_s0.out 2>&1
