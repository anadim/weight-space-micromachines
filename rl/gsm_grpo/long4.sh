#!/usr/bin/env bash
# Arm 4 (calc_s4 + compiled gated rethink, frozen), lr 1e-6, G8, 64 prompts, T1.0, 500 updates, max 700 new tokens.
# usage: bash long4.sh SEED [B_max]
cd "$(dirname "$0")"; export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q A40 && export FLEX_SMALL=1
[ -f env_gcr.sh ] && source env_gcr.sh
d=runs/long/a4_u500_s$1; mkdir -p $d
python train2.py --arm 4 --run_dir $d --updates 500 --prompts 64 --G 8 --T 1.0 --lr 1e-6 --dev_n 800 --dev_every 25 \
   --ckpt_every 100 --fused 1 --seed $1 --max_new 700 --S_max 912 --B_max ${2:-512} --tok_budget 8192 > $d/log 2>&1
