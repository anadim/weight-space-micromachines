#!/usr/bin/env bash
# run4c.sh NAME ARM SEED B_MAX : one co-trained rethink run (300 updates), any GPU
cd "$(dirname "$0")"; export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name --format=csv,noheader -i ${CUDA_VISIBLE_DEVICES:-0} | grep -q A40 && export FLEX_SMALL=1
[ -f env_gcr.sh ] && source env_gcr.sh
d=runs/long/$1; rm -rf $d; mkdir -p $d; echo "$(date) start $1" >> runs/seq4c.log
python train2.py --arm $2 --seed $3 --run_dir $d --updates 300 --prompts 64 --G 8 --T 1.0 --lr 1e-6 --dev_n 800 \
   --dev_every 25 --ckpt_every 100 --fused 1 --max_new 700 --S_max 912 --tok_budget 8192 --det_lr 1e-4 --det_T 1.0 \
   --B_max $4 > $d/log 2>&1
echo "$(date) done $1" >> runs/seq4c.log
