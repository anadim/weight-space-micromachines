#!/usr/bin/env bash
# Ablation runner (dev split B only): lines "name args"; fused one-kernel implant for every implant arm.
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q A40 && export FLEX_SMALL=1
[ -f env_gcr.sh ] && source env_gcr.sh
while read -r name rest; do
  [ -z "$name" ] && continue
  [ -f runs/abl/$name/steps.jsonl ] && grep -q DONE_RUN runs/abl/$name/steps.jsonl && continue
  rm -rf runs/abl/$name; mkdir -p runs/abl/$name
  echo "$(date) start $name $rest"
  python train2.py --run_dir runs/abl/$name --updates 64 --prompts 64 --G 8 --T 1.0 --dev_n 800 --dev_every 16 \
      --ckpt_every 64 --fused 1 $rest > runs/abl/$name/log 2>&1
  echo "$(date) done $name"
done < "$1"
