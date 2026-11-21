#!/usr/bin/env bash
# 500-update runs, lr 1e-6, G8, 64 prompts, T1.0, dev-800 every 25 (implant-on and implant-off), ckpt at 250 and 500 (A100 machine disk is 99% full).
cd "$(dirname "$0")"; export HF_HUB_OFFLINE=1
[ -f env_local.sh ] && source env_local.sh
while read -r name rest; do
  [ -z "$name" ] && continue
  [ -f runs/long/$name/steps.jsonl ] && grep -q DONE_RUN runs/long/$name/steps.jsonl && continue
  rm -rf runs/long/$name; mkdir -p runs/long/$name
  echo "$(date) start $name $rest"
  python train2.py --run_dir runs/long/$name --updates 500 --prompts 64 --G 8 --T 1.0 --lr 1e-6 --dev_n 800 --dev_every 25 \
      --ckpt_every 250 --fused 1 $rest > runs/long/$name/log 2>&1
  echo "$(date) done $name"
done < "$1"
