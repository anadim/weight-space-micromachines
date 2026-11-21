#!/usr/bin/env bash
# Coordinate HP sweep on dev split B, arm 2 (calc_s4 implant attached, frozen). Fixed budget = 32768 rollouts per run.
# usage: bash sweep.sh QUEUE_NAME   (runs each line of the queue: name + extra args)
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
[ -f env_local.sh ] && source env_local.sh
BUDGET=32768
while read -r name rest; do
  [ -z "$name" ] && continue
  [ -f runs/sweep/$name/steps.jsonl ] && grep -q dev_hostonly runs/sweep/$name/steps.jsonl && continue
  P=$(echo "$rest" | grep -oP '(?<=--prompts )\d+' || echo 64); G=$(echo "$rest" | grep -oP '(?<=--G )\d+' || echo 8)
  P=${P:-64}; G=${G:-8}
  U=$((BUDGET / (P * G)))
  mkdir -p runs/sweep/$name
  echo "$(date) start $name U=$U $rest"
  python train2.py --arm 2 --run_dir runs/sweep/$name --updates $U --prompts $P --G $G --dev_n 800 --dev_every $((U / 4)) \
      --ckpt_every 100000 --fused 0 $rest > runs/sweep/$name/log 2>&1
  echo "$(date) done $name"
done < "$1"
