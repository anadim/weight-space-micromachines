#!/usr/bin/env bash
# Main arms (equal compute: same updates x prompts x G, same seeds -> same prompt batches). Fused one-kernel implant.
# usage: bash main.sh QUEUE   (lines: name args...)
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
[ -f env_local.sh ] && source env_local.sh
while read -r name rest; do
  [ -z "$name" ] && continue
  [ -f runs/main/$name/steps.jsonl ] && grep -q '"DONE_RUN"' runs/main/$name/steps.jsonl && continue
  mkdir -p runs/main/$name
  echo "$(date) start $name $rest"
  python train2.py --run_dir runs/main/$name --dev_n 800 --fused 1 $rest > runs/main/$name/log 2>&1
  echo "$(date) done $name"
done < "$1"
