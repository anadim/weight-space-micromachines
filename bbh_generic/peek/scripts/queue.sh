#!/bin/bash
# queue.sh GPU NAME...  -- run arms sequentially on one GPU (skips arms whose output exists)
P="$(cd "$(dirname "$0")/.." && pwd)"; O=${OUT:-$P/out}; mkdir -p "$O" "$P/logs"
g=$1; shift
for n in "$@"; do
  [ -f "$O/$n.json" ] && continue
  CUDA_VISIBLE_DEVICES=$g bash "$P/scripts/run_job.sh" $n > "$P/logs/$n.log" 2>&1
done
