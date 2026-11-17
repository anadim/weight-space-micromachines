#!/bin/bash
# usage: bash scripts/rethink_v2/queue.sh JOBS_FILE NPAR [GPU]   (each line: NAME ARGS...; output results/rethink_v2/final/NAME.json)
# Re-runs the whole list (skipping finished jobs) until every job has its json (shared GPU: jobs can OOM and are retried).
J=$1; N=$2; G=${3:-0}; PY=${PY:-python}
mkdir -p results/rethink_v2/final logs_v2/final
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for round in 1 2 3 4 5 6 7 8; do
  grep -v '^#' "$J" | grep . | xargs -P "$N" -L 1 bash -c 'n=$0; [ -f results/rethink_v2/final/$n.json ] && exit 0; CUDA_VISIBLE_DEVICES='"$G"' '"$PY"' scripts/rethink_v2/run_gsm_v2.py results/rethink_v2/final/$n.json "$@" > logs_v2/final/$n.log 2>&1'
  left=$(grep -v '^#' "$J" | grep . | awk '{print $1}' | while read n; do [ -f results/rethink_v2/final/$n.json ] || echo $n; done | wc -l)
  echo "round $round left $left"; [ "$left" = 0 ] && break; sleep 60
done
