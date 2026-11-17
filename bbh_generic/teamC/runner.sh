#!/bin/bash
# runner.sh JOBFILE : run the jobs one after another (foreground), retry a job up to 5 times if it died of CUDA OOM
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while read -r n rest; do
  [ -z "$n" ] && continue
  for try in 1 2 3 4 5; do
    python run_gen.py out/$n.json $rest > logs/$n.log 2>&1
    grep -qi "OutOfMemory\|ALLOC_FAILED\|out of memory" logs/$n.log || break
    sleep 60
  done
done < "$1"
