#!/bin/bash
# stop my L500_* wiring runs once their u300 checkpoint (written after the u300 dev + ablation evals) exists
cd "$(dirname "$0")/runs"
while true; do
  for d in L500_*/; do d=${d%/}
    if [ -f "$d/step_000300/DONE" ]; then
      for p in $(pgrep -f "run_dir runs/$d --updates 500"); do echo "$(date +%H:%M:%S) cap300 stop $d pid $p" >> cap300.log; kill $p; done
    fi
  done
  sleep 60
done
