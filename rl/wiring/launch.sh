#!/bin/bash
# Dynamic-gated launcher. Usage: launch.sh <gpu_index> <max_parallel> <need_free_MiB> <extra_env> <runlist_file>
# runlist lines: "<arm> <seed> [extra args]". Starts a run only when (#my running < max_parallel) and GPU free >= need,
# then waits SETTLE seconds so the new run reaches its steady memory before considering the next. Never kills anything.
GPU=$1; PAR=$2; NEED=$3; EXTRA_ENV="$4"; LIST=$5
SETTLE=${SETTLE:-300}
COMMON="--updates 300 --dev_every 25 --dev_extra 64 --dev_n 800 --B_max 128 --tok_budget 4096 --ckpt_every 100 --eval_all 1"
LOG=runs/launch_gpu${GPU}.log
mkdir -p runs
echo "=== launcher gpu$GPU start $(date) PAR=$PAR NEED=$NEED ===" >> $LOG
running () { pgrep -f "^python train_wiring.py.*--gpu_tag g$GPU" | wc -l; }
freemem () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU | head -1; }
while read -r ARM SEED REST; do
  [ -z "$ARM" ] && continue
  case "$ARM" in \#*) continue;; esac
  TAG="${ARM}_s${SEED}"; [ -n "$REST" ] && TAG="${TAG}_$(echo $REST | tr -cd 'a-zA-Z0-9')"
  DIR=runs/L300_$TAG
  if [ -f "$DIR/step_000300/DONE" ]; then echo "skip $TAG" >> $LOG; continue; fi
  while true; do
    R=$(running); F=$(freemem)
    if [ "$R" -lt "$PAR" ] && [ "$F" -ge "$NEED" ]; then break; fi
    sleep 60
  done
  echo "$(date +%H:%M:%S) launch $TAG (running=$R free=$F)" >> $LOG
  mkdir -p $DIR
  env $EXTRA_ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU \
    nohup python train_wiring.py --arm $ARM --seed $SEED --run_dir $DIR $COMMON $REST --gpu_tag g$GPU \
    > $DIR.out 2>&1 < /dev/null &
  sleep $SETTLE
done < $LIST
echo "=== launcher gpu$GPU all launched $(date) ===" >> $LOG
