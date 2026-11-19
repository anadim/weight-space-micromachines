#!/bin/bash
# Gated launcher v2. Usage: launch2.sh <gpu> <max_procs_on_gpu> <need_free_MiB> <extra_env> <list> <lib_ready_dir>
# Lines: "<arm> <seed> [args]" (train_wiring.py, 300 updates) or "!<tag> <shell command>" (raw, e.g. train2.py / evals).
# Defers while rl_wiring/HOLD exists or any library script is pending; slots = compute processes on the GPU (all owners).
GPU=$1; PAR=$2; NEED=$3; EXTRA_ENV="$4"; LIST=$5; LIBREADY=$6
SETTLE=${SETTLE:-300}
COMMON="--updates 300 --dev_every 25 --dev_extra 64 --dev_n 800 --B_max 128 --tok_budget 4096 --ckpt_every 300 --eval_all 1"
LOG=runs/launch2_gpu${GPU}.log; mkdir -p runs
echo "=== launch2 gpu$GPU start $(date -u) PAR=$PAR NEED=$NEED list=$LIST ===" >> $LOG
nproc_gpu () { nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader | grep -c . ; }
freemem () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU | head -1; }
while read -r FIRST REST; do
  [ -z "$FIRST" ] && continue; case "$FIRST" in \#*) continue;; esac
  if [[ "$FIRST" == !* ]]; then TAG=${FIRST#!}; DIR=runs/RAW_$TAG; CMD="$REST"
  else ARM=$FIRST; set -- $REST; SEED=$1; shift; R2="$@"
       TAG="${ARM}_s${SEED}"; [ -n "$R2" ] && TAG="${TAG}_$(echo $R2 | tr -cd 'a-zA-Z0-9')"; DIR=runs/L300_$TAG
       CMD="python train_wiring.py --arm $ARM --seed $SEED --run_dir $DIR $COMMON $R2 --gpu_tag g$GPU"; fi
  if [ -f "$DIR/step_000300/DONE" ] || [ -f "$DIR.started" ]; then echo "skip $TAG" >> $LOG; continue; fi
  while true; do
    P=$(nproc_gpu); F=$(freemem)
    PEND=$(ls $LIBREADY/*.sh 2>/dev/null | grep -c .)
    if [ ! -f HOLD ] && [ "$PEND" -eq 0 ] && [ "$P" -lt "$PAR" ] && [ "$F" -ge "$NEED" ]; then break; fi
    sleep 60
  done
  echo "$(date -u +%H:%M:%S) launch $TAG (procs=$P free=$F)" >> $LOG
  mkdir -p $DIR; touch $DIR.started
  env $EXTRA_ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU \
    setsid nohup bash -c "$CMD" > $DIR.out 2>&1 < /dev/null &
  sleep $SETTLE
done < $LIST
echo "=== launch2 gpu$GPU list done $(date -u) ===" >> $LOG
