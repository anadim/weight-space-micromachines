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
freemem () { python3 effree.py $GPU "$(pwd)"; }       # free minus not-yet-allocated library reservations
NEED_SHORT=22000
while read -r FIRST REST; do
  [ -z "$FIRST" ] && continue; case "$FIRST" in \#*) continue;; esac
  LNEED=""; if [[ "$FIRST" == NEED=* ]]; then LNEED=${FIRST#NEED=}; read -r FIRST REST <<< "$REST"; fi   # per-line need
  SHORT=0; if [[ "$FIRST" == !!* ]]; then SHORT=1; FIRST=${FIRST#!}; fi      # '!!' = short eval: may run during HOLD
  if [[ "$FIRST" == !* ]]; then TAG=${FIRST#!}; DIR=runs/RAW_$TAG; CMD="$REST"
  else ARM=$FIRST; set -- $REST; SEED=$1; shift; R2="$@"
       TAG="${ARM}_s${SEED}"; [ -n "$R2" ] && TAG="${TAG}_$(echo $R2 | tr -cd 'a-zA-Z0-9')"; DIR=runs/L300_$TAG
       CMD="python train_wiring.py --arm $ARM --seed $SEED --run_dir $DIR $COMMON $R2 --gpu_tag g$GPU"; fi
  if [ -f "$DIR/step_000300/DONE" ] || [ -f "$DIR.started" ]; then echo "skip $TAG" >> $LOG; continue; fi
  while true; do
    P=$(nproc_gpu); F=$(freemem)
    PEND=$(ls $LIBREADY/*.sh 2>/dev/null | grep -c .)
    if [ "$PEND" -gt 0 ]; then                  # a small job may still go if it cannot block any pending library script
      MAXN=0; for sc in $LIBREADY/*.sh; do m=$(grep -m1 -oE 'MEM_GB=[0-9]+' "$sc" | cut -d= -f2); m=$(( (${m:-40} + 4) * 1024 )); [ $m -gt $MAXN ] && MAXN=$m; done
      TOT=$(nvidia-smi -i $GPU --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
      NN0=$NEED; [ $SHORT -eq 1 ] && NN0=$NEED_SHORT; [ -n "$LNEED" ] && NN0=$LNEED
      [ $(( NN0 + MAXN )) -le "$TOT" ] && PEND=0
      [ "$MAXN" -gt "$TOT" ] && PEND=0           # pending script can never fit a GPU here: do not wait for it
    fi
    NN=$NEED; [ $SHORT -eq 1 ] && NN=$NEED_SHORT; [ -n "$LNEED" ] && NN=$LNEED
    LIBHERE=0                                     # coordinator rule: nothing starts on a GPU that hosts a library run
    if [ -f RESERVE_g$GPU ]; then while read rp rm; do [ -d /proc/$rp ] && LIBHERE=1; done < RESERVE_g$GPU; fi
    [ $LIBHERE -eq 1 ] && { sleep 60; continue; }
    if { [ ! -f HOLD ] || [ $SHORT -eq 1 ]; } && [ "$PEND" -eq 0 ] && [ "$P" -lt "$PAR" ] && [ "$F" -ge "$NN" ]; then break; fi
    sleep 60
  done
  echo "$(date -u +%H:%M:%S) launch $TAG (procs=$P free=$F)" >> $LOG
  mkdir -p $DIR; touch $DIR.started
  env $EXTRA_ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU \
    setsid nohup bash -c "$CMD" > $DIR.out 2>&1 < /dev/null &
  sleep $SETTLE
done < $LIST
echo "=== launch2 gpu$GPU list done $(date -u) ===" >> $LOG
