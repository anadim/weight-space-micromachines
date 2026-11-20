#!/bin/bash
# usage: run_rl.sh NAME GPU [ENV=VAL ...]  RL job with fail-fast + auto-resume (resume file in $WSM_RESUME_DIR); host = $CKPT; XARGS
N=$1; G=$2; shift 2
source "$(dirname "$0")/../env.sh"; D=${STK_ROOT:-$WSM_WORK/final}
cd $WIRING_DIR
export DIG_HOOK=$STK_DIR/rl_hook.py CUDA_VISIBLE_DEVICES=$G SAN_SMAX=${SAN_SMAX:-1280} C2_NOCOMPACT=1 SER_WS=1
mkdir -p $D/logs $D/rd $D/rl/$N
for try in 1 2 3 4 5 6; do
  echo "$(date +%H:%M:%S) start $N gpu$G try$try" >> $D/logs/rl_queue.log
  env RL_RUN=$D/rl/$N "$@" python3 $HARNESS --run_dir $D/rd/rl_$N --arm 2 --fused 0 \
    --B_max ${BMAX:-128} --tok_budget 16384 --eval_ckpt ${CKPT:?set CKPT to the host checkpoint dir, see gsm_push/env.sh} $XARGS >> $D/logs/rl_$N.out 2>&1
  rc=$?
  if grep -q RLDONE $D/logs/rl_$N.out; then echo "$(date +%H:%M:%S) done $N" >> $D/logs/rl_queue.log; break; fi
  echo "$(date +%H:%M:%S) FAIL $N rc=$rc (resume)" >> $D/logs/rl_queue.log
  F=$(df -BG --output=avail "$WSM_WORK" | tail -1 | tr -dc 0-9); [ "$F" -lt 6 ] && { echo "low disk $F GB, stop" >> $D/logs/rl_queue.log; break; }
  sleep 20
done
