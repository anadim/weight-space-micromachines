#!/bin/bash
# usage: run_rd.sh NAME GPU MEMFRAC VARIANTS   (one job; watchdog kills OUR job if GPU free < 3000 MiB)
N=$1; G=$2; MF=$3; V=$4
source "$(dirname "$0")/../env.sh"; RD_DIR=$(cd "$(dirname "$0")" && pwd); D=${RD_WORK:-$WSM_WORK/reading}; mkdir -p $D
cd $WIRING_DIR
export SAN_DIR=$RD_DIR RD_DIR DIG_HOOK=$RD_DIR/${HOOK:-rd_eval.py} CUDA_VISIBLE_DEVICES=$G SAN_SMAX=${SMAX:-1024}
echo "$(date +%H:%M:%S) start $N gpu$G $V" >> $D/queue.log
env RD_OUT=$D/out/$N RD_VARIANTS=$V RD_N=${RD_N:-0} python3 $HARNESS --run_dir $D/rd/$N --arm 2 --B_max ${BMAX:-16} --tok_budget 2048 --mem_frac $MF --eval_ckpt ${CKPT:?set CKPT to the host checkpoint dir, see gsm_push/env.sh} $XARGS > $D/$N.$(date +%H%M%S).out 2>&1 &
P=$!
while kill -0 $P 2>/dev/null; do
  F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $G)
  if [ "$F" -lt 3000 ]; then echo "$(date +%H:%M:%S) $N KILLED low free $F" >> $D/queue.log; kill $P; sleep 3; kill -9 $P 2>/dev/null; fi
  sleep 2
done
echo "$(date +%H:%M:%S) end $N" >> $D/queue.log
