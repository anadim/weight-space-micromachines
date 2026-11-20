#!/bin/bash
# usage: run_c2_h100.sh NAME GPU MEMFRAC BMAX [ENV=VAL ...]  one job on H100; watchdog kills OUR job if GPU free < 3000 MiB
N=$1; G=$2; MF=$3; BM=$4; shift 4
source "$(dirname "$0")/../env.sh"; cd $WIRING_DIR
C2_DIR=$(cd "$(dirname "$0")" && pwd); D=${C2_WORK:-$WSM_WORK/calc_v2}; mkdir -p $D
export DIG_HOOK=$C2_DIR/c2_eval.py C2_DIR CUDA_VISIBLE_DEVICES=$G
echo "$(date -u +%H:%M:%S) start $N gpu$G" >> $D/queue.log
env C2_OUT=$D/out/$N "$@" python3 $HARNESS --B_max $BM --tok_budget 2048 --mem_frac $MF --arm 2 \
   --run_dir $D/rd/$N --eval_ckpt ${CKPT:?set CKPT to the host checkpoint dir, see gsm_push/env.sh} > $D/$N.out 2>&1 &
P=$!
while kill -0 $P 2>/dev/null; do
  F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $G)
  if [ "$F" -lt 3000 ]; then echo "$(date -u +%H:%M:%S) $N KILLED low free $F" >> $D/queue.log; kill $P; sleep 3; kill -9 $P 2>/dev/null; fi
  sleep 2
done
echo "$(date -u +%H:%M:%S) end $N $(grep -c '^C2 ' $D/$N.out) evals $(grep -q C2DONE $D/$N.out && echo OK)" >> $D/queue.log
