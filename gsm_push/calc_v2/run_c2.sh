#!/bin/bash
# usage: run_c2.sh NAME MEMFRAC BMAX [ENV=VAL ...]   one job on A100; watchdog kills OUR job if GPU free < 2000 MiB
N=$1; MF=$2; BM=$3; shift 3
source "$(dirname "$0")/../env.sh"; cd $WIRING_DIR
C2_DIR=$(cd "$(dirname "$0")" && pwd); export DIG_HOOK=$C2_DIR/c2_eval.py C2_DIR
D=${C2_WORK:-$WSM_WORK/calc_v2}; mkdir -p $D
echo "$(date -u +%H:%M:%S) start $N" >> $D/queue.log
env C2_OUT=$D/out/$N "$@" python $HARNESS --B_max $BM --tok_budget 8192 --mem_frac $MF --fused 0 --arm 2 \
   --run_dir $D/rd/$N --eval_ckpt ${CKPT:?set CKPT to the host checkpoint dir, see gsm_push/env.sh} > $D/$N.out 2>&1 &
P=$!
while kill -0 $P 2>/dev/null; do
  F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  if [ "$F" -lt 2000 ]; then echo "$(date -u +%H:%M:%S) $N KILLED low free $F" >> $D/queue.log; kill $P; sleep 5; kill -9 $P 2>/dev/null; fi
  sleep 2
done
echo "$(date -u +%H:%M:%S) end $N $(grep -c '^C2 ' $D/$N.out) evals $(grep -q C2DONE $D/$N.out && echo OK)" >> $D/queue.log
