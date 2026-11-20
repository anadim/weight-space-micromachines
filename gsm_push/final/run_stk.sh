#!/bin/bash
# usage: run_stk.sh NAME GPU SET STEPS [ENV=VAL ...]   one job; host = $CKPT (default RL a2_u300_s2); XARGS e.g. "--host_ckpt base"
N=$1; G=$2; SET=$3; STEPS=$4; shift 4
source "$(dirname "$0")/../env.sh"; D=${STK_ROOT:-$WSM_WORK/final}
cd $WIRING_DIR
export DIG_HOOK=$STK_DIR/${HOOK:-stack_hook.py} CUDA_VISIBLE_DEVICES=$G SAN_SMAX=${SAN_SMAX:-1024} C2_NOCOMPACT=1
mkdir -p $D/logs $D/rd
echo "$(date +%H:%M:%S) start $N gpu$G $SET $STEPS" >> $D/logs/queue.log
env STK_OUT=$D/out/$N STK_SET=$SET STK_STEPS=$STEPS "$@" python3 $HARNESS --run_dir $D/rd/$N --arm 2 --fused 0 \
  --B_max ${BMAX:-128} --tok_budget 2048 --eval_ckpt ${CKPT:?set CKPT to the host checkpoint dir, see gsm_push/env.sh} $XARGS > $D/logs/$N.$SET.out 2>&1
echo "$(date +%H:%M:%S) end $N $SET rc=$? $(grep -q STKDONE $D/logs/$N.$SET.out && echo OK)" >> $D/logs/queue.log
