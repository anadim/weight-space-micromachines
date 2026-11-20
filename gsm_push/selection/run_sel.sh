#!/bin/bash
# usage: run_sel.sh NAME GPU MINFREE_MIB [env...] -- [tw_dump_lean args]
# watchdog: kills OUR job only, if the GPU's free memory falls below MINFREE_MIB
N=$1; G=$2; MINF=$3; shift 3
source "$(dirname "$0")/../env.sh"; SEL_DIR=$(cd "$(dirname "$0")" && pwd); R=${SEL_WORK:-$WSM_WORK/selection}; CK=${CKPT:?set CKPT to the host checkpoint dir, see gsm_push/env.sh}
cd $WIRING_DIR
export SEL_DIR DIG_HOOK=${HOOK:-$SEL_DIR/sel_hook.py} CUDA_VISIBLE_DEVICES=$G
ENVS=(); while [ "$1" != "--" ]; do ENVS+=("$1"); shift; done; shift
mkdir -p $R/logs $R/rd
echo "$(date +%H:%M:%S) start $N gpu$G" >> $R/logs/queue.log
env SEL_OUT=$R/out "${ENVS[@]}" python3 $HARNESS --run_dir $R/rd/$N --eval_ckpt $CK --arm 2 "$@" > $R/logs/$N.out 2>&1 &
P=$!
while kill -0 $P 2>/dev/null; do
  F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $G)
  if [ "$F" -lt "$MINF" ]; then echo "$(date +%H:%M:%S) $N KILLED low free $F" >> $R/logs/queue.log; kill $P; sleep 3; kill -9 $P 2>/dev/null; fi
  sleep 2
done
echo "$(date +%H:%M:%S) end $N $(grep -c '^SELJOB' $R/logs/$N.out) jobs $(grep -q SELDONE $R/logs/$N.out && echo OK)" >> $R/logs/queue.log
