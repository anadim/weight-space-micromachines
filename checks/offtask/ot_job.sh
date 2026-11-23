#!/bin/bash
# One off-task row on one GPU: bash checks/offtask/ot_job.sh SIZE ROW HOST [TASKS] [LIM]
#   SIZE  06 (Qwen3-0.6B-Base host, tree j06/tree) | 17 (Qwen3-1.7B-Base host, tree j17/tree)
#   ROW   row tag, e.g. base | P64_s0 | CW_s1 | base17 | P17_s2 | CW17_s0
#   HOST  base (the Hub base model) | ARM (0.6B: $ARMDIR/rl/ARM/step_000300; 1.7B: $ARMDIR17/rl/ARM/step_000300) | /abs/ckpt_dir
#   TASKS default mmlu,arcc,hs,wg,nq;  LIM items per task (0 = all; e.g. 20 for a smoke test)
# Runs stk/ot_hook.py inside dig_san/tw_dump_lean.py exactly as the final_eval launchers run fe_hook.py (rung-2 tree:
# 0.6B uniform/final_eval/trees/stk_r2q via gsm_stack/run_hook.sh, 1.7B uniform/rl28/gsm8k/trees/stk_r2q_17 via
# uniform/rl28/gsm8k/fe/run_hook.sh). Benchmark data: $OT_DATA (default $WSM_WORK/offtask/bench, from dl.sh).
# Rows -> $WSM_WORK/offtask/jSIZE/ot/ROW_<task>.jsonl, then: python3 checks/offtask/mk_items.py <that dir> checks/offtask/jSIZE/res
# Original runs: one job per GPU via a cluster queue (several rows packed per GPU, see ot_pack.sh). Run on one GPU:
#   CUDA_VISIBLE_DEVICES=0 bash checks/offtask/ot_job.sh 06 CW_s0 CW_s0
SIZE=$1; ROW=$2; HOST=$3; TASKS=${4:-mmlu,arcc,hs,wg,nq}; LIM=${5:-0}
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$D/../../gsm_stack/env.sh"
cd "$WSM_ROOT"
case $SIZE in
  06) RH=gsm_stack/run_hook.sh; AD=$ARMDIR;;
  17) RH=uniform/rl28/gsm8k/fe/run_hook.sh; AD=${ARMDIR17:-$WSM_DATA/rl28g};;
  *) echo "SIZE must be 06 or 17"; exit 2;;
esac
case $HOST in base) XA="--host_ckpt base";; /*) XA="--host_ckpt $HOST";; *) XA="--host_ckpt $AD/rl/$HOST/step_000300";; esac
OT_DATA=${OT_DATA:-$WSM_WORK/offtask/bench}; OUT=$WSM_WORK/offtask/j$SIZE
[ -f "$OT_DATA/mmlu_test.jsonl" ] || { echo "no benchmark data in OT_DATA=$OT_DATA (bash checks/offtask/dl.sh)"; exit 2; }
mkdir -p $OUT/ot
echo "$(date -u +%FT%TZ) $SIZE $ROW $HOST $TASKS lim=$LIM gpu $CUDA_VISIBLE_DEVICES" | tee -a $OUT/job_$ROW.log
XARGS="$XA" BMAX=16 SAN_SMAX=1024 bash $RH ot${SIZE}_$ROW checks/offtask/j$SIZE/tree ot_hook.py OT_OUT=$OUT/ot OT_TAG=$ROW \
  OT_DATA=$OT_DATA OT_TASKS=$TASKS OT_LIM=$LIM < /dev/null 2>&1 | grep --line-buffered -E "^OT |rror|Traceback|OTDONE" | cut -c1-600 | tee -a $OUT/job_$ROW.log
L=$WSM_WORK/out/ot${SIZE}_$ROW/log.txt
grep -q OTDONE $L || { echo "FAILED"; tail -n 30 $L; exit 3; }
