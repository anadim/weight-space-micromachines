#!/bin/bash
# Several off-task rows on ONE GPU (0.6B/1.7B evals use a fraction of a B200):
#   bash checks/offtask/ot_pack.sh SIZE [SMOKE=ROW:HOST:TASKS:LIM] ROW:HOST ...
# e.g. CUDA_VISIBLE_DEVICES=0 bash checks/offtask/ot_pack.sh 06 SMOKE=smk:CW_s0:mmlu,nq:20 CW_s0:CW_s0 P64_s1:P64_s1
# Starts are staggered 40 s (each row materializes its own code tree first); exit 3 if any row failed.
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); SIZE=$1; shift
source "$D/../../gsm_stack/env.sh"; O=$WSM_WORK/offtask/j$SIZE; mkdir -p $O
if [[ $1 == SMOKE=* ]]; then IFS=: read r h t l <<< "${1#SMOKE=}"; shift
  bash "$D/ot_job.sh" $SIZE $r $h $t $l > $O/pack_$r.txt 2>&1 || { echo "SMOKE $r FAILED"; tail -n 30 $O/pack_$r.txt; exit 3; }; echo "SMOKE $r ok"; fi
pids=(); rows=()
for a in "$@"; do r=${a%%:*}; h=${a#*:}; bash "$D/ot_job.sh" $SIZE $r $h > $O/pack_$r.txt 2>&1 & pids+=($!); rows+=($r); sleep 40; done
bad=0; for i in ${!pids[@]}; do wait ${pids[$i]}; rc=$?; echo "$(date -u +%TZ) ${rows[$i]} rc=$rc"; [ $rc = 0 ] || { bad=1; tail -n 20 $O/pack_${rows[$i]}.txt; }; done
exit $((bad*3))
