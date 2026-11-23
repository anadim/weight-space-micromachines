#!/bin/bash
# usage: bash uniform/final_eval/fepack.sh TREE LIST1 [LIST2 ...]       (one GPU)
# Packed eval job: several lists as concurrent fejob.sh processes on ONE GPU, each capped at 0.9/N of its memory.
# Each list's stdout -> $WSM_WORK/pack/<list>.log; SM% / memory every 10 s -> $WSM_WORK/pack/dmon.txt.
U="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$U/../env.sh"
S=$1; shift; N=$#
PD=$WSM_WORK/pack; mkdir -p $PD
nvidia-smi dmon -i ${CUDA_VISIBLE_DEVICES%%,*} -s um -d 10 -o DT > $PD/dmon.txt 2>&1 &
DM=$!
export FE_MEMFRAC=$(python3 -c "print(round(0.9/$N, 3))")
# rung-2 KV is preallocated for B_max x S_max (0.6B: ~0.94 GB/seq at 8192), so B_max shrinks with the share (B 128 needs a whole GPU)
PBM=${FE_PACK_BMAX:-$((120 / N))}
P=()
for L in "$@"; do
  bash $U/fejob.sh $S $L $PBM > $PD/$(basename ${L%.list}).log 2>&1 &
  P+=($!)
  sleep 20   # stagger the one-time setup (tree build, test_aw)
done
rc=0
for p in "${P[@]}"; do wait $p || rc=$?; done
kill $DM 2>/dev/null
for L in "$@"; do echo "== $L"; grep -E "^FE |rror|FAILED" $PD/$(basename ${L%.list}).log | tail -n 20; done
awk '!/^#/ && NF>4 {s+=$4; n++} END {if (n) printf "PACK mean SM%% %.1f over %d samples, %d procs\n", s/n, n, '$N'}' $PD/dmon.txt
exit $rc
