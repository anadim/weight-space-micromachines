#!/bin/bash
# task [31] packing: run several job.sh commands as concurrent processes on ONE GPU, each with a memory cap.
# usage (from rl24/bbh/w): bash pack.sh TAG MEMFRAC "CMD1" "CMD2" ...   ; each CMD is a ';'-separated sequence of
# job.sh calls run in order by one worker (e.g. "bash job.sh eval A ...; bash job.sh eval B ...").
# --mem_frac MEMFRAC is appended to every rl/eval/chk call. Logs: out/pack_TAG/wK.log, SM% dmon: out/pack_TAG/dmon.txt
TAG=$1; MF=$2; shift 2; L=out/pack_$TAG; rm -rf $L; mkdir -p $L
V=${CUDA_VISIBLE_DEVICES%%,*}; IDX=$V
if [[ "$V" == GPU-* || "$V" == MIG-* ]]; then IDX=$(nvidia-smi -L | grep -n "$V" | head -1 | cut -d: -f1); IDX=$((IDX - 1)); fi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES -> dmon index $IDX" | tee $L/gpu.txt; nvidia-smi -L >> $L/gpu.txt
nvidia-smi dmon -s u -d 10 -i $IDX > $L/dmon.txt 2>&1 & DM=$!
k=0; PIDS=()
for C in "$@"; do
  C=$(echo "$C" | sed -E "s/(bash job.sh (eval|rl|chk) [^;]*)/\1 --mem_frac $MF/g")
  echo "w$k: $C" | tee -a $L/cmds.txt
  bash -c "$C" > $L/w$k.log 2>&1 & PIDS+=($!); k=$((k + 1)); sleep 20
done
RC=0; for p in "${PIDS[@]}"; do wait $p || RC=1; done
kill $DM 2>/dev/null
awk '$1 ~ /^[0-9]+$/ {n++; s+=$2} END {if (n) printf "mean SM%% %.1f over %d samples\n", s/n, n}' $L/dmon.txt | tee -a $L/gpu.txt
grep -h SUMMARY $L/w*.log
exit $RC
