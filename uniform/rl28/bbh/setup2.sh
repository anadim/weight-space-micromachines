#!/bin/bash
# [28] BBH 1.7B setup, part 2 (usage: bash uniform/rl28/bbh/setup2.sh; one GPU; everything is also copied into ./out).
# j23812 ran setup.sh fine (95% SM) but its results sat in w/out + cg/out, which the original cluster queue did not copy
# back (only ./out); ft17 and probe_ft17.pt persisted on the node. This job (label j23836) redoes the checks/evals and
# re-derives the probe to a side file (sha must match the probe in use; w/cmp_probe.py). chk steps are serialized (flock)
# to bound memory. Not needed for a fresh reproduction: setup.sh alone produces everything.
B="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$B"; source "$B/../../env.sh"; PROBE=$RL28_DATA/bbh/cg/probe_ft17.pt
mkdir -p out w/out cg/out; L=$B/w/out/pack_setup2; mkdir -p $L; date -u > $L/start.txt
trap 'mkdir -p $B/out/w $B/out/cg; cp -r $B/w/out/. $B/out/w/; cp -r $B/cg/out/. $B/out/cg/' EXIT
V=${CUDA_VISIBLE_DEVICES%%,*}; IDX=$V
if [[ "$V" == GPU-* || "$V" == MIG-* ]]; then IDX=$(nvidia-smi -L | grep -n "$V" | head -1 | cut -d: -f1); IDX=$((IDX - 1)); fi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES -> dmon index $IDX" | tee $L/gpu.txt; nvidia-smi -L >> $L/gpu.txt
nvidia-smi dmon -s um -d 10 -i $IDX > $L/dmon.txt 2>&1 & DM=$!
df -h $RL28_DATA | tail -1 | tee $L/df.txt
EV="--B_max 64 --mem_frac 0.28"; LK=$L/chk.lock
CHK() { flock $LK bash job.sh chk "$@"; }
( cd $B/w; bash job.sh ft
  cd $B/cg; PROBE_OUT=probe_ft17_rederive.pt bash job_refit.sh probe; bash job_refit.sh sets; echo W1_DONE ) > $L/w1.log 2>&1 &
P1=$!; sleep 15
( cd $B/w; bash job.sh hc base17_t0 base17 0
  CHK base17_t0 base17 0 - --B_max 96 --budget 8192 --mem_frac 0.40
  bash job.sh eval base17_plain base17 0 - - $EV; echo W2_DONE ) > $L/w2.log 2>&1 &
P2=$!; sleep 15
( cd $B/w; bash job.sh hc ft17_t0 ft17 0; bash job.sh eval ft17_t0_u0 ft17 0 - - $EV
  bash job.sh hc ft17_t1 ft17 1
  CHK ft17_t1 ft17 1 - --B_max 64 --budget 8192 --mem_frac 0.40
  bash job.sh idle ft17 ft17 - --B_max 64 --mem_frac 0.28 --n_check 408
  bash job.sh eval ft17_tools ft17 1 - - $EV; echo W3_DONE ) > $L/w3.log 2>&1 &
P3=$!; sleep 15
( cd $B/w; bash job.sh hc ft17_t1g ft17 1 $PROBE
  CHK ft17_t1g ft17 1 $PROBE --B_max 64 --budget 8192 --mem_frac 0.40
  bash job.sh idle ft17g ft17 $PROBE --B_max 64 --mem_frac 0.28 --n_check 408
  bash job.sh eval ft17_tools_gate ft17 1 $PROBE - $EV; echo W4_DONE ) > $L/w4.log 2>&1 &
P4=$!
wait $P1 $P2 $P3 $P4
kill $DM 2>/dev/null
awk '$1 ~ /^[0-9]+$/ {n++; s+=$2} END {if (n) printf "mean SM%% %.1f over %d samples\n", s/n, n}' $L/dmon.txt | tee -a $L/gpu.txt
df -h $RL28_DATA | tail -1 | tee -a $L/df.txt
grep -h "W._DONE\|SUMMARY\|IDLE\|CHECK\|Traceback\|Error\|max_mem\|AUROC\|auroc" $L/w*.log | cut -c1-1500
cat $B/cg/out/probe_log.txt 2>/dev/null | tail -5
date -u >> $L/start.txt
