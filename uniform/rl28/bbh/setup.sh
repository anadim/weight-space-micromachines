#!/bin/bash
# [28] BBH 1.7B setup (usage: bash uniform/rl28/bbh/setup.sh): ONE GPU (B200), concurrent processes ([31] packing),
# dmon -> w/out/pack_setup/. Original run: one cluster-queue job (label j23812).
#   W1: call-FT 1.7B (ft17) -> then the gate refit (cg: states -> probe -> sets)
#   W2: base17 host check (HF) -> engine check (RL shape B 96 / budget 8192, memory) -> row 0 eval
#   after ft17: W3: hc ft17_t0 -> row 1 eval ; W4: hc ft17_t1 -> chk ft17_t1 -> idle (no gate) -> ctl eval ft17 + tools
#   after the probe: W5: hc ft17_t1g -> chk ft17_t1g -> idle with gate -> row 3 eval
B="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$B"; source "$B/../../env.sh"; PROBE=$RL28_DATA/bbh/cg/probe_ft17.pt
L=$B/w/out/pack_setup; mkdir -p $L; date -u > $L/start.txt
V=${CUDA_VISIBLE_DEVICES%%,*}; IDX=$V
if [[ "$V" == GPU-* || "$V" == MIG-* ]]; then IDX=$(nvidia-smi -L | grep -n "$V" | head -1 | cut -d: -f1); IDX=$((IDX - 1)); fi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES -> dmon index $IDX" | tee $L/gpu.txt; nvidia-smi -L >> $L/gpu.txt
nvidia-smi dmon -s um -d 10 -i $IDX > $L/dmon.txt 2>&1 & DM=$!
df -h $RL28_DATA | tail -1 | tee $L/df.txt
EV="--B_max 64 --mem_frac 0.28"
( cd $B/w; T0=$(date +%s); bash job.sh ft; echo "FT_WALL_S $(( $(date +%s) - T0 ))"
  cd $B/cg; T0=$(date +%s); NOSETS=1 bash job_refit.sh states; bash job_refit.sh probe; echo "REFIT_WALL_S $(( $(date +%s) - T0 ))"
  bash job_refit.sh sets; echo W1_DONE ) > $L/w1.log 2>&1 &
P1=$!
( cd $B/w; bash job.sh hc base17_t0 base17 0
  bash job.sh chk base17_t0 base17 0 - --B_max 96 --budget 8192 --mem_frac 0.40
  bash job.sh eval base17_plain base17 0 - - $EV; echo W2_DONE ) > $L/w2.log 2>&1 &
P2=$!
for i in $(seq 1 120); do [ -f $RL28_DATA/bbh/ft17/ft_args.json ] && break; kill -0 $P1 2>/dev/null || break; sleep 30; done
sleep 20
( cd $B/w; bash job.sh hc ft17_t0 ft17 0; bash job.sh eval ft17_t0_u0 ft17 0 - - $EV; echo W3_DONE ) > $L/w3.log 2>&1 &
P3=$!; sleep 20
( cd $B/w; bash job.sh hc ft17_t1 ft17 1
  bash job.sh chk ft17_t1 ft17 1 - --B_max 64 --budget 8192 --mem_frac 0.40
  bash job.sh idle ft17 ft17 - --B_max 64 --mem_frac 0.28 --n_check 408
  bash job.sh eval ft17_tools ft17 1 - - $EV; echo W4_DONE ) > $L/w4.log 2>&1 &
P4=$!
for i in $(seq 1 240); do [ -f $PROBE ] && break; kill -0 $P1 2>/dev/null || break; sleep 30; done
sleep 10
if [ -f $PROBE ]; then
  ( cd $B/w; bash job.sh hc ft17_t1g ft17 1 $PROBE
    bash job.sh chk ft17_t1g ft17 1 $PROBE --B_max 64 --budget 8192 --mem_frac 0.40
    bash job.sh idle ft17g ft17 $PROBE --B_max 64 --mem_frac 0.28 --n_check 408
    bash job.sh eval ft17_tools_gate ft17 1 $PROBE - $EV; echo W5_DONE ) > $L/w5.log 2>&1 &
  P5=$!
else echo "NO PROBE" | tee -a $L/gpu.txt; fi
wait $P1 $P2 $P3 $P4 $P5
kill $DM 2>/dev/null
awk '$1 ~ /^[0-9]+$/ {n++; s+=$2} END {if (n) printf "mean SM%% %.1f over %d samples\n", s/n, n}' $L/dmon.txt | tee -a $L/gpu.txt
df -h $RL28_DATA | tail -1 | tee -a $L/df.txt
grep -h "W._DONE\|FT_\|REFIT_WALL\|Traceback\|Error" $L/w*.log
date -u >> $L/start.txt
