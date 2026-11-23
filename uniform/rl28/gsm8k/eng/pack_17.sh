#!/bin/bash
# usage: MEMFRAC=0.49 BMAX=96 FT_BUDGET=16384 bash uniform/rl28/gsm8k/eng/pack_17.sh NAME "ARM1 ARM2"      (one GPU)
# Packed 1.7B RL job: several arms as concurrent processes on ONE GPU, each `run_arm_17.sh ARM` with --mem_frac MEMFRAC
# (default 0.45) and B_max BMAX (default 192); restarted (every arm resumes) until every arm has eng/out/arm_<ARM>/DONE.
# P17_* arms always use FT_BUDGET 32768; FT_BUDGET (micro-batch token budget, gradient accumulation only: the recipe is
# unchanged) applies to the CW17_* arms. Original runs (one B200 each; one cluster-queue job per start, 12-h limit):
#   pack P17a:  P17_s0 + P17_s1      MEMFRAC 0.45, BMAX 192
#   pack CW17a: CW17_s0 + CW17_s1    MEMFRAC 0.49, BMAX 96, FT_BUDGET 16384 (later each arm alone, same settings)
#   pack S2:    CW17_s2 + P17_s2     MEMFRAC 0.49, BMAX 96, FT_BUDGET 16384 (CW17) / 32768 (P17)
# The original pack_17.sh / pack2_17.sh / pack3_17.sh differed only in progress lines and the per-arm FT_BUDGET rule; this
# is pack3_17.sh. Logs: eng/out/pack_<NAME>.chain.log, eng/out/inner_<ARM>.txt, SM% eng/out/dmon_<NAME>.txt.
NAME=$1; ARMS=$2
[ -n "$NAME" ] && [ -n "$ARMS" ] || { echo "usage: $0 NAME \"ARM1 ARM2\""; exit 64; }
E="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$E/../../../env.sh"
MF=${MEMFRAC:-0.45}; BM=${BMAX:-192}; FTB0=${FT_BUDGET:-32768}
mkdir -p $E/out; CL=$E/out/pack_$NAME.chain.log
GI=${CUDA_VISIBLE_DEVICES%%,*}
for chain in $(seq 1 ${MAXCHAIN:-12}); do
  echo "$(date -u +%FT%TZ) chain $chain start (pack $NAME: $ARMS, mem_frac $MF, B_max $BM)" >> $CL
  nvidia-smi dmon -s u -d 10 -i $GI >> $E/out/dmon_$NAME.txt 2>&1 &
  DM=$!
  ( while sleep 600; do for a in $ARMS; do f=$ARMDIR17/rl/$a/steps.jsonl
      echo "$(date -u +%H:%MZ) $a $(grep -c '"u":' $f 2>/dev/null) step lines | $(tail -n 1 $f 2>/dev/null | cut -c1-160)"; done; done ) &
  MON=$!
  pids=""
  for a in $ARMS; do
    FTB=$FTB0; case $a in P17_*) FTB=32768;; esac
    MEMFRAC=$MF BMAX=$BM FT_BUDGET=$FTB timeout ${TMO:-43200} bash $E/run_arm_17.sh $a > $E/out/inner_$a.txt 2>&1 &
    pids="$pids $!"; sleep 60
  done
  rc=0; for p in $pids; do wait $p; r=$?; [ $r -gt $rc ] && rc=$r; done
  kill $DM $MON
  echo "$(date -u +%FT%TZ) chain $chain ended rc=$rc" >> $CL
  left=""; for a in $ARMS; do [ -f $E/out/arm_$a/DONE ] || left="$left $a"; done
  if [ -z "$left" ]; then echo "$(date -u +%FT%TZ) pack $NAME RLDONE" >> $CL; exit 0; fi
  case $rc in 64|2|3) echo "  arm failure rc=$rc (left:$left): stop" >> $CL; exit $rc;; esac
  ARMS=$(echo $left); sleep 30
done
exit 1
