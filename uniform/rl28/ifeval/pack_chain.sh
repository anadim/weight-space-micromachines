#!/bin/bash
# pack_chain.sh PACK "ARM ..." [--inner] -- task [28]/[31](a): several 1.7B IFEval RL arms as concurrent processes in ONE job
#   on one GPU, each with a CUDA memory cap; chained across 4-h time limits (every start resumes from $DATA/rl/NAME/resume.pt).
#   (uniform/rl24/ifeval/pack_chain.sh without the side lane.) Each arm = rl_chain.sh NAME --inner with envs/NAME.env:
#   RL_EXTRA="--B_max $BMAX --mem_frac $ARM_FRAC --workers 6", EVAL_EXTRA="--B_max 128 --mem_frac $ARM_FRAC".
#   B_max = rollout batch slots (KV cache), not the recipe. An arm that dies with a CUDA OOM is restarted once in the
#   same job with B_max halved (it resumes from its last resume state). The job logs `nvidia-smi dmon` of its GPU to out/PACK/.
# usage: bash uniform/rl28/ifeval/pack_chain.sh PACK "P_s0 P_s1"   (log out/PACK.chain.log)
# Original packs: pkA = "P_s0 P_s1", pkB = "I_s0 I_s1" (after the setup/check job), pkC = "P_s2 I_s2" (after pkA).
PACK=$1; ITEMS=$2; MODE=$3
HERE=$(cd "$(dirname "$0")" && pwd); cd $HERE
ARM_FRAC=${ARM_FRAC:-0.46}; BMAX=${BMAX:-256}
arms() { echo $ITEMS; }
alldone() { for x in $(arms); do [ -e out/$x/DONE ] || return 1; done; return 0; }
if [ "$MODE" != "--inner" ]; then
  # Original runs: one cluster-queue job per start (B200, 4 h limit). Here: the pack runs on this machine's GPU, restarted until done.
  mkdir -p out/$PACK envs; CL=out/$PACK.chain.log
  for x in $(arms); do
    { echo "export MODEL=\"${MODEL:-Qwen/Qwen3-1.7B-Base}\""; [ -n "$TAG" ] && echo "export TAG=\"$TAG\""; echo "export FMT=\"${FMT:-chat}\""
      echo "export RL_EXTRA=\"--B_max $BMAX --mem_frac $ARM_FRAC --workers 6\""; echo "export EVAL_EXTRA=\"--B_max 128 --mem_frac $ARM_FRAC\""
      echo "export CKPT_EVERY=\"${CKPT_EVERY:-300}\""; } > envs/$x.env
  done
  for chain in $(seq 1 ${MAXCHAIN:-8}); do
    echo "$(date -u +%FT%TZ) chain $chain start ($PACK: $ITEMS)" >> $CL
    timeout ${TMO:-14400} bash pack_chain.sh $PACK "$ITEMS" --inner; rc=$?
    echo "$(date -u +%FT%TZ) chain $chain ended rc=$rc" >> $CL
    if alldone; then echo "$(date -u +%FT%TZ) $PACK ALLDONE" >> $CL; exit 0; fi
    [ $rc = 3 ] && { echo "  in-job failures: stop" >> $CL; exit 3; }
    sleep 30
  done
  exit 1
fi

# ------------------------------------------------------------------------------- inside the job (cwd = this folder)
H=$PWD; C=$H/code; source $C/env.sh; O=$H/out/$PACK; M=$DATA/pack/$PACK; mkdir -p $O $M
JID=$(date -u +%H%M%S)
CV=${CUDA_VISIBLE_DEVICES:-0}; CV=${CV%%,*}
case $CV in GPU-*) GI=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v u=$CV '$2==u{print $1}');; *) GI=$CV;; esac
nvidia-smi dmon -s um -d 10 -i $GI -o DT > $O/dmon_$JID.txt 2>&1 & DM=$!
echo "$(date -u +%FT%TZ) pack $PACK arms: $ITEMS  gpu $CV (nvidia-smi index $GI)" | tee -a $O/pack.log
prep_pylib > $DATA/pylib.log 2>&1; (cd $C/heldout; build_tables)
( while sleep 300 > /dev/null 2>&1; do for x in $(arms); do echo "$(date -u +%H:%MZ) $x $(grep -a '^RL [PI] [{d]' $DATA/rl/$x/job.log 2>/dev/null | tail -1 | grep -oE '"(u|reward|len|roll_s|train_s|update_s|mem_gb|save_block_s|dev_strict)": [-0-9.e]+' | tr -d '"' | tr '\n' ' ')"; done
    nvidia-smi -i $GI --query-gpu=utilization.gpu,memory.used --format=csv,noheader; done ) & PR=$!
PIDS=""
for x in $(arms); do
  [ -e $M/$x.done ] && continue
  (for t in 1 2; do
     bash rl_chain.sh $x --inner > /dev/null 2>&1; rc=$?; [ $rc = 0 ] && { touch $M/$x.done; break; }
     tail -n 200 $DATA/rl/$x/job.log | grep -q "OutOfMemoryError\|out of memory" || break
     b=$(grep -o -- '--B_max [0-9]*' envs/$x.env | head -1 | awk '{print $2}'); nb=$((b / 2))
     sed -i "s/--B_max $b --mem_frac/--B_max $nb --mem_frac/" envs/$x.env
     echo "$(date -u +%FT%TZ) arm $x OOM at B_max $b -> retry with $nb" | tee -a $O/pack.log
   done
   echo "$(date -u +%FT%TZ) arm $x rc=$rc" | tee -a $O/pack.log) & PIDS="$PIDS $!"
done
wait $PIDS
kill $DM $PR 2>/dev/null
awk '$1 !~ /^#/ {s+=$4; n++} END {if (n) printf "mean SM%% %.1f over %d samples\n", s/n, n}' $O/dmon_$JID.txt | tee -a $O/pack.log
alldone && exit 0; for x in $(arms); do [ -e $M/$x.done ] || exit 1; done; exit 0
