#!/bin/bash
# pack_chain.sh PACK "ITEM ..." [--inner] -- task [31](a): several IFEval RL arms (+ side jobs) as concurrent processes in ONE
#   job on one GPU, each with a CUDA memory cap; chained across 4-h time limits like rl_chain.sh (every start resumes).
#   ITEM = an arm name (P_s*/I_s*: rl_chain.sh NAME --inner with envs/NAME.env) or a side job run in its own lane
#   one after the other: degen ([31](b) rule measurement), P_s0x (row-2b re-eval of P_s0 u300 with sd_inst).
#   Arms get RL_EXTRA="--B_max 256 --mem_frac $ARM_FRAC --workers 6" (KV cache of B_max 512 = 72 GB does not fit 2-3 per GPU)
#   and EVAL_EXTRA="--B_max 128 --mem_frac $ARM_FRAC"; side jobs --mem_frac $SIDE_FRAC.
#   Arms start only once $DATA/rl/NAME/steps.jsonl is >= 120 s old (keeps two processes off one run dir). Each job logs `nvidia-smi dmon -s um -d 10` of its GPU to out/PACK/dmon_<jobid>.txt.
# usage: bash uniform/rl24/ifeval/pack_chain.sh PACK "P_s2 I_s2 degen P_s0x"   (log out/PACK.chain.log)
PACK=$1; ITEMS=$2; MODE=$3
HERE=$(cd "$(dirname "$0")" && pwd); cd $HERE
ARM_FRAC=${ARM_FRAC:-0.35}; SIDE_FRAC=${SIDE_FRAC:-0.22}
arms() { for x in $ITEMS; do case $x in [PI]_s[0-9]*) [ "$x" != P_s0x ] && echo $x;; esac; done; }
sides() { for x in $ITEMS; do case $x in degen|P_s0x) echo $x;; esac; done; }
alldone() { for x in $(arms); do [ -e out/$x/DONE ] || return 1; done; for x in $(sides); do [ -e out/$PACK/$x.done ] || return 1; done; return 0; }
if [ "$MODE" != "--inner" ]; then
  # Original runs: one cluster-queue job per start (B200, 4 h limit). Here: the pack runs on this machine's GPU, restarted until done.
  mkdir -p out/$PACK envs; CL=out/$PACK.chain.log
  for x in $(arms); do
    { echo "export MODEL=\"${MODEL:-Qwen/Qwen3-0.6B-Base}\""; echo "export TAG=\"${TAG:-inst}\""; echo "export FMT=\"${FMT:-chat}\""
      echo "export RL_EXTRA=\"--B_max 256 --mem_frac $ARM_FRAC --workers 6\""; echo "export EVAL_EXTRA=\"--B_max 128 --mem_frac $ARM_FRAC\""
      echo "export CKPT_EVERY=\"${CKPT_EVERY:-300}\""; } > envs/$x.env
  done
  echo "$ITEMS" > envs/$PACK.pack
  for chain in $(seq 1 ${MAXCHAIN:-8}); do
    echo "$(date -u +%FT%TZ) chain $chain start ($PACK: $ITEMS)" >> $CL
    timeout ${TMO:-14400} bash pack_chain.sh $PACK "$ITEMS" --inner; rc=$?
    echo "$(date -u +%FT%TZ) chain $chain ended rc=$rc" >> $CL
    if alldone; then
      echo "$(date -u +%FT%TZ) $PACK ALLDONE" >> $CL
      [ -e out/$PACK/eval_P_s0x.imp.score.json ] && cp out/$PACK/eval_P_s0x.imp.score.json out/P_s0/eval_u300.imp.score.json \
        && echo "  row 2b: out/P_s0/eval_u300.imp.score.json <- out/$PACK/eval_P_s0x.imp.score.json" >> $CL
      ls out/{P,I}_s{0,1,2}/DONE > /dev/null 2>&1 && AO=out python3 analyze.py --out RESULTS_TABLE.md > /dev/null && echo "  all 6 arms DONE; RESULTS_TABLE.md written" >> $CL
      exit 0
    fi
    [ $rc = 3 ] && { echo "  in-job failures: stop" >> $CL; exit 3; }
    sleep 30
  done
  exit 1
fi

# ------------------------------------------------------------------------------- inside the job (cwd = this folder)
H=$PWD; C=$H/code; source $C/env.sh; O=$H/out/$PACK; M=$DATA/pack/$PACK; mkdir -p $O $M $DATA/degen
JID=$(date -u +%H%M%S)
CV=${CUDA_VISIBLE_DEVICES:-0}; CV=${CV%%,*}
case $CV in GPU-*) GI=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v u=$CV '$2==u{print $1}');; *) GI=$CV;; esac
nvidia-smi dmon -s um -d 10 -i $GI -o DT > $O/dmon_$JID.txt 2>&1 & DM=$!
echo "$(date -u +%FT%TZ) pack $PACK items: $ITEMS  gpu $CV (nvidia-smi index $GI)" | tee -a $O/pack.log
prep_pylib > $DATA/pylib.log 2>&1; (cd $C/heldout; build_tables)
SD=$DATA/eng/sd_inst.pt
( flock 9; [ -e $SD ] || (cd $C/heldout && PYTHONPATH=$C/heldout:$C/f14 python ../hostchk/mk_sd.py $SD Qwen/Qwen3-0.6B) ) 9> $DATA/eng.lock
side() {
  case $1 in
    degen) G=$DATA/eval; RL=$DATA/rl
      (cd $C/eng && python degen.py --sd $SD --mem_frac $SIDE_FRAC --B_max 128 --workers 6 --out $DATA/degen/degen \
        --cfg base_u0:Qwen/Qwen3-0.6B-Base:-:plain:2 base_u0_imp:Qwen/Qwen3-0.6B-Base:-:imp:2 \
              P_s0_u100:Qwen/Qwen3-0.6B-Base:$RL/P_s0/step_000100/model.safetensors:plain:1 \
              P_s0_u300:Qwen/Qwen3-0.6B-Base:$RL/P_s0/step_000300/model.safetensors:plain:1 \
              I_s0_u100_imp:Qwen/Qwen3-0.6B-Base:$RL/I_s0/step_000100/model.safetensors:imp:1 \
        --jsonl g541_u0_plain:$G/u0/eval_u0.plain.jsonl g541_u0_imp:$G/u0/eval_u0.imp.jsonl \
                g541_P_s0_plain:$RL/P_s0/eval_u300.plain.jsonl g541_P_s0_imp_refit:$RL/P_s0/eval_u300.imp.jsonl \
                g541_ftint_plain:$G/ftint/eval_ftint.plain.jsonl g541_ftint_imp:$G/ftint/eval_ftint.imp.jsonl) \
        > $O/degen.log 2>&1 && cp $DATA/degen/degen.summary.json $O/ && touch $M/degen.done;;
    P_s0x) R=$DATA/eval/P_s0x; mkdir -p $R
      (cd $C/eng && python eval541.py --model Qwen/Qwen3-0.6B-Base --ckpt $DATA/rl/P_s0/step_000300/model.safetensors --sd $SD \
         --modes imp,plain --fmt chat --out $R/eval_P_s0x --B_max 64 --mem_frac $SIDE_FRAC) > $O/P_s0x.log 2>&1 && \
      PYTHONPATH=$PYL python $C/score_reduce.py $C/ifdata/ifeval.jsonl $R/eval_P_s0x.imp.jsonl $O/eval_P_s0x.imp.score.json \
          --command "rl24/ifeval P_s0x u300 sd_inst imp" > /dev/null && \
      PYTHONPATH=$PYL python $C/score_reduce.py $C/ifdata/ifeval.jsonl $R/eval_P_s0x.plain.jsonl $O/eval_P_s0x.plain.score.json \
          --command "rl24/ifeval P_s0x u300 sd_inst plain" > /dev/null && \
      cp $R/eval_P_s0x.summary.json $O/ && touch $M/P_s0x.done;;
  esac
  echo "$(date -u +%FT%TZ) side $1 done=$([ -e $M/$1.done ] && echo yes || echo NO)" | tee -a $O/pack.log
}
( while sleep 300 > /dev/null 2>&1; do for x in $(arms); do echo "$(date -u +%H:%MZ) $x $(grep -a '^RL [PI] [{d]' $DATA/rl/$x/job.log 2>/dev/null | tail -1 | cut -c1-230)"; done
    for x in $(sides); do [ -e $O/$x.log ] && echo "$(date -u +%H:%MZ) $x $(grep -av 'Unable to detect' $O/$x.log | tail -1 | cut -c1-200)"; done
    nvidia-smi -i $GI --query-gpu=utilization.gpu,memory.used --format=csv,noheader; done ) & PR=$!
PIDS=""
for x in $(arms); do
  [ -e $M/$x.done ] && continue
  (while [ -e $DATA/rl/$x/steps.jsonl ] && [ $(( $(date +%s) - $(stat -c %Y $DATA/rl/$x/steps.jsonl) )) -lt 120 ]; do sleep 15; done
   [ -e $M/$x.done ] && exit 0
   bash rl_chain.sh $x --inner > /dev/null 2>&1; rc=$?; [ $rc = 0 ] && touch $M/$x.done
   echo "$(date -u +%FT%TZ) arm $x rc=$rc" | tee -a $O/pack.log) & PIDS="$PIDS $!"
done
( for x in $(sides); do [ -e $M/$x.done ] && { touch $O/$x.done; continue; }; side $x; [ -e $M/$x.done ] && touch $O/$x.done; done ) & PIDS="$PIDS $!"
wait $PIDS
kill $DM $PR 2>/dev/null
awk '$1 !~ /^#/ {s+=$4; n++} END {if (n) printf "mean SM%% %.1f over %d samples\n", s/n, n}' $O/dmon_$JID.txt | tee -a $O/pack.log
alldone && exit 0; exit 1
