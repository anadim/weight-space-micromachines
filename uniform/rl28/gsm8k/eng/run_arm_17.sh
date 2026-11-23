#!/bin/bash
# usage: GPU=0 bash uniform/rl28/gsm8k/eng/run_arm_17.sh ARM        run / resume 1.7B RL arm ARM (300 GRPO updates) on one GPU.
# gsm_stack/engine/run_arm.sh for the Qwen3-1.7B-Base host (GSM8K uniform rows 2 and 4, REPRODUCE §18). Same recipe as
# P64_sK / CW_sK (GRPO 300 u, 64 prompts x G8, N=4, lr 1e-6, T=1, forget M=1e4, train split A only), host = Qwen3-1.7B-Base:
#   P17_sK   = P64_sK config (RL_ARM=P, S_max 1280, B_max 384), seed K
#   CW17_sK  = CW_sK  config (RL_ARM=A + WPR=1, S_max 2944), seed K; B_max 256 (1.7B trainer state + KV do not fit at 384) and
#              SW_DMARG=40 (writer margin); verifier probe = the 1.7B refit (stk/verifier_probe_L18_q17.pt of the overlay)
# Code: the overlay uniform/rl28/gsm8k/trees/arm17 (rel/v5 engine + the HOST_MODEL patches), frozen per arm under
# $WSM_WORK/trees/arm17_<ARM>. The implant wiring comes from the 0.6B u0 host $CKPT_U0 (--eval_ckpt), the host weights
# from the Hub (--host_ckpt base with HOST_MODEL). Every start resumes from $RL_RESUME_DIR/rlstack_<ARM>.pt (default
# $ARMDIR17/states, written every 2 updates); 3 tries per call (B_max halved after an OOM); call again to continue
# (original runs: one cluster-queue job per start, chained across 12-h limits, on a B200).
# Outputs: $ARMDIR17/logs/rl_<ARM>.out, $ARMDIR17/rl/<ARM>/{steps.jsonl,step_000300/}, eng/out/arm_<ARM>/ (log tail, DONE).
# Env: MEMFRAC (per-process GPU memory cap, packed runs: eng/pack_17.sh), BMAX, FT_BUDGET (micro-batch token budget,
# gradient accumulation only; default 32768), KEEP_STEPS=1 keeps step_000100/200 (deleted by default; only u300 is used).
ARM=$1
[ -n "$ARM" ] || { echo "usage: GPU=0 $0 ARM"; exit 64; }
E="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$E/../../../env.sh"
[ -f "$CKPT_U0/model.safetensors" ] || { echo "no u0 host at CKPT_U0=$CKPT_U0 (REPRODUCE.md section 16.1)"; exit 2; }
export HOST_MODEL=Qwen/Qwen3-1.7B-Base
D=$ARMDIR17
export RL_RESUME_DIR=${RL_RESUME_DIR:-$D/states} RL_RESUME_EVERY=2
case $ARM in
  P17_s*)  CFG=P; HOST="--host_ckpt base"; SEED=${ARM#P17_s};  SMAX=1280; NMAX=4; PR=64; BM=384; XENV="";;
  CW17_s*) CFG=A; HOST="--host_ckpt base"; SEED=${ARM#CW17_s}; SMAX=2944; NMAX=4; PR=64; BM=256; XENV="WPR=1 SW_DMARG=40";;
  *)    echo "unknown arm $ARM"; exit 64;;
esac
H=$WSM_WORK/trees/arm17_$ARM
if [ ! -d $H/stk ]; then
  H=$(cd "$WSM_ROOT" && gs_tree "$E/../trees/arm17" arm17_$ARM) || exit 2
  cp $H/stk/verifier_probe_L18_q17.pt $H/stk/verifier_probe_L18.pt   # gs_tree copied the 0.6B probe
fi
export WIRING_DIR=$H/wiring STK_DIR=$H/stk SAN_DIR=$H/stk DIG_HOOK=$H/stk/rl_hook.py SAN_SMAX=$SMAX C2_NOCOMPACT=1 SER_WS=1
export RL_FASTROLL=1 FAST_TK=1 FAST_COMPACT=1 FAST_OVL=1 FAST_ATT3=1 FLEX_SMALL=1
export RL_FASTTRAIN=1 FT_ATTN=triton FT_CKPT=1 FT_LP=1 FT_DEDUP=1 FT_FUSED_ADAM=1 FT_COMPILE=1 FT_BUDGET=${FT_BUDGET:-32768}
export RL_ASYNC_SAVE=1 RL_ASYNC=0
export TORCHINDUCTOR_CACHE_DIR=$WSM_WORK/inductor17_$ARM
RUN=$D/rl/$ARM; LOGF=$D/logs/rl_$ARM.out; QL=$D/logs/rl_queue.log; O=$E/out/arm_$ARM
mkdir -p $D/logs $D/rd $RUN $RL_RESUME_DIR $TORCHINDUCTOR_CACHE_DIR $O
report() { tail -n 80 $LOGF > $O/log_tail.txt; cp $RUN/steps.jsonl $O/ 2>/dev/null; grep -E "FASTCHECK|FIRST_UPDATE_RATIO|Traceback|rror" $LOGF | head -20 > $O/checks.txt; }
if [ ! -f $RL_RESUME_DIR/rlstack_$ARM.pt ]; then
  if grep -q '"resumed_from"\|"u": [2-9]' $RUN/steps.jsonl 2>/dev/null; then
    echo "arm $ARM has steps but no resume state" | tee -a $QL > $O/log_tail.txt; exit 2; fi
  echo "$(date -u +%FT%TZ) $ARM: fresh start at u0" >> $QL
fi
grep -q '"fastcheck"' $RUN/steps.jsonl 2>/dev/null || export RL_FASTCHECK=${RL_FASTCHECK:-600}
BMAX=${BMAX:-$BM}
cd $WIRING_DIR
for try in 1 2 3; do
  echo "$(date -u +%FT%TZ) start $ARM (RL_ARM=$CFG seed $SEED $XENV, host $HOST_MODEL) B_max $BMAX try$try" >> $QL
  env RL_RUN=$RUN RL_ARM=$CFG SW_NMAX=$NMAX RL_N=4 RL_PROMPTS=$PR RL_G=8 RL_UPDATES=300 RL_SEED=$SEED \
    STK_FORGET_M=10000 $XENV python3 $H/dig_san/tw_dump_lean.py --run_dir $D/rd/rl_$ARM --arm 2 --fused 0 --B_max $BMAX \
    --tok_budget 16384 --eval_ckpt $CKPT_U0 $HOST ${MEMFRAC:+--mem_frac $MEMFRAC} >> $LOGF 2>&1
  rc=$?
  report
  if tail -n 50 $LOGF | grep -q RLDONE; then
    echo "$(date -u +%FT%TZ) done $ARM" >> $QL; touch $O/DONE; rm -f $RL_RESUME_DIR/rlstack_$ARM.pt
    [ -n "$KEEP_STEPS" ] || rm -rf $RUN/step_000100 $RUN/step_000200
    exit 0
  fi
  echo "$(date -u +%FT%TZ) FAIL $ARM rc=$rc (resume)" >> $QL
  if tail -n 200 $LOGF | grep -q "CUDA out of memory"; then BMAX=$((BMAX / 2)); echo "  OOM -> B_max $BMAX" >> $QL; fi
  unset RL_FASTCHECK
  sleep 10
done
exit 3
