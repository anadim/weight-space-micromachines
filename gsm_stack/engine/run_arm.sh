#!/bin/bash
# usage: GPU=0 bash gsm_stack/engine/run_arm.sh ARM        run / resume RL arm ARM (300 GRPO updates) on one GPU.
#
# The code runs from a frozen copy of the engine (rel/v5) under $WSM_WORK/trees/arm_<ARM> (ENGINE_TREE=v42 for the
# rel/v4.2 tree of the first arms), so edits to the repo never reach a running arm. Every start resumes from
# $WSM_DATA/states/rlstack_<ARM>.pt (written atomically every RL_RESUME_EVERY=2 updates by rl_hook.py); the script
# retries up to 3 times in one call (halving B_max after an out-of-memory error) and can simply be called again.
# Needs: the u0 host $CKPT_U0 (= a2_u300_s2, also the implant source of every arm), runs/veto_qwen3.pt, data/gsm8k.
#
# Arm table (all G=8, RL_N=4, lr 1e-6, forget M=1e4, 300 updates; SW_NMAX = serial tries per rollout):
#   C      RL_ARM=A,  base host (--host_ckpt base), seed 5, 64 prompts, SW_NMAX=4, S_max 2944, B_max 384
#   A      RL_ARM=A,  u0 host, seed 3, 64 prompts, SW_NMAX=8 (dev vote8), S_max 5696, B_max 192
#   Ar     A-reset: as A + RL_POSRESET=1 (programmatic position reset: NON-COMPLIANT reference arm)
#   A1     A' : RL_ARM=A1 (attempt-level advantage), otherwise as A (seed 3: the original A1 run's seed was not
#          recorded; its replicates A1_s2 / A1_s3 are seeds 4 / 6)
#   A1W_sK / CW_sK  A1 / C + WPR=1 (weight-space position reset, implants/wpr), seed K; the trainer uses reset positions
#   B      RL_ARM=B,  S2F single-attempt control (stackh), u0 host, seed 1, 256 prompts (token-matched), S_max 1280, B_max 384
#   A_s2 / B_s2   second seeds of A / B (seed 4 / 2);  A1_s2 / A1_s3 / A1_s4 / A_s3 / B_s3 / C_s2 / C_s3: seeds 4/6/8/6/3/7/6
#   Bb_sK / Pb_sK  B config from the BASE host / plain RL with no implants (RL_ARM=P) from the base host, seed K
#   A1D_s0 / A1D_s1 / A1D_s2   A1 + diversity term RL_DIV=0.5 (seeds 0 / 1 / 4)
#   C600   C continued: host $ARMDIR/rl/C/step_000300, fresh AdamW + warmup, 300 more updates (u counts from 0), seed 5
#   CW600  CW_s0 continued the same way (host $ARMDIR/rl/CW_s0/step_000300, WPR=1, seed 0)
#   (any other name starting with C / A / B: that config, e.g. smoke tests with RL_UPDATES=2 in a scratch ARMDIR)
# Engine flags (all verified bitwise, KERNEL_NOTES.md): RL_FASTROLL=1 FAST_TK=1 FAST_COMPACT=1 FAST_OVL=1 FAST_ATT3=1
#   (+ defaults FAST_FSKIP=1 FAST_EXACT_PREFILL=1), FLEX_SMALL=1. On the first start, RL_FASTCHECK=600 logs the greedy
#   old-vs-fast loop check. Outputs: $ARMDIR/logs/rl_<ARM>.out, $ARMDIR/rl/<ARM>/steps.jsonl (dev evals every 25
#   updates), $ARMDIR/rl/<ARM>/step_000{100,200,300}/. Env overrides: ARMDIR RL_RESUME_DIR RL_UPDATES RL_PROMPTS BMAX
#   FAST_* RL_FASTCHECK RL_FASTTRAIN FT_* RL_ASYNC_SAVE RL_ASYNC.
ARM=$1
[ -n "$ARM" ] || { echo "usage: GPU=0 $0 ARM"; exit 64; }
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
[ -f "$CKPT_U0/model.safetensors" ] || { echo "no u0 host at CKPT_U0=$CKPT_U0 (REPRODUCE.md section 16.1)"; exit 2; }
export RL_RESUME_DIR=${RL_RESUME_DIR:-$WSM_DATA/states} RL_RESUME_EVERY=${RL_RESUME_EVERY:-2}
case $ARM in
  Bb_s*) CFG=B; HOST="--host_ckpt base"; SEED=${ARM#Bb_s}; SMAX=1280; NMAX=4; PR=256; BM=384; XENV="";;
  Pb_s*) CFG=P; HOST="--host_ckpt base"; SEED=${ARM#Pb_s}; SMAX=1280; NMAX=4; PR=256; BM=384; XENV="";;
  CW_s*) CFG=A; HOST="--host_ckpt base"; SEED=${ARM#CW_s}; SMAX=2944; NMAX=4; PR=64; BM=384; XENV="WPR=1";;
  A1W_s*) CFG=A1; HOST=""; SEED=${ARM#A1W_s}; SMAX=5696; NMAX=8; PR=64; BM=192; XENV="WPR=1";;
  C*)   CFG=A;  HOST="--host_ckpt base"; SEED=5; SMAX=2944; NMAX=4; PR=64;  BM=384; XENV="";;
  A1*)  CFG=A1; HOST=""; SEED=3; SMAX=5696; NMAX=8; PR=64;  BM=192; XENV="";;
  Ar*)  CFG=A;  HOST=""; SEED=3; SMAX=5696; NMAX=8; PR=64;  BM=192; XENV="RL_POSRESET=1";;
  A*)   CFG=A;  HOST=""; SEED=3; SMAX=5696; NMAX=8; PR=64;  BM=192; XENV="";;
  B*)   CFG=B;  HOST=""; SEED=1; SMAX=1280; NMAX=4; PR=256; BM=384; XENV="";;
  *)    echo "unknown arm $ARM"; exit 64;;
esac
case $ARM in A_s2*) SEED=4;; B_s2*) SEED=2;; esac
case $ARM in
  A1_s2) SEED=4;;  A1_s3) SEED=6;;  A_s3) SEED=6;;  B_s3) SEED=3;;  C_s2) SEED=7;;  C_s3) SEED=6;;  A1_s4) SEED=8;;
  A1D_s0) SEED=0; XENV="RL_DIV=0.5";;  A1D_s1) SEED=1; XENV="RL_DIV=0.5";;  A1D_s2) SEED=4; XENV="RL_DIV=0.5";;
  C600) HOST="--host_ckpt $ARMDIR/rl/C/step_000300";;
  CW600) HOST="--host_ckpt $ARMDIR/rl/CW_s0/step_000300"; SEED=0; XENV="WPR=1";;
esac
H=$WSM_WORK/trees/arm_$ARM
[ -d $H/stk ] || H=$(gs_tree ${ENGINE_TREE:-engine} arm_$ARM) || exit 2        # frozen per arm: kept across restarts
export WIRING_DIR=$H/wiring STK_DIR=$H/stk SAN_DIR=$H/stk DIG_HOOK=$H/stk/rl_hook.py SAN_SMAX=$SMAX C2_NOCOMPACT=1 SER_WS=1
export RL_FASTROLL=${RL_FASTROLL:-1} FAST_TK=${FAST_TK:-1} FAST_COMPACT=${FAST_COMPACT:-1} FAST_OVL=${FAST_OVL:-1} FAST_ATT3=${FAST_ATT3:-1} FLEX_SMALL=1
# fast trainer (stk/fasttrain.py + stk/tattn.py); RL_FASTTRAIN=0 = the reference trainer
export RL_FASTTRAIN=${RL_FASTTRAIN:-1} FT_ATTN=${FT_ATTN:-triton} FT_CKPT=${FT_CKPT:-1} FT_LP=${FT_LP:-1} FT_DEDUP=${FT_DEDUP:-1} \
  FT_FUSED_ADAM=${FT_FUSED_ADAM:-1} FT_COMPILE=${FT_COMPILE:-1} FT_BUDGET=${FT_BUDGET:-32768}
export RL_ASYNC_SAVE=${RL_ASYNC_SAVE:-1} RL_ASYNC=${RL_ASYNC:-0}
D=$ARMDIR; RUN=$D/rl/$ARM; LOGF=$D/logs/rl_$ARM.out; QL=$D/logs/rl_queue.log
export TORCHINDUCTOR_CACHE_DIR=$WSM_WORK/inductor_$ARM
mkdir -p $D/logs $D/rd $RUN $RL_RESUME_DIR $TORCHINDUCTOR_CACHE_DIR
if [ ! -f $RL_RESUME_DIR/rlstack_$ARM.pt ]; then
  if grep -q '"resumed_from"\|"u": [2-9]' $RUN/steps.jsonl 2>/dev/null; then       # a started arm lost its state: never restart silently
    echo "arm $ARM has steps but no resume state $RL_RESUME_DIR/rlstack_$ARM.pt" | tee -a $QL; exit 2; fi
  echo "$(date -u +%FT%TZ) $ARM: no resume state: fresh start at u0" >> $QL
fi
grep -q '"fastcheck"' $RUN/steps.jsonl 2>/dev/null || export RL_FASTCHECK=${RL_FASTCHECK:-600}
BMAX=${BMAX:-$BM}
cd $WIRING_DIR
for try in 1 2 3; do
  echo "$(date -u +%FT%TZ) start $ARM (RL_ARM=$CFG seed $SEED $XENV, tree $(cat $H/TREE)) gpu$CUDA_VISIBLE_DEVICES B_max $BMAX try$try" >> $QL
  env RL_RUN=$RUN RL_ARM=$CFG SW_NMAX=$NMAX RL_N=4 RL_PROMPTS=${RL_PROMPTS:-$PR} RL_G=8 RL_UPDATES=${RL_UPDATES:-300} RL_SEED=$SEED \
    STK_FORGET_M=10000 $XENV python3 $H/dig_san/tw_dump_lean.py --run_dir $D/rd/rl_$ARM --arm 2 --fused 0 --B_max $BMAX \
    --tok_budget 16384 --eval_ckpt $CKPT_U0 $HOST >> $LOGF 2>&1
  rc=$?
  if tail -n 50 $LOGF | grep -q RLDONE; then echo "$(date -u +%FT%TZ) done $ARM" >> $QL; exit 0; fi
  echo "$(date -u +%FT%TZ) FAIL $ARM rc=$rc (resume)" >> $QL
  if tail -n 200 $LOGF | grep -q "CUDA out of memory"; then BMAX=$((BMAX / 2)); echo "  OOM -> B_max $BMAX" >> $QL; fi
  unset RL_FASTCHECK
  sleep 10
done
exit 3
