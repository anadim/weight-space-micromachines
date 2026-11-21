#!/bin/bash
# usage: bash gsm_stack/run_hook.sh NAME TREE HOOK [ENV=VAL ...]
# One GPU job: materialize code tree TREE (engine | v42 | r2 | r3 | rungs/rungN | an overlay folder, see mk_tree.py) into
# $WSM_WORK/trees/NAME and run dig_san/tw_dump_lean.py with DIG_HOOK=stk/HOOK on the arm-A serial-stack config.
# Host: the u0 host $CKPT_U0 (also the implant source) unless XARGS says otherwise, e.g.
#   XARGS="--host_ckpt base"                                   (Qwen3-0.6B-Base, downloaded from the Hub)
#   XARGS="--host_ckpt $ARMDIR/rl/A1/step_000300"              (an RL arm's u300 host)
# Knobs: BMAX (B_max, KV rows), SAN_SMAX (KV length per row), GPU. Output: $WSM_WORK/out/NAME/{log.txt,rd/}.
# Hooks write their own rows where their env says (fe_hook: FE_OUT). Never touches an RL arm's files.
NAME=$1; TREE=$2; HOOK=$3; shift 3
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
[ -f "$CKPT_U0/model.safetensors" ] || { echo "no u0 host at CKPT_U0=$CKPT_U0 (REPRODUCE.md section 16.1)"; exit 2; }
H=$(gs_tree "$TREE" "$NAME") || exit 2
export WIRING_DIR=$H/wiring STK_DIR=$H/stk SAN_DIR=$H/stk DIG_HOOK=$H/stk/$HOOK SAN_SMAX=${SAN_SMAX:-2944} C2_NOCOMPACT=1 SER_WS=1
O=$WSM_WORK/out/$NAME; mkdir -p $O
cd $WIRING_DIR
nvidia-smi -i ${CUDA_VISIBLE_DEVICES%%,*} --query-gpu=name,clocks.sm,clocks.mem,power.draw --format=csv,noheader
env RL_RUN=$O RL_ARM=${RL_ARM:-A} SW_NMAX=4 RL_N=4 RL_PROMPTS=64 RL_G=8 RL_SEED=5 STK_FORGET_M=10000 RL_FASTROLL=${RL_FASTROLL:-1} "$@" \
  python3 $H/dig_san/tw_dump_lean.py --run_dir $O/rd --arm 2 --fused 0 --B_max ${BMAX:-384} --tok_budget 16384 \
  --eval_ckpt $CKPT_U0 $XARGS > $O/log.txt 2>&1
rc=$?
tail -n 40 $O/log.txt
grep -E "^(BENCH|PROF|TKCHK|RL )" $O/log.txt | cut -c1-2000 | tail -n 40
exit $rc
