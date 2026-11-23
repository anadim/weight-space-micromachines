#!/bin/bash
# usage: bash uniform/rl28/gsm8k/fe/run_hook.sh NAME TREE HOOK [ENV=VAL ...]       (one GPU)
# gsm_stack/run_hook.sh for the Qwen3-1.7B-Base host (REPRODUCE §18): HOST_MODEL defaults to Qwen/Qwen3-1.7B-Base (the hooks
# load the host's HF model / tokenizer from it), SW_PAD_NONDEG_OK=1 (1.7B padded rows are distinct, see comp_serial
# _implant_stop), and the stack's verifier probe is the 1.7B refit (stk/verifier_probe_L18_q17.pt of the stk_r2q_17 overlay;
# its stk/stoprow.pt is the 1.7B refit too). The implant wiring still comes from the 0.6B u0 host $CKPT_U0 (--eval_ckpt);
# the host weights come from XARGS="--host_ckpt base | $ARMDIR17/rl/ARM/step_000300".
NAME=$1; TREE=$2; HOOK=$3; shift 3
source "$(dirname "${BASH_SOURCE[0]}")/../../../env.sh"
export HOST_MODEL=${HOST_MODEL:-Qwen/Qwen3-1.7B-Base}
[ "$HOST_MODEL" = Qwen/Qwen3-1.7B-Base ] && export SW_PAD_NONDEG_OK=1
[ -f "$CKPT_U0/model.safetensors" ] || { echo "no u0 host at CKPT_U0=$CKPT_U0 (REPRODUCE.md section 16.1)"; exit 2; }
cd "$WSM_ROOT"
H=$(gs_tree "$TREE" "$NAME") || exit 2
[ -f $H/stk/verifier_probe_L18_q17.pt ] && cp $H/stk/verifier_probe_L18_q17.pt $H/stk/verifier_probe_L18.pt   # gs_tree copied the 0.6B probe
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
