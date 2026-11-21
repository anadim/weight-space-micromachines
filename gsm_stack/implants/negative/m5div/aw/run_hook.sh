#!/bin/bash
# usage (inside a gq job, cwd = job snapshot of gsm_stack/engine):
#   bash run_hook.sh NAME STKDIR HOOK [ENV=VAL ...]
# Runs dig_san/tw_dump_lean.py with DIG_HOOK=STKDIR/HOOK on the arm-A serial-stack config (a2s2 host unless XARGS says
# otherwise). Scratch run dir under out/NAME (comes back with the job). Never a real arm.
NAME=$1; SD=$2; HOOK=$3; shift 3
H=$PWD; ROOT=$WSM_DATA
source $ROOT/venv/bin/activate
# the node env points triton at the system CUDA 13.4 toolchain (ptxas 13.4 breaks triton 3.4): use triton's bundled one
env | grep -E '^(TRITON|CUDA_HOME|CUDA_PATH|LD_LIBRARY_PATH)' | cut -c1-200
for v in $(env | grep -o '^TRITON_[A-Z_]*PATH'); do unset $v; done
TB=$ROOT/venv/lib/python3.12/site-packages/triton/backends/nvidia/bin
export TRITON_PTXAS_PATH=$TB/ptxas TRITON_CUOBJDUMP_PATH=$TB/cuobjdump TRITON_NVDISASM_PATH=$TB/nvdisasm
export HF_HOME=$ROOT/hf HF_HUB_CACHE=$ROOT/hf/hub HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=$WSM_DATA/triton_cache TORCHINDUCTOR_CACHE_DIR=$WSM_DATA/inductor_bench
export WIRING_DIR=$H/${WIRING:-wiring} STK_DIR=$H/$SD SAN_DIR=$H/$SD DIG_HOOK=$H/$SD/$HOOK SAN_SMAX=${SAN_SMAX:-2944} C2_NOCOMPACT=1 SER_WS=1
# data files of the stack (DFA cache, verifier probe) from the node copy
for f in $WSM_DATA/gsm_final/code/*.pt; do [ -e $H/$SD/$(basename $f) ] || cp $f $H/$SD/; done
# *.pt data files excluded from the local copies (wsmrepo/veto_qwen3.pt, ...): symlinks to the node copies
lnpt() { (cd $1 && find . -name '*.pt') | while read f; do [ -e $2/$f ] || { mkdir -p $(dirname $2/$f); ln -s $1/$f $2/$f; }; done; }
lnpt $ROOT/gsm_sel/code/rl_wiring $WIRING_DIR; lnpt $ROOT/gsm_sel/code/dig_san $H/dig_san
mkdir -p $H/out/$NAME
cd $WIRING_DIR
nvidia-smi -i ${CUDA_VISIBLE_DEVICES:-0} --query-gpu=name,clocks.sm,clocks.mem,power.draw --format=csv,noheader
env RL_RUN=$H/out/$NAME RL_ARM=${RL_ARM:-A} SW_NMAX=4 RL_N=4 RL_PROMPTS=64 RL_G=8 RL_SEED=5 STK_FORGET_M=10000 RL_FASTROLL=${RL_FASTROLL:-1} "$@" \
  python3 $H/dig_san/tw_dump_lean.py --run_dir $H/out/$NAME/rd --arm 2 --fused 0 --B_max ${BMAX:-384} --tok_budget 16384 \
  --eval_ckpt $ROOT/gsm_sel/ckpt_a2s2 $XARGS > $H/out/$NAME/log.txt 2>&1
rc=$?
tail -n 40 $H/out/$NAME/log.txt
grep -E "^(BENCH|PROF|TKCHK|RL )" $H/out/$NAME/log.txt | cut -c1-2000 | tail -n 40
exit $rc
