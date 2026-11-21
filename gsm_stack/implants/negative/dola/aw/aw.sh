#!/bin/bash
# aw.sh NAME [ENV=VAL ...]  (gq job, cwd = snapshot of gsm_stack/implants/aw/dev_tree): all-weights SerialWD stack (no position reset) on the
# u0 host (ckpt_a2s2), engine = copy of gsm_stack/eval (frozen rel/v4.2 + fe_hook path). Hook stk/aw_hook.py (AW_* env).
# Rows -> out/NAME/*.jsonl.gz (gen/out format). Same env/flags as stack-lane final_eval/fe.sh + run_hook.sh, caches under implant-lane.
NAME=$1; shift
H=$PWD; ROOT=$WSM_DATA; MY=$WSM_DATA/implants
source $ROOT/venv/bin/activate
for v in $(env | grep -o '^TRITON_[A-Z_]*PATH'); do unset $v; done
TB=$ROOT/venv/lib/python3.12/site-packages/triton/backends/nvidia/bin
export TRITON_PTXAS_PATH=$TB/ptxas TRITON_CUOBJDUMP_PATH=$TB/cuobjdump TRITON_NVDISASM_PATH=$TB/nvdisasm
export HF_HOME=$ROOT/hf HF_HUB_CACHE=$ROOT/hf/hub HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $MY/triton_cache_aw $MY/inductor_aw
export TRITON_CACHE_DIR=$MY/triton_cache_aw TORCHINDUCTOR_CACHE_DIR=$MY/inductor_aw
export WIRING_DIR=$H/wiring STK_DIR=$H/stk SAN_DIR=$H/stk DIG_HOOK=$H/stk/aw_hook.py SAN_SMAX=${SAN_SMAX:-5696} C2_NOCOMPACT=1 SER_WS=1
for f in $ROOT/gsm_final/code/*.pt; do [ -e $H/stk/$(basename $f) ] || cp $f $H/stk/; done
lnpt() { (cd $1 && find . -name '*.pt') | while read f; do [ -e $2/$f ] || { mkdir -p $(dirname $2/$f); ln -s $1/$f $2/$f; }; done; }
lnpt $ROOT/gsm_sel/code/rl_wiring $WIRING_DIR; lnpt $ROOT/gsm_sel/code/dig_san $H/dig_san
mkdir -p $H/out/$NAME
cd $WIRING_DIR
nvidia-smi -i ${CUDA_VISIBLE_DEVICES:-0} --query-gpu=name,memory.used --format=csv,noheader
env RL_RUN=$H/out/$NAME/rl RL_ARM=A SW_NMAX=8 RL_N=4 RL_PROMPTS=64 RL_G=8 RL_SEED=5 STK_FORGET_M=10000 \
  RL_FASTROLL=1 FAST_TK=1 FAST_COMPACT=1 FAST_OVL=1 FAST_ATT3=1 FLEX_SMALL=1 AW_OUT=$H/out/$NAME "$@" \
  python3 $H/dig_san/tw_dump_lean.py --run_dir $H/out/$NAME/rd --arm 2 --fused 0 --B_max ${BMAX:-192} --tok_budget 16384 \
  --eval_ckpt $ROOT/gsm_sel/ckpt_a2s2 $XARGS > $H/out/$NAME/log.txt 2>&1
rc=$?
grep -E "^AW |rror|Traceback" $H/out/$NAME/log.txt | cut -c1-400 | tail -n 40
tail -n 5 $H/out/$NAME/log.txt
rm -rf $H/out/$NAME/rd $H/out/$NAME/rl
exit $rc
