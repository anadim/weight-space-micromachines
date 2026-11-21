#!/bin/bash
# gen.sh SET SEEDS SER [BMAX] : S3-serial stack rollouts with full tokens/text, on the B200 node via gq.
SET=$1; SEEDS=$2; SER=$3; BMAX=${4:-128}
H=$PWD; ROOT=$WSM_DATA; MY=$WSM_DATA/implants
source $ROOT/venv/bin/activate
for v in $(env | grep -o '^TRITON_[A-Z_]*PATH'); do unset $v; done
TB=$ROOT/venv/lib/python3.12/site-packages/triton/backends/nvidia/bin
export TRITON_PTXAS_PATH=$TB/ptxas TRITON_CUOBJDUMP_PATH=$TB/cuobjdump TRITON_NVDISASM_PATH=$TB/nvdisasm
export HF_HOME=$ROOT/hf HF_HUB_CACHE=$ROOT/hf/hub HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $MY/triton_cache $MY/inductor $MY/rd
export TRITON_CACHE_DIR=$MY/triton_cache TORCHINDUCTOR_CACHE_DIR=$MY/inductor FLEX_SMALL=1
export WIRING_DIR=$H/wiring STK_DIR=$H/stk SAN_DIR=$H/stk DIG_HOOK=$H/stk/gen_hook.py SAN_SMAX=5696 C2_NOCOMPACT=1 SER_WS=1
for f in $ROOT/gsm_final/code/*.pt; do [ -e $H/stk/$(basename $f) ] || cp $f $H/stk/; done
lnpt() { (cd $1 && find . -name '*.pt') | while read f; do [ -e $2/$f ] || { mkdir -p $(dirname $2/$f); ln -s $1/$f $2/$f; }; done; }
lnpt $ROOT/gsm_sel/code/rl_wiring $WIRING_DIR; lnpt $ROOT/gsm_sel/code/dig_san $H/dig_san
mkdir -p $H/out
cd $WIRING_DIR
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
env STK_OUT=$H/out STK_SET=$SET STK_SER="$SER" STK_SEEDS=$SEEDS python3 $H/dig_san/tw_dump_lean.py --run_dir $MY/rd/gen_${SET}_$$ --arm 2 --fused 0 \
  --B_max $BMAX --tok_budget 2048 --eval_ckpt $ROOT/gsm_sel/ckpt_a2s2 2>&1 | grep -v "^\s*$" | tail -n 400
for f in $H/out/*.jsonl; do gzip -f $f; done
ls -la $H/out
