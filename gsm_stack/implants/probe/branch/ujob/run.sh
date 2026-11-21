#!/bin/bash
# run.sh CMD... : node env (as gen/gen.sh) then run CMD in this job dir. e.g. bash run.sh python3 extract.py --sets dev --limit_items 20
H=$PWD; ROOT=$WSM_DATA; MY=$WSM_DATA/implants
source $ROOT/venv/bin/activate
for v in $(env | grep -o '^TRITON_[A-Z_]*PATH'); do unset $v; done
TB=$ROOT/venv/lib/python3.12/site-packages/triton/backends/nvidia/bin
export TRITON_PTXAS_PATH=$TB/ptxas TRITON_CUOBJDUMP_PATH=$TB/cuobjdump TRITON_NVDISASM_PATH=$TB/nvdisasm
export HF_HOME=$ROOT/hf HF_HUB_CACHE=$ROOT/hf/hub HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FLEX_SMALL=1 TRITON_CACHE_DIR=$MY/triton_cache TORCHINDUCTOR_CACHE_DIR=$MY/inductor
mkdir -p $H/out $MY/branch
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv 2>/dev/null
"$@"
