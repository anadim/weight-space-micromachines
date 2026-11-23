#!/bin/bash
# C35 jobs (arm 1 plain RL, arm 2 RL + calculator implant, arm X RL + text <calc> tool).
# One job per GPU:  CUDA_VISIBLE_DEVICES=0 bash job.sh MODE ...
#   check                          text-tool parity check (check_tool.py -> res/check_tool.json)
#   train ARM SEED NAME [args]     ARM in 1 | 2 | X; 300 updates -> $WSM_DATA/runs/NAME
#   eval  ARM CKPT NAME [args]     CKPT = base | a run NAME (its step_000300) | an absolute checkpoint dir;
#                                  GSM8K test / GSM-Hard / GSM-Plus testmini, eval seeds 7/11/13 -> $WSM_DATA/eval/NAME
# The exact per-seed command lines are in commands.sh. Code = ./code (build it first: bash mk_code.sh).
# Per-run outputs (steps.jsonl, config.json, eval *.json*) are copied to ./out; the reducer
# (../calcimp/mk_res.py, see README.md) reduces them to the text-free res/ form that the analysis scripts read.
set -o pipefail
MODE=$1; shift
H=$(cd "$(dirname "$0")" && pwd)
export WSM_ROOT=${WSM_ROOT:-$(cd "$H/../.." && pwd)}
export WSM_DATA=${WSM_DATA:-$WSM_ROOT/runs/tooltext}
source "$WSM_ROOT/rl/gsm_grpo/env_local.sh"
if [ "${WSM_TRITON_BUNDLED:-0}" = 1 ]; then   # use the ptxas bundled in the triton wheel (see gsm_stack/env.sh)
  for v in $(env | grep -o '^TRITON_[A-Z_]*PATH'); do unset $v; done
  TB=$(python3 -c "import os, triton; print(os.path.join(os.path.dirname(triton.__file__), 'backends', 'nvidia', 'bin'))")
  export TRITON_PTXAS_PATH=$TB/ptxas TRITON_CUOBJDUMP_PATH=$TB/cuobjdump TRITON_NVDISASM_PATH=$TB/nvdisasm
fi
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$WSM_DATA/cache/triton} TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$WSM_DATA/cache/inductor}
# B200: the default flex-attention backward tiles fail in triton make_ttgir; small tiles as in every RL arm of the paper
export FLEX_SMALL=1
C=$H/code
[ -f "$C/rl/gsm_grpo/train2.py" ] && [ -f "$C/data/gsm8k/shots_v1.json" ] || { echo "build the code tree first: bash $H/mk_code.sh"; exit 2; }
export WSM_ROOT=$C   # the code reads data/, micromachines/ and runs/veto_qwen3.pt relative to this root
mkdir -p "$H/out"; RD=$WSM_DATA/runs; ED=$WSM_DATA/eval; mkdir -p "$RD" "$ED"
echo "$(date -u +%FT%TZ) $MODE $* gpu $CUDA_VISIBLE_DEVICES" | tee -a "$H/out/job.log"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> "$H/out/job.log" 2>/dev/null
cd "$C/rl/gsm_grpo"
case $MODE in
  check)
    CHK_OUT=$H/out/check_tool.json python3 check_tool.py 2>&1 | tee "$H/out/check_tool.log"; RC=$? ;;
  train)
    ARM=$1; SEED=$2; NAME=$3; shift 3
    python3 train2.py --arm $ARM --seed $SEED --run_dir $RD/$NAME --updates 300 --prompts 64 --G 8 --T 1.0 --max_new 320 \
      --lr 1e-6 --kl 0 --loss drgrpo --clip 0.2 --cispo_max 4 --tok_budget 16384 --warmup 5 --B_max 512 --S_max 544 --fused 1 \
      --dev_every 25 --dev_n 800 --ckpt_every 100 "$@" > "$H/out/$NAME.train.log" 2>&1
    RC=$?; echo "rc=$RC" >> "$H/out/$NAME.train.log"
    cp $RD/$NAME/steps.jsonl "$H/out/$NAME.steps.jsonl"; cp $RD/$NAME/config.json "$H/out/$NAME.config.json" ;;
  eval)
    ARM=$1; CK=$2; NAME=$3; shift 3
    case $CK in base|/*) ;; *) CK=$RD/$CK/step_000300 ;; esac   # run name -> its u300 checkpoint
    python3 fe_tool.py --arm $ARM --ckpt $CK --out $ED/$NAME "$@" > "$H/out/$NAME.eval.log" 2>&1
    RC=$?; echo "rc=$RC" >> "$H/out/$NAME.eval.log"
    mkdir -p "$H/out/$NAME"; cp $ED/$NAME/*.json* "$H/out/$NAME/" ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
echo "$(date -u +%FT%TZ) done $MODE $* rc=${RC:-0}" | tee -a "$H/out/job.log"
exit ${RC:-0}
