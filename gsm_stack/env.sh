# Shared settings for the gsm_stack launchers (sourced by run_hook.sh, engine/run_arm.sh, eval/fe.sh, eval/jobs/*.sh).
# Every variable can be overridden in the environment.
#   WSM_ROOT    repo root (derived from this file)
#   WSM_DATA    data + run root (default $WSM_ROOT/runs/gsm_stack; not tracked). Layout:
#                 $WSM_DATA/ckpt_a2s2/           the u0 host = RL+calc host a2_u300_s2 (model.safetensors), built by
#                                                rl/gsm_grpo (REPRODUCE.md section 16.1). It is also the implant source of every arm.
#                 $WSM_DATA/gsm_final/rl/ARM/    RL arm outputs: steps.jsonl, step_000100..300/model.safetensors
#                 $WSM_DATA/gsm_final/logs/      arm logs (rl_ARM.out, rl_queue.log)
#                 $WSM_DATA/states/              resume states rlstack_ARM.pt (fp32 host + AdamW, ~7 GB, every 2 updates)
#                 $WSM_DATA/gadgets/             optional fitted gadgets that are not in git (vf_w.pt, sn_w.pt; see COMPLIANCE.md)
#   WSM_WORK    scratch: materialized code trees, eval rows, caches (default $WSM_DATA/work)
#   CKPT_U0     the u0 host checkpoint dir (default $WSM_DATA/ckpt_a2s2)
#   ARMDIR      RL arm root (default $WSM_DATA/gsm_final)
#   WSM_VETO    calculator veto (default $WSM_ROOT/runs/veto_qwen3.pt; python scripts/solve_veto.py --device cuda)
#   GPU         CUDA device index (default 0)
GS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WSM_ROOT=${WSM_ROOT:-$(cd "$GS_DIR/.." && pwd)}
export WSM_DATA=${WSM_DATA:-$WSM_ROOT/runs/gsm_stack}
export WSM_WORK=${WSM_WORK:-$WSM_DATA/work}
export CKPT_U0=${CKPT_U0:-$WSM_DATA/ckpt_a2s2}
export ARMDIR=${ARMDIR:-$WSM_DATA/gsm_final}
export WSM_VETO=${WSM_VETO:-$WSM_ROOT/runs/veto_qwen3.pt}
export DFA_CACHE=${DFA_CACHE:-$WSM_WORK/dfa_tables.pt}          # sanity-DFA tables, built from the tokenizer on first use
export WSM_RESUME_DIR=${WSM_RESUME_DIR:-/dev/shm}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${GPU:-0}}
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$WSM_WORK/triton_cache} TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$WSM_WORK/inductor}
# Triton 3.4 needs its own bundled ptxas. If your image points TRITON_PTXAS_PATH at a newer system CUDA, set
# WSM_TRITON_BUNDLED=1 to use the binaries shipped inside the triton wheel instead.
if [ "${WSM_TRITON_BUNDLED:-0}" = 1 ]; then
  for v in $(env | grep -o '^TRITON_[A-Z_]*PATH'); do unset $v; done
  TB=$(python3 -c "import os, triton; print(os.path.join(os.path.dirname(triton.__file__), 'backends', 'nvidia', 'bin'))")
  export TRITON_PTXAS_PATH=$TB/ptxas TRITON_CUOBJDUMP_PATH=$TB/cuobjdump TRITON_NVDISASM_PATH=$TB/nvdisasm
fi
mkdir -p "$WSM_WORK"

# gs_tree TREE NAME: materialize code tree TREE (engine | v42 | r2 | r3 | rungs/rungN | overlay dir) into
# $WSM_WORK/trees/NAME, add the shipped gadgets, print the path.
gs_tree() {
  local T=$WSM_WORK/trees/$2
  python3 "$GS_DIR/mk_tree.py" "$T" "$1" > /dev/null || return 1
  cp "$WSM_ROOT/gsm_push/final/verifier_probe_L18.pt" "$T/stk/"            # 6 KB L18 verifier probe (shipped)
  ln -sf "$DFA_CACHE" "$T/stk/dfa_tables.pt"; ln -sf "$DFA_CACHE" "$T/dig_san/dfa_tables.pt"
  for g in vf_w.pt sn_w.pt; do [ -f "$WSM_DATA/gadgets/$g" ] && cp "$WSM_DATA/gadgets/$g" "$T/stk/"; done
  echo "$T"
}
