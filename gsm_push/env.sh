# Shared paths for the gsm_push launchers (sourced by every run_*.sh). Override any of them in the environment.
#   WSM_ROOT    repo root (default: derived from this file)
#   WSM_WORK    where runs, logs and caches are written (default: $WSM_ROOT/runs/gsm_push; not tracked)
#   CKPT        host checkpoint dir with model.safetensors. The RL+calc host (Qwen3-0.6B-Base after 300 GRPO updates,
#               arm a2_u300_s2) is NOT released: rebuild it with rl/gsm_grpo (see gsm_push/README.md), or pass
#               XARGS="--host_ckpt base" for the base host.
GP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WSM_ROOT=${WSM_ROOT:-$(cd "$GP_DIR/.." && pwd)}
export WSM_WORK=${WSM_WORK:-$WSM_ROOT/runs/gsm_push}
export WIRING_DIR=${WIRING_DIR:-$WSM_ROOT/rl/wiring}           # engine, fastqwen, implant, common (RL wiring engine)
export HARNESS=${HARNESS:-$GP_DIR/harness/tw_dump_lean.py}     # lean train_wiring.py driver that exec's $DIG_HOOK
export STK_DIR=${STK_DIR:-$GP_DIR/final}                       # the final implant stack (code + verifier probe)
export SAN_DIR=${SAN_DIR:-$STK_DIR}                            # san_engine.py lives next to the stack code
export DFA_CACHE=${DFA_CACHE:-$WSM_WORK/dfa_tables.pt}         # sanity-DFA tables, built from the tokenizer on first use
export WSM_RESUME_DIR=${WSM_RESUME_DIR:-/dev/shm}              # RAM-disk resume files for RL jobs
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$WSM_WORK"
