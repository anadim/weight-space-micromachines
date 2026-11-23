# Shared settings for the uniform-row launchers (REPRODUCE §17 / §18). Sources gsm_stack/env.sh (WSM_ROOT, WSM_DATA,
# WSM_WORK, CKPT_U0, ARMDIR, WSM_VETO, gs_tree, triton) and rl/gsm_grpo/env_local.sh (your venv), then sets:
#   FT2_HOST   the BBH call-FT host ft_int_v2 (REPRODUCE §6; not released; default bbh_generic/integrate/runs/ft_int_v2)
#   ARMDIR17   1.7B GSM8K RL arm root: $ARMDIR17/rl/ARM/step_000300 (default $WSM_DATA/rl28g)
#   RL24_DATA  0.6B BBH / IFEval RL runs + caches (default $WSM_DATA/rl24)
#   RL28_DATA  1.7B BBH / IFEval RL runs + caches (default $WSM_DATA/rl28)
#   BBH_DIR / BBH_COT  official BBH task files / 3-shot CoT prompts (default data/bbh/{data,cot}; data/download_bbh.sh)
UNI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$UNI_DIR/../gsm_stack/env.sh"
[ -f "$WSM_ROOT/rl/gsm_grpo/env_local.sh" ] && source "$WSM_ROOT/rl/gsm_grpo/env_local.sh"
export FT2_HOST=${FT2_HOST:-$WSM_ROOT/bbh_generic/integrate/runs/ft_int_v2}
export ARMDIR17=${ARMDIR17:-$WSM_DATA/rl28g}
export RL24_DATA=${RL24_DATA:-$WSM_DATA/rl24} RL28_DATA=${RL28_DATA:-$WSM_DATA/rl28}
export BBH_DIR=${BBH_DIR:-$WSM_ROOT/data/bbh/data} BBH_COT=${BBH_COT:-$WSM_ROOT/data/bbh/cot}
export PYTHONUNBUFFERED=1
