#!/bin/bash
# Build the full runnable code tree for job.sh:  bash mk_code.sh [TARGET]   (default TARGET = ./code, next to this script)
# The tree the runs used = the release's rl/gsm_grpo + micromachines, plus the files shipped in ./code/rl/gsm_grpo (new or
# modified, see README.md), plus the data and the calculator veto vector:
#   data/gsm8k/{train,test,train_scaled}.jsonl, shots_v1.json   python3 $WSM_ROOT/data/prepare_gsm.py
#   data/gsmhard/test.jsonl                                     (same script)
#   data/gsmplus/testmini.jsonl                                 python3 $WSM_ROOT/uniform/final_eval/data/prepare_gsmplus.py
#                                                               (or GSMPLUS=/path/to/testmini.jsonl)
#   runs/veto_qwen3.pt                                          python3 $WSM_ROOT/scripts/solve_veto.py (or WSM_VETO=...)
# The copied data and veto are checked against code_data.sha256 (the files the runs used).
set -e
H=$(cd "$(dirname "$0")" && pwd)
R=$H; while [ "$R" != / ] && [ ! -f "$R/rl/gsm_grpo/train2.py" ]; do R=$(dirname "$R"); done
WSM_ROOT=${WSM_ROOT:-$R}
T=${1:-$H/code}
OV=$H/code/rl/gsm_grpo
GSMPLUS=${GSMPLUS:-$WSM_ROOT/data/gsmplus/testmini.jsonl}
WSM_VETO=${WSM_VETO:-$WSM_ROOT/runs/veto_qwen3.pt}
mkdir -p "$T/rl/gsm_grpo" "$T/data/gsm8k" "$T/data/gsmhard" "$T/data/gsmplus" "$T/runs"
if [ "$(cd "$T" && pwd)" = "$H/code" ]; then
  cp -rn "$WSM_ROOT/rl/gsm_grpo/." "$T/rl/gsm_grpo/"          # keep the shipped overlay files
else
  cp -r "$WSM_ROOT/rl/gsm_grpo/." "$T/rl/gsm_grpo/"; cp -r "$OV/." "$T/rl/gsm_grpo/"
fi
rm -rf "$T/micromachines"; cp -r "$WSM_ROOT/micromachines" "$T/micromachines"
miss=0
cpf() { if [ -f "$1" ]; then cp "$1" "$2"; else echo "MISSING $1  ($3)"; miss=1; fi; }
for f in train test train_scaled; do cpf "$WSM_ROOT/data/gsm8k/$f.jsonl" "$T/data/gsm8k/" "python3 data/prepare_gsm.py"; done
cpf "$WSM_ROOT/data/gsm8k/shots_v1.json" "$T/data/gsm8k/" "python3 data/prepare_gsm.py"
cpf "$WSM_ROOT/data/gsmhard/test.jsonl" "$T/data/gsmhard/" "python3 data/prepare_gsm.py"
cpf "$GSMPLUS" "$T/data/gsmplus/testmini.jsonl" "python3 uniform/final_eval/data/prepare_gsmplus.py"
cpf "$WSM_VETO" "$T/runs/veto_qwen3.pt" "python3 scripts/solve_veto.py"
find "$T" -name __pycache__ -prune -exec rm -rf {} +
if [ $miss = 0 ]; then
  (cd "$T" && sha256sum --quiet -c "$H/code_data.sha256") || { echo "data/veto differ from code_data.sha256"; exit 3; }
  echo "data + veto match code_data.sha256"
fi
echo "built $T"; exit $miss
