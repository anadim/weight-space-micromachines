#!/bin/bash
# CPU only, ~10 s. Needs torch + transformers and Qwen/Qwen3-0.6B-Base from the HF Hub (HF_HOME as you like).
# Original run: one CPU job via a cluster queue (no GPU). Run from anywhere: bash checks/sourced/vocabjob/run.sh
set -e
D=$(cd "$(dirname "$0")" && pwd); ROOT=${WSM_ROOT:-$(cd "$D/../../.." && pwd)}
cd "$ROOT"; mkdir -p "$D/res"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
PYTHONPATH="$ROOT" python3 "$D/vocab_reader.py" Qwen/Qwen3-0.6B-Base 2>&1 | tee "$D/res/run06.log"
# optional, 1.7B host: PYTHONPATH="$ROOT" python3 "$D/vocab_reader.py" Qwen/Qwen3-1.7B-Base | tee "$D/res/run17.log"
