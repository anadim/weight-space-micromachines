#!/bin/bash
# Re-run every cell of the compiled latent write (one GPU). Run from anywhere; outputs go to latent_write/out/
# (override with LW_OUT). Env: LW_DEV (default cuda:0), LW_CAP_GB (per-process GPU memory cap, default 12),
# LW_MODEL (default Qwen/Qwen3-0.6B-Base). HF_HOME as usual for the Hugging Face cache.
set -e
cd "$(dirname "$0")"
mkdir -p out
export LW_MODEL=Qwen/Qwen3-0.6B-Base
python run.py lost mid 0,2,4,8,10,12,14,16,18,20 > out/log_lost_mid_0.6B.txt 2>&1
for t in "muladd mid" "muladd hard" "mulsub mid"; do
  python run.py $t 0,2,4,8,10,12,14,16,18,20,22 > "out/log_${t// /_}_0.6B.txt" 2>&1
done
export LW_MODEL=Qwen/Qwen3-1.7B-Base LW_CAP_GB=${LW_CAP_GB_17:-14}
for t in "muladd mid" "muladd hard"; do
  python run.py $t 0,2,4,8,10,12,14,16,18 > "out/log_${t// /_}_1.7B.txt" 2>&1
done
python summarize.py out > out/summary.md
python plot.py Qwen3-0.6B-Base out
