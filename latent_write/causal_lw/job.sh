#!/bin/bash
# job.sh MODE [ARGS] -> out/cw_MODE.json + out/log_MODE.txt
export LW_MODEL=Qwen/Qwen3-0.6B-Base LW_CAP_GB=60 LW_OUT=out PYTHONPATH=$PWD
mkdir -p out
bash run.sh python3 cw.py "$@" 2>&1 | tee out/log_$1${CW_TAG}.txt
