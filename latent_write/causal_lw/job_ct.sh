#!/bin/bash
# job_ct.sh : compiled-trigger causal latent write (cw_ct.py prompt + gen) -> out/ct_{prompt,gen}.json, out/log_ct_*.txt
export LW_MODEL=Qwen/Qwen3-0.6B-Base LW_CAP_GB=60 LW_OUT=out PYTHONPATH=$PWD
mkdir -p out
for m in ${MODES:-prompt gen}; do bash run.sh python3 cw_ct.py $m 2>&1 | tee out/log_ct_$m.txt; done
