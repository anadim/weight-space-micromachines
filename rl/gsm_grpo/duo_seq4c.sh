#!/usr/bin/env bash
# A40 machine GPU1 (A40): 4c s0 (B 128) -> arm-3 reruns
cd "$(dirname "$0")"; export CUDA_VISIBLE_DEVICES=1
bash run4c.sh a4c_u300_s0 4c 0 128
export FLEX_SMALL=1; bash abl.sh q_abl_h100b.txt > runs/abl_duo_a3.log 2>&1
