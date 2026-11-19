#!/usr/bin/env bash
# A40 machine GPU1: stop arm 4 s0 at u300 (checkpoint), then its finals.  $1 = PID of the trainer
cd "$(dirname "$0")"; export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1
bash stop_at.sh $1 runs/long/a4_u500_s0 300 ckpt >> runs/duo_seq300.log
while kill -0 "$1" 2>/dev/null; do sleep 10; done
printf 'D_a4_s0_u300 runs/long/a4_u500_s0/step_000300 4\nD_base_a4 base 4\nD_base_calc base 2\n' > qf_duo_long.txt
FE_ARGS="--B_max 128" bash final_pod.sh qf_duo_long.txt > runs/final_duo_long.log 2>&1
