#!/usr/bin/env bash
# after the running a2_lr1e-6_s1 (PID $1): the remaining lr 1e-6 seeds on this A100, then the one-shot final evals
cd "$(dirname "$0")"
while kill -0 "$1" 2>/dev/null; do sleep 30; done
bash abl.sh q_abl_a100c.txt >> runs/abl_a100.log 2>&1
bash final_pod.sh qf_a100.txt > runs/final_a100.log 2>&1
