#!/bin/bash
# start my GPU-0 queue only after the ablation agent's marker AND an empty GPU 0
cd "$(dirname "$0")"; L=runs/handoff.log
echo "$(date) waiting for marker (h100 GPU0)" >> $L
until grep -q "H100 GPU0 queue empty" ../gsm_grpo/runs/h100_seq300.log 2>/dev/null; do sleep 60; done
echo "$(date) marker seen" >> $L
until [ -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 30; done
echo "$(date) GPU0 empty -> launch runs_extra_A.txt" >> $L
exec bash launch.sh 0 2 34000 "" runs_extra_A.txt
