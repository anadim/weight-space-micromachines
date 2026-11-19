#!/bin/bash
cd "$(dirname "$0")"; source env_gcr.sh; L=runs/handoff.log
echo "$(date) waiting for marker (A100)" >> $L
until grep -q "done a4c_u300_s1" ../gsm_grpo/runs/seq4c.log 2>/dev/null; do sleep 60; done
echo "$(date) marker seen" >> $L
until ! pgrep -f "train2.py" > /dev/null; do sleep 30; done
echo "$(date) no train2.py left -> launch runs_extra_B.txt (free $(nvidia-smi --query-gpu=memory.free --format=csv,noheader))" >> $L
exec bash launch.sh 0 2 36000 "" runs_extra_B.txt
