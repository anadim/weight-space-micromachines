#!/bin/bash
cd "$(dirname "$0")"; L=runs/handoff.log; S=../gsm_grpo/runs/long/a4c_u300_s0/steps.jsonl
echo "$(date) waiting for marker (A40 machine GPU1)" >> $L
until python3 -c "
import json,sys
u=max([json.loads(l).get('u',0) for l in open('$S') if l.strip().startswith('{')] or [0])
sys.exit(0 if u>=299 else 1)" 2>/dev/null; do sleep 60; done
echo "$(date) marker seen (u299)" >> $L
until [ -z "$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 30; done
echo "$(date) GPU1 empty -> launch runs_a40_g1.txt" >> $L
exec bash launch.sh 1 1 25000 "FLEX_SMALL=1" runs_a40_g1.txt
