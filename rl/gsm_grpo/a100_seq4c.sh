#!/usr/bin/env bash
# A100: after a100_after.sh (PID $1: a1_u300_s0 + its test eval): 4cRL s0 -> 4c s1 (B 384: the ifeval job keeps ~9 GB)
cd "$(dirname "$0")"
while kill -0 "$1" 2>/dev/null; do sleep 20; done
bash run4c.sh a4cRL_u300_s0 4cRL 0 384
bash run4c.sh a4c_u300_s1 4c 1 384
