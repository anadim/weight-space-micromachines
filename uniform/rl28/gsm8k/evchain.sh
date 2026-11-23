#!/bin/bash
# [28] row-2/row-4 eval chain of REPRODUCE §18 (1.7B), local sequential version.
# Original runs: a detached driver that waited for each RL pack (eng/pack_17.sh) to reach RLDONE and then sent one
# cluster-queue job per stage (one B200, timeout 2 h each): E4a = CW17_s2 test || hard (+ P17_s2 after test),
# E4b = CW17_s0 test+hard || CW17_s1 test+hard. Here: run after the RL arms are done, on one GPU:
#   CUDA_VISIBLE_DEVICES=0 bash uniform/rl28/gsm8k/evchain.sh
# Rows land in fe/out/fe/*.jsonl; mkrows.py / mkresults.py then read them via FE_ROWS (uniform/items.py).
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -e
PROCS="cw17_s2_test+p17_s2 cw17_s2_hard" E4TAG=a bash "$H/fe/jobE4.sh"
PROCS="cw17_s0_test+cw17_s0_hard cw17_s1_test+cw17_s1_hard" E4TAG=b bash "$H/fe/jobE4.sh"
export FE_ROWS="${FE_ROWS:-$H/fe/out/fe}"
python3 "$H/mkrows.py"
python3 "$H/mkresults.py" > /dev/null
echo "rows.json + RESULTS.md regenerated from $FE_ROWS"
