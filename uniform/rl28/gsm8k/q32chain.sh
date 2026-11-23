#!/bin/bash
# [32](b)/(c) on 1.7B (REPRODUCE §18), local sequential version. Needs rows 2 and 4 (evchain.sh) for mk32.py's N.
# Original runs: two cluster-queue jobs on one B200 each (timeout 3 h): Q32c (eval seeds 11/13 of rows 1 and 3; it was sent
# early by a separate one-shot script with the same PROCS) and Q32b (matched-compute SCH lists + the row-1 seed lists).
#   CUDA_VISIBLE_DEVICES=0 bash uniform/rl28/gsm8k/q32chain.sh
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -e
export FE_ROWS="${FE_ROWS:-$H/fe/out/fe}"   # your rows first; shipped bundles fill the rest (uniform/items.py)
python3 "$H/mk32.py"    # -> fe/lists/q32b_{test,hard}.list (the shipped lists are its output for the original rows)
PROCS="q32c_h+q32c_r3_s11_test+q32c_r3_s11_hard q32c_r3_s13_test+q32c_r3_s13_hard" E4TAG=Q32c_ bash "$H/fe/jobE4.sh"
PROCS="q32b_test q32b_hard" E4TAG=Q32b_ bash "$H/fe/jobE4.sh"
python3 "$H/mkrows.py"
python3 "$H/mkresults.py" > /dev/null
