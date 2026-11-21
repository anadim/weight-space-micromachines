#!/bin/bash
# run.sh "SER1|SER2|..." [exact] : implants/trunc job; one process per SER group (floor on/off never mixed in a process).
# exact=1 first runs the CPU fuzz test and the forced-path exactness check (first wave = dev items 0..127, seed 7,
# floor path with m=0; compare to gen/out baseline rows). The original path was checked in j23173 (128/128 identical).
IFS='|' read -ra GR <<< "$1"; EX=${2:-0}
if [ "$EX" = 1 ]; then
  ( source $WSM_DATA/venv/bin/activate && python3 test_trunc.py ) 2>&1 | tail -5
  STK_N=128 STK_TAG=_n128 bash gen.sh dev 7 "8:1:1:1:0:T1.0:mp0"
fi
for SER in "${GR[@]}"; do for S in dev dev2; do bash gen.sh $S 7,11 "$SER"; done; done
echo RUNDONE
