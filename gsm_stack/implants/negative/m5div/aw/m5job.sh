#!/bin/bash
# m5div job: node fuzz; off-gate identity (patched stack, M5DIV unset, dev2 seed 7 first in process == aw/out/r1f_dev2 seed 7);
# then implant configs. One process per (config, set), seeds in r1f order so seed 7 is first in the process as in r1f.
# usage: SEEDS=7,11 [OFF=1] bash m5job.sh CFG...    (CFG = beta/P/mode[/tau])
source $WSM_DATA/venv/bin/activate
STK_DIR=$PWD/stk python3 test_m5.py || exit 3
[ -n "$OFF" ] && bash aw.sh off_dev2 AW_SETS=dev2 AW_SEEDS=7 WPR=1
for c in "$@"; do t=$(echo $c | tr / _); for st in dev dev2; do bash aw.sh m5_${t}_$st AW_SETS=$st AW_SEEDS=${SEEDS:-7,11} WPR=1 M5DIV=$c; done; done
