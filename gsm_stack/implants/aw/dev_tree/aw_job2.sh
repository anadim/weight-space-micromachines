#!/bin/bash
# aw_job2.sh SETS : rung 2 = WPR=1 (weight-space position reset, lane wpr) + A+B+C+stop4 (aw_adapt), 4 seeds. CPU fuzz first.
SETS=$1
source $WSM_DATA/venv/bin/activate
STK_DIR=$PWD/stk python3 test_aw.py || exit 3
env SAN_SMAX=8192 BMAX=128 bash aw.sh r2_$(echo $SETS | tr , _) AW_SETS=$SETS AW_SEEDS=${SEEDS:-7,11,13,17} WPR=1 SW_NS=16 SW_MAXN=7936 \
  AW_ADAPT=${SPEC:-A=0.7/16/4:S=4/0.5:B=3:C=400}
