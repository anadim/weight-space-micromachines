#!/bin/bash
# usage: U=2 GPU=0 bash gsm_stack/engine/jobs/arm_smoke.sh NAME
# Smoke test: fresh arm NAME (config from run_arm.sh's table by name prefix) in a scratch ARMDIR; RL_UPDATES=${U:-0}
# (0: only the u0 dev eval). SW_DEVDUMP dumps the dev-eval registers. Prints FIRST_UPDATE_RATIO (should be ~1.000).
NAME=$1
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
T=$WSM_WORK/arm_test/$NAME; rm -rf $T; mkdir -p $T/states
export SW_DEVDUMP=$T/dev
ARMDIR=$T RL_RESUME_DIR=$T/states RL_UPDATES=${U:-0} RL_FASTCHECK=${FC:-0} bash $GS_DIR/engine/run_arm.sh $NAME; rc=$?
echo "rc=$rc"; grep -E "^(RL |RLDONE|FIRST_UPDATE|Traceback)|Error" $T/logs/rl_$NAME.out | cut -c1-900
exit $rc
