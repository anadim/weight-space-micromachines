#!/bin/bash
# usage: SRC=C GPU=0 bash gsm_stack/engine/jobs/train_bench.sh
# Trainer bench on a scratch copy of arm SRC's resume state (never the real arm): 1 rollout, then stk/train_bench.py
# on that batch (reference trainer vs fasttrain variants: time, profile, gradient agreement). env: SRC, TB_* knobs.
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
SRC=${SRC:-C}; ARM=${SRC}T; T=$WSM_WORK/arm_test/TB_$SRC; rm -rf $T; mkdir -p $T/states
cp $WSM_DATA/states/rlstack_$SRC.pt $T/states/rlstack_$ARM.pt || exit 2
ARMDIR=$T RL_RESUME_DIR=$T/states RL_FASTCHECK=0 RL_TRAIN_BENCH=1 bash $GS_DIR/engine/run_arm.sh $ARM; rc=$?
grep -E "^(BENCH|PROF|RL .*RESUMED)|Error|error" $T/logs/rl_$ARM.out | cut -c1-700
exit $rc
