#!/bin/bash
# usage: bash gsm_stack/eval/jobs/rung_job.sh TREE LIST       e.g.  rung_job.sh r2 wpr_rung2.list  |  rung_job.sh r3 wpr_rung3.list
# The rung-2 / rung-3 test jobs: the adaptive-tries CPU fuzz (test_aw.py) on the tree first, then the eval at
# B_max 128 / S_max 8192 (the settings of every rung-2 / rung-3 test number).
#   r2: wpr_rung2.list (rung 2, u0), wpr_r2rl.list (A1W hosts + rung 2), wpr_r2cw.list (CW hosts + rung 2)
#   r3: wpr_rung3.list (rung 3, u0; needs $WSM_DATA/gadgets/vf_w.pt), wpr_r2n24.list (rung 2 + Nmax 24, no VF)
TREE=$1; LIST=$2
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
T=$(gs_tree $TREE fuzz_$TREE) || exit 2
(cd $T/stk && STK_DIR=$PWD python3 test_aw.py) || { echo "test_aw FAILED"; exit 3; }
FE_TREE=$TREE FE_BMAX=128 FE_SMAX=8192 bash $GS_DIR/eval/fe.sh $GS_DIR/eval/lists/$LIST
