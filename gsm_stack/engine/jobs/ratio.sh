#!/bin/bash
# usage: GPU=0 bash gsm_stack/engine/jobs/ratio.sh      teacher-forcing ratio of the first update after a fastroll rollout
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
BMAX=384 XARGS="--host_ckpt base" bash $GS_DIR/run_hook.sh ratio_fs${FS} engine fast_bench_hook.py RL_FASTROLL=0 FAST_TK=0 FAST_TIME=1 BENCH_STEPS=fast,ratio ${FS:+FLEX_SMALL=1}
