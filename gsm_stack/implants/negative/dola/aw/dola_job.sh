#!/bin/bash
# dola_job.sh CFG[,CFG...] [SETS=dev,dev2] : lane dola on the rung-4 config (lad4_job.sh sn) + DOLA env.
#   off            : rung-4 sn config, DOLA unset, dev seed 7 (gate-off: must reproduce sidenet/aw/out/sn_dev seed 7 token-identically)
#   dL_A[_B]       : sn + DOLA=L:A[:B], SETS, seeds ${SEEDS:-7,11}   (e.g. d18_0.25_0.1)
CFGS=$1; SETS=${2:-dev,dev2}
source $WSM_DATA/venv/bin/activate
for v in $(env | grep -o '^TRITON_[A-Z_]*PATH'); do unset $v; done
TB=$WSM_DATA/venv/lib/python3.12/site-packages/triton/backends/nvidia/bin
export TRITON_PTXAS_PATH=$TB/ptxas TRITON_CUOBJDUMP_PATH=$TB/cuobjdump TRITON_NVDISASM_PATH=$TB/nvdisasm
mkdir -p $WSM_DATA/implants/triton_cache_aw; export TRITON_CACHE_DIR=$WSM_DATA/implants/triton_cache_aw
export WIRING_DIR=$PWD/wiring STK_DIR=$PWD/stk
STK_DIR=$PWD/stk python3 test_aw.py || exit 3
STK_DIR=$PWD/stk python3 test_dola.py || exit 4
for CFG in ${CFGS//,/ }; do
  X=""; SS=$SETS; SD=${SEEDS:-7,11}
  case $CFG in
    off) SS=dev; SD=7;;
    d*) P=${CFG#d}; X="DOLA=${P//_/:}";;
    *) echo BADCFG $CFG; exit 2;;
  esac
  for SET in ${SS//,/ }; do
    env SAN_SMAX=8192 BMAX=128 bash aw.sh ${CFG}_$SET AW_SETS=$SET AW_SEEDS=$SD \
      SW_NS=24 SW_MAXN=7936 AW_ADAPT=A=0.7/24/4:S=4/0.5:B=3:C=400 WPR=1 AW_VF=1 AW_SN=$PWD/stk/sn_w.pt $X
    ls out/${CFG}_$SET | head -2
  done
done
echo DOLADONE
