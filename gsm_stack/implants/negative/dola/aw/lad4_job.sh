#!/bin/bash
# lad4_job.sh CFG[,CFG...] [SETS=dev,dev2] : lane sidenet rung-4 candidate = rung-3 avf config (lad3_job.sh avf) + AW_SN.
#   off : avf config, AW_SN unset, dev seed 7 only (gate-off sanity: must reproduce ladr3 out/avf_dev seed 7 token-identically)
#   snq : avf + AW_SN=stk/sn_w.pt, dev seed 7 only (smoke / exactness check before the 4-seed run)
#   sn  : avf + AW_SN=stk/sn_w.pt, SETS, seeds ${SEEDS:-7,11,13,17}
CFGS=$1; SETS=${2:-dev,dev2}
source $WSM_DATA/venv/bin/activate
for v in $(env | grep -o '^TRITON_[A-Z_]*PATH'); do unset $v; done
TB=$WSM_DATA/venv/lib/python3.12/site-packages/triton/backends/nvidia/bin
export TRITON_PTXAS_PATH=$TB/ptxas TRITON_CUOBJDUMP_PATH=$TB/cuobjdump TRITON_NVDISASM_PATH=$TB/nvdisasm
mkdir -p $WSM_DATA/implants/triton_cache_aw; export TRITON_CACHE_DIR=$WSM_DATA/implants/triton_cache_aw
export WIRING_DIR=$PWD/wiring STK_DIR=$PWD/stk
STK_DIR=$PWD/stk python3 test_aw.py || exit 3
for CFG in ${CFGS//,/ }; do
  X=""; SS=$SETS; SD=${SEEDS:-7,11,13,17}
  case $CFG in
    off) SS=dev; SD=7;;
    snq) SS=dev; SD=7; X="AW_SN=$PWD/stk/sn_w.pt";;
    sn) X="AW_SN=$PWD/stk/sn_w.pt";;
    *) echo BADCFG $CFG; exit 2;;
  esac
  for SET in ${SS//,/ }; do
    env SAN_SMAX=8192 BMAX=128 bash aw.sh ${CFG}_$SET AW_SETS=$SET AW_SEEDS=$SD \
      SW_NS=24 SW_MAXN=7936 AW_ADAPT=A=0.7/24/4:S=4/0.5:B=3:C=400 WPR=1 AW_VF=1 $X
    ls out/${CFG}_$SET | head -2
  done
done
echo LAD4DONE
