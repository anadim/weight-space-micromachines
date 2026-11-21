#!/bin/bash
# lad3_job.sh CFG[,CFG...] [SETS=dev,dev2] : rung-3 candidates on top of rung 2 (WPR + A+B+C+stop4). One aw.sh process per
# (cfg, set), seeds ${SEEDS:-7,11,13,17} in that process (= aw/aw_job3.sh r2f_SET structure, so streams pair with aw/out/r2f_SET).
#   eq24 : rung-2 config but SW_NS=24 (slot count only; dev seed 7 only; sanity: must reproduce r2f rows)
#   a    : A=0.7/24/4      b : A=0.8/24/4   a32: A=0.7/32/4 (SW_NS=32)   (+vf suffix: AW_VF=1 answer-group voter;
#   cvf: A=0.6/24/4:S=3/0 + AW_VF   c32vf: A=0.6/32/4:S=3/0 + AW_VF (SW_NS=32)  (diff-block lane lead: cheaper stop/extend)
#   +qc suffix: QS=const_mix_b100 constant question-tail steering vector, lane m3qs qs_engine.py; r2vfqc = rung 2 + vf + qc)
CFGS=$1; SETS=${2:-dev,dev2}
source $WSM_DATA/venv/bin/activate
for v in $(env | grep -o '^TRITON_[A-Z_]*PATH'); do unset $v; done
TB=$WSM_DATA/venv/lib/python3.12/site-packages/triton/backends/nvidia/bin
export TRITON_PTXAS_PATH=$TB/ptxas TRITON_CUOBJDUMP_PATH=$TB/cuobjdump TRITON_NVDISASM_PATH=$TB/nvdisasm
mkdir -p $WSM_DATA/implants/triton_cache_aw; export TRITON_CACHE_DIR=$WSM_DATA/implants/triton_cache_aw
export WIRING_DIR=$PWD/wiring STK_DIR=$PWD/stk
STK_DIR=$PWD/stk python3 test_aw.py || exit 3
REST="S=4/0.5:B=3:C=400"
for CFG in ${CFGS//,/ }; do
  case $CFG in
    eq24) A="A=0.7/16/4";; a|avf|avfqc) A="A=0.7/24/4";; b|bvf|bvfqc) A="A=0.8/24/4";;
    a32|a32vf|a32vfqc) A="A=0.7/32/4";; r2vfqc) A="A=0.7/16/4";; cvf) A="A=0.6/24/4";; c32vf) A="A=0.6/32/4";;
    *) echo BADCFG $CFG; exit 2;;
  esac
  REST="S=4/0.5:B=3:C=400"; case $CFG in c*) REST="S=3/0:B=3:C=400";; esac
  NS=24; case $CFG in a32*|c32*) NS=32;; r2*) NS=16;; esac
  X=""; case $CFG in *vf|*vfqc) X="AW_VF=1";; esac
  case $CFG in *qc) X="$X QS=$WSM_DATA/implants/m3qs/side_const_mix_b100.pt";; esac
  SS=$SETS; SD=${SEEDS:-7,11,13,17}; [ $CFG = eq24 ] && { SS=dev; SD=7; }
  for SET in ${SS//,/ }; do
    env SAN_SMAX=8192 BMAX=128 bash aw.sh ${CFG}_$SET AW_SETS=$SET AW_SEEDS=$SD \
      SW_NS=$NS SW_MAXN=7936 AW_ADAPT=$A:$REST WPR=1 $X
    ls out/${CFG}_$SET | head -2
  done
done
echo LAD3DONE
