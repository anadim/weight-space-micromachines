#!/bin/bash
# lad_job2.sh CFG : candidate rung on top of rung 2 (WPR + A+B+C+stop4, _wpr_ar fixed), seeds 7,11,13,17, one process per set
# (same process structure as aw/aw_job3.sh r2f_SET, so seed streams line up with the fixed rung-2 reference).
#   CFG=k1 : WPR_K=1 (fp32-composed rotation row); k2 : WPR_K=2 (table row at pos-off); vf : AW_VF=1 (answer-group voter); k1vf/k2vf : both.
CFG=$1; SETS=${2:-dev,dev2}
source $WSM_DATA/venv/bin/activate
for v in $(env | grep -o '^TRITON_[A-Z_]*PATH'); do unset $v; done
TB=$WSM_DATA/venv/lib/python3.12/site-packages/triton/backends/nvidia/bin
export TRITON_PTXAS_PATH=$TB/ptxas TRITON_CUOBJDUMP_PATH=$TB/cuobjdump TRITON_NVDISASM_PATH=$TB/nvdisasm
mkdir -p $WSM_DATA/implants/triton_cache_aw; export TRITON_CACHE_DIR=$WSM_DATA/implants/triton_cache_aw
export WIRING_DIR=$PWD/wiring STK_DIR=$PWD/stk
STK_DIR=$PWD/stk python3 test_aw.py || exit 3
STK_DIR=$PWD/stk python3 test_vf.py || exit 5
R2="SW_NS=16 SW_MAXN=7936 AW_ADAPT=A=0.7/16/4:S=4/0.5:B=3:C=400 WPR=1"
case $CFG in k1) X="WPR_K=1";; k2) X="WPR_K=2";; vf) X="AW_VF=1";; k1vf) X="WPR_K=1 AW_VF=1";; k2vf) X="WPR_K=2 AW_VF=1";; *) exit 2;; esac
for SET in ${SETS//,/ }; do
  env SAN_SMAX=8192 BMAX=128 bash aw.sh ${CFG}_$SET AW_SETS=$SET AW_SEEDS=${SEEDS:-7,11,13,17} $R2 $X
  case $CFG in *vf) (export HF_HOME=$WSM_DATA/hf HF_HUB_CACHE=$WSM_DATA/hf/hub HF_HUB_OFFLINE=1
                     python3 vf_check.py out/${CFG}_$SET/*.jsonl.gz);; esac
done
echo LADJOB2DONE
