#!/bin/bash
# run_vf.sh SEEDS [SETS] : S3-serial stack with the vfeat voter (SER_VF=1); same config as the base gen runs.
SEEDS=${1:-7,11}; SETS=${2:-dev,dev2}
export SER_VF=1 SER_VF_W=$PWD/stk/vf_w.pt
for S in ${SETS//,/ }; do bash gen.sh $S $SEEDS '8:1:1:1:0:T1.0' 128 || exit 1; done
