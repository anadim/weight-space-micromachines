#!/bin/bash
# run_v2.sh SEEDS [SETS] : S3-serial stack with the v2 verifier + voter (SER_V2=1); same config as the base gen runs.
SEEDS=${1:-7,11}; SETS=${2:-dev,dev2}
export SER_V2=1 SER_V2_W=$PWD/stk/voter2_w.pt
for S in ${SETS//,/ }; do bash gen.sh $S $SEEDS '8:1:1:1:0:T1.0' 128 || exit 1; done
