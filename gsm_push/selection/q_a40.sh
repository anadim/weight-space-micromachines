#!/bin/bash
# A40 queue (after 08:30 CT, when the RL replicates are done): waits until GPU g has >= 30 GB free, then runs.
cd "$(dirname "$0")"
g=$1; name=$2; jobs=$3
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g)" -gt 30000 ]; do sleep 30; done
./run_sel.sh $name $g 3000 SEL_JOBS=$jobs -- --B_max 192 --tok_budget 2048 --mem_frac 0.6
