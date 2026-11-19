#!/bin/bash
# usage: drvr.sh REPO MODEL VETO PREFIX BS ; retry each arm until its json exists (OOM on shared GPUs)
cd $1; mkdir -p out; M=$2; V=$3; P=$4; BS=${5:-24}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for split in hard test; do for mode in calc plain; do
  out=out/x_${P}_${split}_${mode}.json
  for try in 1 2 3 4 5 6; do
    [ -f $out ] && break
    python transfer/xfamily/code/run_gsm_x.py $out --model $M --split $split --mode $mode --veto $V --bs $BS > logs_x_${P}_${split}_${mode}.try$try 2>&1
    grep -h RESULT logs_x_${P}_${split}_${mode}.try$try || { echo "arm $out try $try failed: $(grep -m1 -E 'Error|CANNOT' logs_x_${P}_${split}_${mode}.try$try | cut -c1-80)"; sleep 60; }
  done
done; done
echo "DRIVER $P DONE"
