#!/bin/bash
# usage: drvx.sh REPO MODEL VETO PREFIX
cd $1; mkdir -p out; M=$2; V=$3; P=$4
for split in hard test; do for mode in calc plain; do
  python transfer/xfamily/code/run_gsm_x.py out/x_${P}_${split}_${mode}.json --model $M --split $split --mode $mode --veto $V --bs 32 2>&1 | grep -E "RESULT|Error|CANNOT"
done; done
echo "DRIVER $P DONE"
