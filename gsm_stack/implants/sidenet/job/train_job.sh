#!/bin/bash
# train_job.sh TAG GRID [extra args] : wait for the extraction outputs, then train_sn.py
R=$WSM_DATA/implants/sidenet
for i in $(seq 1 120); do [ -f $R/trainC/meta.npz ] && [ -f $R/dev2/meta.npz ] && break; sleep 20; done
TAG=$1; GRID=$2; shift 2
python3 train_sn.py --tag $TAG --grid $GRID "$@"
