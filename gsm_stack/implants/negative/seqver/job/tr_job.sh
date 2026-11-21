#!/bin/bash
# tr_job.sh TAG GRID [extra train_sq args]
TAG=$1; GRID=$2; shift 2
python3 train_sq.py --tag $TAG --grid $GRID "$@"
