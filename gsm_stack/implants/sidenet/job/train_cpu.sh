#!/bin/bash
# CPU variant of run.sh for train_sn.py (gq --gpu none)
ROOT=$WSM_DATA; source $ROOT/venv/bin/activate
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
mkdir -p out; nproc; "$@"
