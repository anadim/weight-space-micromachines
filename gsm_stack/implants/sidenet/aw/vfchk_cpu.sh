#!/bin/bash
# CPU: VF fuzz (test_vf.py) + in-stream register/pick check (vf_check.py) on the given row dirs
source $WSM_DATA/venv/bin/activate
export HF_HOME=$WSM_DATA/hf HF_HUB_CACHE=$WSM_DATA/hf/hub HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=
STK_DIR=$PWD/stk python3 test_vf.py; echo "TESTVF rc=$?"
for d in "$@"; do echo "== $d"; python3 vf_check.py $d/*.jsonl.gz; done
echo VFCHKDONE
