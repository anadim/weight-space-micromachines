#!/bin/bash
ROOT=$WSM_DATA; source $ROOT/venv/bin/activate
for v in $(env | grep -o '^TRITON_[A-Z_]*PATH'); do unset $v; done
WIRING_DIR=$PWD/wiring STK_DIR=$PWD/stk python3 stk/wpr_fuzz.py
