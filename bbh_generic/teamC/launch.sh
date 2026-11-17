#!/bin/bash
# usage: launch.sh NAME args...   (runs run_gen.py on GPU 0 in background, log in logs/NAME.log, output out/NAME.json)
cd "$(dirname "$0")"
n=$1; shift
CUDA_VISIBLE_DEVICES=0 nohup python run_gen.py out/$n.json "$@" > logs/$n.log 2>&1 &
