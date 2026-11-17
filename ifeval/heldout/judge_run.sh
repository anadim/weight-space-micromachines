#!/bin/bash
# run a judge's variant items: plain and implant generations + official scoring (one GPU)
# usage: ./judge_run.sh judge/TOOL_variants.jsonl TAG      (run from ifeval/heldout)
cd "$(dirname "$0")"; mkdir -p runs; export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} VECS=${VECS:-vecs2_forcefix.pt}; PY=${PY:-python}
$PY run.py $1 runs/j_$2_plain.jsonl --mode plain --bs 16 > runs/j_$2_plain.log 2>&1 &
$PY run.py $1 runs/j_$2_imp.jsonl --mode implant --bs 16 > runs/j_$2_imp.log 2>&1 &
wait
$PY score.py $1 runs/j_$2_plain.jsonl runs/j_$2_imp.jsonl 2>&1 | grep -v -i warn
