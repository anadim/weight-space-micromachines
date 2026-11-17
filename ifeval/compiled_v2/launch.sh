#!/bin/bash
# Held-out run (odd IFEval keys, 299 prompts), 4 shards on one GPU: compiled implant, same-harness baseline, Python reference.
# Run from ifeval/compiled_v2 (needs ../data/ifeval.jsonl from ../data/prepare_ifeval.py and dirs.pt).
cd "$(dirname "$0")"; mkdir -p runs
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
N=4
for i in 0 1 2 3; do
  python gen_c.py runs/odd_comp_s$i.jsonl --keys odd --bs 16 --shard $i/$N > runs/odd_comp_s$i.log 2>&1 &
  python gen.py runs/odd_none_s$i.jsonl --keys odd --rules none --bs 16 --shard $i/$N > runs/odd_none_s$i.log 2>&1 &
  python gen.py runs/odd_pyref_s$i.jsonl --keys odd --rules no_comma,lower,quote --vscale upper=2 --subset 0 --bs 16 --shard $i/$N > runs/odd_pyref_s$i.log 2>&1 &
done
wait
python merge.py odd_comp odd_none odd_pyref
python score2.py runs/odd_none.jsonl runs/odd_comp.jsonl runs/odd_pyref.jsonl --keys odd
python analyze.py
