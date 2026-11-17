#!/bin/bash
# usage: ./queue.sh NPAR "DATA OUT MODE SHARD" ...   (run from ifeval/heldout; one GPU)
# e.g.   ./queue.sh 4 "fresh.jsonl fresh_plain.s0.jsonl plain 0/6" "fresh.jsonl fresh_imp.s0.jsonl implant 0/6" ...
# DATA is a file in data/ (fresh.jsonl, ood.jsonl, dev.jsonl, ..., or ifeval.jsonl -> link ../data/ifeval.jsonl into data/).
NPAR=$1; shift
cd "$(dirname "$0")"; mkdir -p runs
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PY=${PY:-python}
export VECS=${VECS:-vecs2_forcefix.pt}          # the frozen circuit reads only the force_fix rows of this file
for j in "$@"; do
  set -- $j
  while [ $(jobs -rp | wc -l) -ge $NPAR ]; do sleep 5; done
  [ -f runs/$2 ] && continue; echo "start $j"; $PY run.py data/$1 runs/$2 --mode $3 --shard $4 --bs 24 > runs/$2.log 2>&1 &
  sleep 2
done
wait; echo ALLDONE
