#!/bin/bash
# usage: run_seq.sh MODEL(0.6B|1.7B) "kind:rounds[:noop]" ...   -- runs oracle kinds one after another, one process at a time
# Run from math500/ after gen.py + grade.py + autopass.py produced graded_<model>.json. One GPU (CUDA_VISIBLE_DEVICES).
cd "$(dirname "$0")"
M=$1; m=$(echo $M | tr A-Z a-z); shift
PY=${PY:-python}
for spec in "$@"; do
  IFS=: read k r n <<< "$spec"
  extra=""; suf=""
  [ "$n" = "noop" ] && extra="--noop" && suf="_noop"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY oracle_run.py --model Qwen/Qwen3-$M-Base \
     --graded graded_$m.json --kind $k --rounds $r --bs 24 $extra --out or_${m}_$k$suf.json > log_or_${m}_$k$suf.txt 2>&1
done
echo SEQDONE $M
