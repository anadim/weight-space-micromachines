#!/bin/bash
# usage: bbhx.sh MODEL VETO PREFIX ; base vs +TeamC (frozen, passive, oe=veto) on 3 tool + 4 no-tool BBH tasks
M=$1; V=$2; P=$3; X=${XF:-$(cd "$(dirname "$0")/../../.." && pwd)}  # repo root
export MODEL=$M WSM_REPO=$X PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $X; mkdir -p out/xfamily_bbh
T=multistep_arithmetic_two,boolean_expressions,object_counting,date_understanding,sports_understanding,causal_judgement,hyperbaton
for tools in C none; do
  out=out/xfamily_bbh/${P}_${tools}.json
  for try in 1 2 3; do
    [ -f $out ] && [ "$(python3 -c "import json;print(len(json.load(open('$out'))['per_task']))")" = 7 ] && break
    python transfer/xfamily/code/run27x.py $out --model $M --tools $tools --veto $V --tasks $T --max_new 400 --bs 32 --data data/bbh/data --cot data/bbh/cot > out/xfamily_bbh/${P}_${tools}.log 2>&1
    grep -E "MEAN|Error|SANITIZE" out/xfamily_bbh/${P}_${tools}.log | tail -2
  done
done
echo "BBH $P DONE"
