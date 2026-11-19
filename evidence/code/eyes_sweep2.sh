#!/bin/bash
# Polite sequential eyes sweep: waits for item2 to finish, guards on free GPU mem, retries on OOM.
cd "$(dirname "$0")/../../finetuned_phase/eyes_and_finetune"
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-python}
S=${S:-../../evidence/eyes_sweep/results}; mkdir -p $S; SUM=$S/summary.txt; : > $SUM
# wait until the big item2 job is gone
while pgrep -f item2_silent.py >/dev/null; do sleep 30; done
echo "item2 finished; starting eyes sweep $(date)" >> $SUM
guard() { # wait until >=9GB free
  while true; do fm=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$fm" -ge 9000 ] && break; sleep 20; done; }
run() { # n arm seed ep
  local n=$1 arm=$2 seed=$3 ep=$4 mask=1 MM="--machines calc,eye"
  [ $arm = notool ] && { mask=0; MM="--no_mm"; }
  local out=runs/sw_${n}_${arm}_s${seed}
  for try in 1 2 3; do
    guard
    $PY ft.py --model Qwen/Qwen2.5-0.5B --fmt calc --mask_output $mask --lr 1e-5 --epochs $ep --gsm 0 \
      --tok_budget 2048 --data data/eyes_train.jsonl:$n --init_special calc+eye --seed $seed --out $out > $S/ft_${n}_${arm}_s${seed}.log 2>&1
    [ -d $out ] || { echo "retry $try ft $n $arm s$seed (no ckpt)"; sleep 30; continue; }
    guard
    $PY bbh_gen.py $out $S/${n}_${arm}_s${seed}_test.json --data data/eyes_test.jsonl $MM --bs 16 > $S/ev_${n}_${arm}_s${seed}.log 2>&1
    local m=$($PY -c "import json;print(round(json.load(open('$S/${n}_${arm}_s${seed}_test.json'))['macro'],2))" 2>/dev/null)
    [ -n "$m" ] && { echo "n=$n arm=$arm seed=$seed test_macro=$m" | tee -a $SUM; rm -rf $out; return; }
    echo "retry $try eval $n $arm s$seed"; rm -rf $out; sleep 30
  done
  echo "n=$n arm=$arm seed=$seed test_macro=FAILED" | tee -a $SUM
}
for seed in 0 1 2 3; do
  run 8  tool   $seed 200
  run 8  notool $seed 200
done
for seed in 0 1 2 3; do
  run 32 tool   $seed 50
  run 32 notool $seed 50
done
echo "SWEEP2_DONE $(date)" >> $SUM
