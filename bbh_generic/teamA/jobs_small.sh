# ~100-row fine-tunes: 3 settings, chosen on DEV only
PY=${PY:-python}; T=track_gen,order_gen
for cfg in "a 2e-5 6" "b 3e-5 10" "c 1e-5 10"; do set -- $cfg
  ( $PY ft.py --data data/train100.jsonl --out runs/ft100$1 --lr $2 --epochs $3 --tok_budget 8000 > logs/ft100$1.log 2>&1 &&
    $PY run_eval.py data/dev.jsonl res/dev_ft100$1_tools.json --model runs/ft100$1 --tools $T > logs/dev_ft100$1_tools.log 2>&1 &&
    $PY run_eval.py data/dev.jsonl res/dev_ft100$1_notools.json --model runs/ft100$1 > logs/dev_ft100$1_notools.log 2>&1 ) &
done
wait
