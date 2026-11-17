# FINAL (frozen) for the ~100-row host ft100a (chosen on DEV: lr 2e-5, 6 epochs) + its age/price ablation
PY=${PY:-python}; M=runs/ft100a; T=track_gen,order_gen
$PY ft.py --data data/train100_ablate.jsonl --out runs/ft100a_abl --lr 2e-5 --epochs 6 --tok_budget 8000 > logs/ft100a_abl.log 2>&1 &
for D in test oos oos2 oos3; do
  $PY run_eval.py data/$D.jsonl res/${D}_ft100a_tools.json --model $M --tools $T > logs/${D}_ft100a_tools.log 2>&1 &
  $PY run_eval.py data/$D.jsonl res/${D}_ft100a_notools.json --model $M > logs/${D}_ft100a_notools.log 2>&1 &
done
$PY run_eval.py judge/variants_123.jsonl res/var123_ft100a_tools.json --model $M --tools $T --bs 12 > logs/var123_ft100a_tools.log 2>&1 &
$PY run_eval.py judge/variants_123.jsonl res/var123_ft100a_notools.json --model $M --bs 12 > logs/var123_ft100a_notools.log 2>&1 &
wait
$PY run_eval.py data/test.jsonl res/test_ft100a_abl_tools.json --model runs/ft100a_abl --tools $T > logs/test_ft100a_abl_tools.log 2>&1 &
$PY run_eval.py data/oos2.jsonl res/oos2_ft100a_abl_tools.json --model runs/ft100a_abl --tools $T > logs/oos2_ft100a_abl_tools.log 2>&1 &
wait
