# ft_v3: dev (v2 dev), oos1, oos2 with tools; dev + oos2 without tools
PY=${PY:-python}; M=runs/ft_v3; T=track_gen,order_gen
$PY run_eval.py data/dev.jsonl res/dev_ftv3_tools.json --model $M --tools $T > logs/dev_ftv3_tools.log 2>&1 &
$PY run_eval.py data/oos.jsonl res/oos_ftv3_tools.json --model $M --tools $T > logs/oos_ftv3_tools.log 2>&1 &
$PY run_eval.py data/oos2.jsonl res/oos2_ftv3_tools.json --model $M --tools $T > logs/oos2_ftv3_tools.log 2>&1 &
$PY run_eval.py data/dev.jsonl res/dev_ftv3_notools.json --model $M > logs/dev_ftv3_notools.log 2>&1 &
$PY run_eval.py data/oos2.jsonl res/oos2_ftv3_notools.json --model $M > logs/oos2_ftv3_notools.log 2>&1 &
$PY run_eval.py data/oos.jsonl res/oos_ftv3_notools.json --model $M > logs/oos_ftv3_notools.log 2>&1 &
$PY run_eval.py judge/variants_12.jsonl res/var12_ftv3_tools.json --model $M --tools $T --bs 12 > logs/var12_ftv3_tools.log 2>&1 &
$PY run_eval.py judge/variants_12.jsonl res/var12_ftv3_notools.json --model $M --bs 12 > logs/var12_ftv3_notools.log 2>&1 &
wait
