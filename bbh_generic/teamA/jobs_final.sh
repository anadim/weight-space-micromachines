# FRESH TEST (seed 1), generated at freeze time. Run once.
PY=${PY:-python}; T=track_gen,order_gen
$PY run_eval.py data/test.jsonl res/test_base_plain.json > logs/test_base_plain.log 2>&1 &
$PY run_eval.py data/test.jsonl res/test_ftv3_tools.json --model runs/ft_v3 --tools $T > logs/test_ftv3_tools.log 2>&1 &
$PY run_eval.py data/test.jsonl res/test_ftv3_notools.json --model runs/ft_v3 > logs/test_ftv3_notools.log 2>&1 &
$PY run_eval.py data/test.jsonl res/test_ftv3abl_tools.json --model runs/ft_v3abl --tools $T > logs/test_ftv3abl_tools.log 2>&1 &
wait
