# after ft_v1: dev v1 + oos + judge variants, FT+tools / FT no tools / base; then the ablation fine-tune
PY=${PY:-python}; M=runs/ft_v1
$PY run_eval.py data_v1/dev.jsonl res/dev1_ftv1_tools.json --model $M --tools track_gen,order_gen > logs/dev1_ftv1_tools.log 2>&1 &
$PY run_eval.py data_v1/dev.jsonl res/dev1_ftv1_notools.json --model $M > logs/dev1_ftv1_notools.log 2>&1 &
$PY run_eval.py data/oos.jsonl res/oos_ftv1_tools.json --model $M --tools track_gen,order_gen > logs/oos_ftv1_tools.log 2>&1 &
$PY run_eval.py judge/variants_1.jsonl res/var1_ftv1_tools.json --model $M --tools track_gen,order_gen --bs 10 > logs/var1_ftv1_tools.log 2>&1 &
$PY run_eval.py judge/variants_1.jsonl res/var1_ftv1_notools.json --model $M --bs 10 > logs/var1_ftv1_notools.log 2>&1 &
$PY run_eval.py judge/variants_1.jsonl res/var1_base.json --bs 10 > logs/var1_base.log 2>&1 &
$PY run_eval.py data/dev.jsonl res/dev2_base_plain.json > logs/dev2_base_plain.log 2>&1 &
wait
