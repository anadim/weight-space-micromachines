# FINAL (frozen): ft_v4 on everything; ablation on fresh test. Fresh test generated at freeze, run once.
PY=${PY:-python}; M=runs/ft_v4; T=track_gen,order_gen
for D in dev test oos oos2 oos3; do
  $PY run_eval.py data/$D.jsonl res/${D}_ftv4_tools.json --model $M --tools $T > logs/${D}_ftv4_tools.log 2>&1 &
  $PY run_eval.py data/$D.jsonl res/${D}_ftv4_notools.json --model $M > logs/${D}_ftv4_notools.log 2>&1 &
done
wait
$PY run_eval.py judge/variants_123.jsonl res/var123_ftv4_tools.json --model $M --tools $T --bs 12 > logs/var123_ftv4_tools.log 2>&1 &
$PY run_eval.py judge/variants_123.jsonl res/var123_ftv4_notools.json --model $M --bs 12 > logs/var123_ftv4_notools.log 2>&1 &
$PY run_eval.py data/test.jsonl res/test_ftv4abl_tools.json --model runs/ft_v4abl --tools $T > logs/test_ftv4abl_tools.log 2>&1 &
$PY run_eval.py data/oos2.jsonl res/oos2_ftv4abl_tools.json --model runs/ft_v4abl --tools $T > logs/oos2_ftv4abl_tools.log 2>&1 &
wait
