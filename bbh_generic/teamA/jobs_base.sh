# frozen base on the fresh test, oos3 and all judge variants (GPU 1)
for D in test oos3; do CUDA_VISIBLE_DEVICES=1 python run_eval.py data/$D.jsonl res/${D}_base_plain.json > logs/${D}_base_plain.log 2>&1; done
CUDA_VISIBLE_DEVICES=1 python run_eval.py judge/variants_123.jsonl res/var123_base.json --bs 12 > logs/var123_base.log 2>&1
