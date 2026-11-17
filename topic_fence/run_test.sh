#!/bin/sh
# final TEST evaluation (1.7B host; run from topic_fence/ after the dev pipeline, s3_latent.py and s3b_latent.py):
# generations per condition, then the LLM judge. Math and ML fences in separate processes / files.
cd "$(dirname "$0")"; mkdir -p res
python build_test.py                                    # data/test.jsonl + data/mmlu_ret.jsonl (built once)
C=plain,sys,sys_strong,kw,kw_stop,probe_text,probe_text_stop,probe_latent,monitor
FENCES=math OUT=res/test_q17_gen_math.json python s5_eval.py Qwen/Qwen3-1.7B q17 $C > s5_q17_math.log 2>&1
FENCES=ml   OUT=res/test_q17_gen_ml.json   python s5_eval.py Qwen/Qwen3-1.7B q17 $C > s5_q17_ml.log 2>&1
python s6_judge.py q17 res/test_q17_gen_math.json > s6_q17_math.log 2>&1
python s6_judge.py q17 res/test_q17_gen_ml.json > s6_q17_ml.log 2>&1
