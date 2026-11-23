#!/bin/bash
# IFEval-541 eval (greedy, max_new 2048, implant on + off, idle identity, seeded lm_eval scoring):
#   bash eng/job_eval.sh NAME MODEL HOSTTAG(inst|base) [CKPT.safetensors] [FMT]
#   NAME=u0 -> table rows 1 (plain) and 3 (implant). Results -> out/NAME/eval_NAME.{imp,plain}.score.json + summary.
cd $(dirname $0)/..; C=$PWD; source env.sh
NAME=$1 MODEL=$2 TAG=$3 CK=$4 FMT=${5:-chat}
R=$DATA/eval/$NAME; O=$C/../out/$NAME; mkdir -p $R $O; exec > >(tee -a $R/job.log) 2>&1
nvidia-smi -i ${CUDA_VISIBLE_DEVICES:-0} --query-gpu=name,memory.total --format=csv,noheader; date -u +%FT%TZ
prep_pylib > $R/pylib.log 2>&1
(cd heldout; build_tables)
SD=$DATA/eng/sd_$TAG.pt; mkdir -p $DATA/eng
( flock 9; [ -e $SD ] || (cd heldout && PYTHONPATH=$C/heldout:$C/f14 python ../hostchk/mk_sd.py $SD $( [ $TAG = inst ] && echo Qwen/Qwen3-0.6B || echo Qwen/Qwen3-0.6B-Base )) ) 9> $DATA/eng.lock
X=""; [ -n "$CK" ] && X="--ckpt $CK"
(cd eng && python eval541.py --model $MODEL $X --sd $SD --modes imp,plain --fmt $FMT --out $R/eval_$NAME) || exit 2
for m in imp plain; do
  PYTHONPATH=$PYL python score_reduce.py ifdata/ifeval.jsonl $R/eval_$NAME.$m.jsonl $O/eval_$NAME.$m.score.json \
    --command "rl24/ifeval $NAME $MODEL $m" > /dev/null || exit 3
done
cp $R/eval_$NAME.summary.json $O/; tail -n 40 $R/job.log > $O/log_tail.txt; date -u +%FT%TZ
