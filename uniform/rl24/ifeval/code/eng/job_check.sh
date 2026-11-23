#!/bin/bash
# fast-engine exactness/speed check: bash eng/job_check.sh NAME MODEL HOSTTAG(inst|base) FMT [N] [MAX_NEW]
cd $(dirname $0)/..; C=$PWD; source env.sh
NAME=$1 MODEL=$2 TAG=$3 FMT=$4 N=${5:-128} MN=${6:-1024}
O=$DATA/eng/$NAME; mkdir -p $O; exec > >(tee -a $O/job.log) 2>&1
nvidia-smi -i ${CUDA_VISIBLE_DEVICES:-0} --query-gpu=name,memory.total --format=csv,noheader; date -u +%FT%TZ
cd heldout; build_tables
SD=$DATA/eng/sd_$TAG.pt
( flock 9; [ -e $SD ] || PYTHONPATH=$C/heldout:$C/f14 python ../hostchk/mk_sd.py $SD $( [ $TAG = inst ] && echo Qwen/Qwen3-0.6B || echo Qwen/Qwen3-0.6B-Base ) ) 9> $DATA/eng.lock; [ -e $SD ] || exit 1
cd $C/eng
python check.py --model $MODEL --sd $SD --data $C/gen/dev.jsonl --n $N --max_new $MN --out $O --fmt $FMT
rc=$?; date -u +%FT%TZ
mkdir -p $C/../out/eng/$NAME && python - $O/check.jsonl > $C/../out/eng/$NAME/check_summary.txt <<'PY'
import json,sys
for l in open(sys.argv[1]):
    d=json.loads(l); d.pop('divs',None) if 'divs' in d and len(json.dumps(d['divs']))>4000 else None; print(json.dumps(d))
PY
exit $rc
