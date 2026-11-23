#!/bin/bash
# RL smoke: bash eng/job_smoke.sh NAME MODEL HOSTTAG(inst|base) [UPD] -- arm I then arm P, a few updates each (TF ratio at u1, s/update)
cd $(dirname $0)/..; C=$PWD; source env.sh
NAME=$1 MODEL=$2 TAG=$3 UPD=${4:-4}
O=$DATA/smoke/$NAME; mkdir -p $O; exec > >(tee -a $O/job.log) 2>&1
nvidia-smi -i ${CUDA_VISIBLE_DEVICES:-0} --query-gpu=name,memory.total --format=csv,noheader; date -u +%FT%TZ
prep_pylib > $O/pylib.log 2>&1 || { echo pylib failed; tail $O/pylib.log; }
cd heldout; build_tables
SD=$DATA/eng/sd_$TAG.a.pt
( flock 9; [ -e $SD ] || PYTHONPATH=$C/heldout:$C/f14 python ../hostchk/mk_sd.py $SD $( [ $TAG = inst ] && echo Qwen/Qwen3-0.6B || echo Qwen/Qwen3-0.6B-Base ) ) 9> $DATA/eng.lock
cd $C/eng
for ARM in I P; do
  R=$O/$ARM; rm -rf $R
  X=""; [ $ARM = I ] && X="--sd $SD"
  python rl_if.py --model $MODEL --arm $ARM $X --run $R --seed 0 --stop_after $UPD --dev_n 64 --resume_every 2
  echo "arm $ARM rc $?"; date -u +%FT%TZ
done
mkdir -p $C/../out/smoke/$NAME; for ARM in I P; do cp $O/$ARM/steps.jsonl $C/../out/smoke/$NAME/steps_$ARM.jsonl; done
