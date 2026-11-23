#!/bin/bash
# fast-engine dev host check: bash eng/job_hostdev.sh NAME MODEL FMTS
cd $(dirname $0)/..; C=$PWD; source env.sh
NAME=$1 MODEL=${2:-Qwen/Qwen3-0.6B-Base} FMTS=${3:-chat,chatnt,plain}
R=$DATA/hostdev/$NAME; O=$C/../out/hostdev/$NAME; mkdir -p $R $O; exec > >(tee -a $R/job.log) 2>&1
prep_pylib > $R/pylib.log 2>&1; (cd heldout; build_tables)
SDB=$DATA/eng/sd_base.pt; SDI=$DATA/eng/sd_inst.pt
for SD in $SDB $SDI; do T=$( [ $SD = $SDI ] && echo Qwen/Qwen3-0.6B || echo Qwen/Qwen3-0.6B-Base )
  ( flock 9; [ -e $SD ] || (cd heldout && PYTHONPATH=$C/heldout:$C/f14 python ../hostchk/mk_sd.py $SD $T) ) 9> $DATA/eng.lock; done
rm -f $R/res.jsonl
(cd eng && python hostdev.py --model $MODEL --fmts $FMTS --sds refit=$SDB,snap=$SDI --out $R/res.jsonl)
cp $R/res.jsonl* $O/
