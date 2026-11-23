#!/bin/bash
# HOST CHECK job: exports (chat snapshot / Base refit), 7 configs concurrently on one GPU, then seeded lm_eval scoring.
cd $(dirname $0)/..; C=$PWD; source env.sh
O=$DATA/hostchk; mkdir -p $O; exec > >(tee -a $O/job.log) 2>&1
nvidia-smi -i ${CUDA_VISIBLE_DEVICES:-0} --query-gpu=name,memory.total --format=csv,noheader; date -u +%FT%TZ
(prep_pylib > $O/pylib.log 2>&1; echo pylib rc $?) &
cd heldout; build_tables; sha256sum tables.pt
[ -e $O/sd_inst.pt ] || PYTHONPATH=$C/heldout:$C/f14 python ../hostchk/mk_sd.py $O/sd_inst.pt Qwen/Qwen3-0.6B || exit 1
[ -e $O/sd_base.pt ] || PYTHONPATH=$C/heldout:$C/f14 python ../hostchk/mk_sd.py $O/sd_base.pt Qwen/Qwen3-0.6B-Base || exit 1
cd $C; D=$C/gen/dev.jsonl; pids=""
run() { n=$1; shift; [ -e $O/$n.jsonl ] || { python hostchk/hc.py $D $O/$n.jsonl "$@" > $O/$n.log 2>&1 & pids="$pids $!"; }; }
run base_chat --model Qwen/Qwen3-0.6B-Base --fmt chat
run base_chat_snap --model Qwen/Qwen3-0.6B-Base --fmt chat --sd $O/sd_inst.pt
run base_chat_refit --model Qwen/Qwen3-0.6B-Base --fmt chat --sd $O/sd_base.pt
run base_plain --model Qwen/Qwen3-0.6B-Base --fmt plain
run base_plain_refit --model Qwen/Qwen3-0.6B-Base --fmt plain --sd $O/sd_base.pt
run inst_chat --model Qwen/Qwen3-0.6B --fmt chat
run inst_chat_imp --model Qwen/Qwen3-0.6B --fmt chat --sd $O/sd_inst.pt
wait
date -u +%FT%TZ
for n in base_chat base_chat_snap base_chat_refit base_plain base_plain_refit inst_chat inst_chat_imp; do
  tail -1 $O/$n.log
  [ -e $O/$n.jsonl ] && PYTHONPATH=$PYL python score_reduce.py $D $O/$n.jsonl $O/$n.score.json --command "hostchk $n" > /dev/null 2>&1
done
python hostchk/summ.py $O | tee $O/summary.txt
mkdir -p $C/../out/hostchk && cp $O/summary.txt $O/*.score.json $C/../out/hostchk/ 2>/dev/null
