#!/bin/bash
# MM12 eyes + the remaining no-tool controls, queued behind the jobs already on each GPU.
cd "$(dirname "$0")"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
INIT=calc+swap+truth+bool+logic+dyck+walk+time+sort+date+shape+eye; ALL=calc,swap,bool,truth,logic,dyck,walk,time,sort,date,shape
waitfor() { until grep -q "$2" $1 2>/dev/null; do sleep 60; done; }
eyes_arm() { # gpu name data mask
  R=results/eyes
  CUDA_VISIBLE_DEVICES=$1 python ft.py --model Qwen/Qwen2.5-0.5B --fmt calc --mask_output $4 --lr 1e-5 --epochs 3 --gsm 0 --data $3 --init_special calc+eye --out runs/eyes_$2 > $R/ft_$2.log 2>&1
  for sp in dev test; do
    if [ $4 = 1 ]; then CUDA_VISIBLE_DEVICES=$1 python bbh_gen.py runs/eyes_$2 $R/$2_$sp.json --data data/eyes_$sp.jsonl --machines calc,eye --bs 32 > $R/$2_$sp.log 2>&1
    else CUDA_VISIBLE_DEVICES=$1 python bbh_gen.py runs/eyes_$2 $R/$2_$sp.json --data data/eyes_$sp.jsonl --no_mm --bs 32 > $R/$2_$sp.log 2>&1; fi
  done; grep MACRO $R/$2_*.log >> $R/summary.txt; }
q0() { waitfor results/reg/summary.txt REG_calc_DONE; eyes_arm 0 tool_4k data/eyes_train.jsonl:4000 1
       # register format, no tool: box in the loss, the model computes every box itself
       CUDA_VISIBLE_DEVICES=0 python ft.py --model Qwen/Qwen2.5-0.5B --fmt calc --mask_output 0 --lr 1e-5 --epochs 3 --gsm 0 --aug data/aug.jsonl --data data/gsm_reg_train.jsonl --init_special calc --out runs/gsm_reg_notool > results/reg/ft_reg_notool.log 2>&1
       for sp in "dev dev" "test test" "data/gsmhard.jsonl hard"; do set -- $sp
         CUDA_VISIBLE_DEVICES=0 python gen_eval.py --model runs/gsm_reg_notool --calc self --split $1 --out results/reg/reg_notool_$2.jsonl 2>&1 | grep RESULT >> results/reg/summary.txt; done
       echo REG_notool_DONE >> results/reg/summary.txt; }
q2() { waitfor results/reg/summary.txt REG_reg_DONE; eyes_arm 2 notool_4k data/eyes_train.jsonl:4000 0
       CUDA_VISIBLE_DEVICES=2 python ft.py --model Qwen/Qwen2.5-0.5B --fmt calc --mask_output 0 --lr 1e-5 --epochs 3 --gsm 0 --aug data/aug.jsonl --data data/gsm_calc_train.jsonl --init_special calc --out runs/gsm_calc_notool > results/reg/ft_calc_notool.log 2>&1
       for sp in "dev dev" "test test" "data/gsmhard.jsonl hard"; do set -- $sp
         CUDA_VISIBLE_DEVICES=2 python gen_eval.py --model runs/gsm_calc_notool --calc self --split $1 --out results/reg/calc_notool_$2.jsonl 2>&1 | grep RESULT >> results/reg/summary.txt; done
       echo REG_calcnotool_DONE >> results/reg/summary.txt; }
q3() { waitfor results/eqv/summary.txt EQV_DONE; waitfor results/ctrl/summary_quad.txt OOD_notool_DONE; eyes_arm 3 notool_16k data/eyes_train.jsonl 0; eyes_arm 3 tool_16k data/eyes_train.jsonl 1; }
q1() { waitfor results/comp/summary.txt COMP_DONE
       # composition, no tool: iteration-5 data + mixed questions, box in the loss, no machines
       CUDA_VISIBLE_DEVICES=1 python ft.py --model Qwen/Qwen2.5-0.5B --fmt calc --mask_output 0 --lr 1e-5 --epochs 2 --gsm 1 --aug data/aug.jsonl \
         --data data/bbh5_train.jsonl,data/replay4.jsonl,data/compose_train.jsonl --init_special calc+swap+truth+bool+logic+dyck+walk+time+sort+date+shape --out runs/comp_N1 > results/comp/ft_N1.log 2>&1
       CUDA_VISIBLE_DEVICES=1 python bbh_gen.py runs/comp_N1 results/comp/N1_compose.json --data data/compose_test.jsonl --no_mm > results/comp/N1_compose.log 2>&1
       grep MACRO results/comp/N1_compose.log >> results/comp/summary.txt; echo COMPN_DONE >> results/comp/summary.txt; }
q0 & q1 & q2 & q3 & wait; echo EYES_ALL_DONE >> results/eyes/summary.txt
