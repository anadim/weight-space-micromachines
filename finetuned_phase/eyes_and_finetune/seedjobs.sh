#!/bin/bash
# One job = one (table, arm, seed): fine-tune, evaluate, append tagged result lines to results/seeds/<table>.txt,
# keep the per-task JSONs, delete the checkpoint. Usage: seedjobs.sh FUNC ARGS...   (CUDA_VISIBLE_DEVICES set by the worker)
cd "$(dirname "$0")"; S=results/seeds; mkdir -p $S runs/seeds; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ALL=calc,swap,bool,truth,logic,dyck,walk,time,sort,date,shape; INIT=calc+swap+truth+bool+logic+dyck+walk+time+sort+date+shape
tag() { sed "s|^|$1 seed=$2 |" >> $S/$3.txt; }
gsm_eval() { # model calcmode table arm seed
  for sp in "dev dev" "test test" "data/gsmhard.jsonl hard" "data/bare_test.jsonl bare"; do set -- $1 $2 $3 $4 $5 $sp
    python gen_eval.py --model $1 --calc $2 --split $6 --out $S/$3_$4_s$5_$7.jsonl 2>&1 | grep RESULT | sed "s|^|split=$7 |" | tag $4 $5 $3; done; }
gsm() { # arm seed      plain | unmasked | calc | reg | reg_notool
  local a=$1 s=$2 M=runs/seeds/gsm_$1_s$2
  case $a in
    plain)      python ft.py --fmt plain --aug data/aug.jsonl --seed $s --out $M > $S/ft_gsm_${a}_s$s.log 2>&1; gsm_eval $M self gsm $a $s ;;
    unmasked)   python ft.py --fmt calc --mask_output 0 --aug data/aug.jsonl --seed $s --out $M > $S/ft_gsm_${a}_s$s.log 2>&1; gsm_eval $M self gsm $a $s ;;
    calc)       python ft.py --fmt calc --aug data/aug.jsonl --seed $s --out $M > $S/ft_gsm_${a}_s$s.log 2>&1; gsm_eval $M net gsm $a $s ;;
    reg)        python ft.py --fmt calc --gsm 0 --data data/gsm_reg_train.jsonl --aug data/aug.jsonl --seed $s --out $M > $S/ft_gsm_${a}_s$s.log 2>&1; gsm_eval $M oracle gsm $a $s ;;
    reg_notool) python ft.py --fmt calc --mask_output 0 --gsm 0 --data data/gsm_reg_train.jsonl --aug data/aug.jsonl --seed $s --out $M > $S/ft_gsm_${a}_s$s.log 2>&1; gsm_eval $M self gsm $a $s ;;
  esac; rm -rf $M; }
bbh() { # arm seed      tool | notool   (iteration-5 recipe; tool also tested on the composition set = arm A0)
  local a=$1 s=$2 M=runs/seeds/bbh_$1_s$2 mask=1; [ $a = notool ] && mask=0
  python ft.py --model Qwen/Qwen2.5-0.5B --fmt calc --mask_output $mask --lr 1e-5 --epochs 2 --gsm 1 --aug data/aug.jsonl --data data/bbh5_train.jsonl,data/replay4.jsonl \
    --init_special $INIT --seed $s --out $M > $S/ft_bbh_${a}_s$s.log 2>&1
  if [ $a = tool ]; then MM="--machines $ALL"; C=net; else MM="--no_mm"; C=self; fi
  python bbh_gen.py $M $S/bbh_${a}_s${s}_TEST.json --test $MM --bs 64 > $S/bbh_${a}_s${s}_TEST.log 2>&1; grep MACRO $S/bbh_${a}_s${s}_TEST.log | tag $a $s bbh
  python bbh_gen.py $M $S/comp0_${a}_s${s}.json --data data/compose_test.jsonl $MM > $S/comp0_${a}_s${s}.log 2>&1; grep MACRO $S/comp0_${a}_s${s}.log | tag $a $s comp0
  for sp in "test test" "data/gsmhard.jsonl hard"; do set -- $sp
    python gen_eval.py --model $M --calc $C --machines $ALL --split $1 --out $S/bbhgsm_${a}_s${s}_$2.jsonl 2>&1 | grep RESULT | sed "s|^|split=$2 |" | tag $a $s bbh_gsm; done
  rm -rf $M; }
scale() { # arm n seed
  local a=$1 n=$2 s=$3 M=runs/seeds/scale_$1_$2_s$3 mask=1; [ $a = notool ] && mask=0
  local ep=$(python -c "print(max(2, round(25600/(16*$n))))")
  python ft.py --model Qwen/Qwen2.5-0.5B --fmt calc --mask_output $mask --lr 1e-5 --epochs $ep --gsm 0 --data data/scale_$n.jsonl --init_special $INIT --seed $s --out $M > $S/ft_scale_${a}_${n}_s$s.log 2>&1
  if [ $a = tool ]; then MM="--machines $ALL"; else MM="--no_mm"; fi
  python bbh_gen.py $M $S/scale_${a}_${n}_s${s}.json --data data/bbh5_dev.jsonl $MM > $S/scale_${a}_${n}_s${s}.log 2>&1; grep MACRO $S/scale_${a}_${n}_s${s}.log | tag ${a}_$n $s scale
  rm -rf $M; }
ood() { # arm seed
  local a=$1 s=$2 M=runs/seeds/ood_$1_s$2 mask=1; [ $a = notool ] && mask=0
  python ft.py --model Qwen/Qwen2.5-0.5B --fmt calc --mask_output $mask --lr 1e-5 --epochs 2 --gsm 0 --data data/ood_train.jsonl --init_special $INIT --seed $s --out $M > $S/ft_ood_${a}_s$s.log 2>&1
  if [ $a = tool ]; then MM="--machines $ALL"; else MM="--no_mm"; fi
  for t in out in; do python bbh_gen.py $M $S/ood_${a}_s${s}_$t.json --data data/ood_$([ $t = out ] && echo test || echo in).jsonl $MM > $S/ood_${a}_s${s}_$t.log 2>&1
    grep MACRO $S/ood_${a}_s${s}_$t.log | sed "s|^|set=$t |" | tag $a $s ood; done
  rm -rf $M; }
comp() { # arm seed     A1 (tool) | N1 (no tool): iteration-5 recipe + 900 mixed questions
  local a=$1 s=$2 M=runs/seeds/comp_$1_s$2 mask=1; [ $a = N1 ] && mask=0
  python ft.py --model Qwen/Qwen2.5-0.5B --fmt calc --mask_output $mask --lr 1e-5 --epochs 2 --gsm 1 --aug data/aug.jsonl --data data/bbh5_train.jsonl,data/replay4.jsonl,data/compose_train.jsonl \
    --init_special $INIT --seed $s --out $M > $S/ft_comp_${a}_s$s.log 2>&1
  if [ $a = A1 ]; then MM="--machines $ALL"; else MM="--no_mm"; fi
  python bbh_gen.py $M $S/comp_${a}_s${s}.json --data data/compose_test.jsonl $MM > $S/comp_${a}_s${s}.log 2>&1; grep MACRO $S/comp_${a}_s${s}.log | tag $a $s comp
  rm -rf $M; }
eyes() { # arm seed     tool_4k | notool_4k | tool_16k | notool_16k
  local a=$1 s=$2 M=runs/seeds/eyes_$1_s$2 mask=1 data=data/eyes_train.jsonl:4000; [[ $a == notool* ]] && mask=0; [[ $a == *16k ]] && data=data/eyes_train.jsonl
  python ft.py --model Qwen/Qwen2.5-0.5B --fmt calc --mask_output $mask --lr 1e-5 --epochs 3 --gsm 0 --data $data --init_special calc+eye --seed $s --out $M > $S/ft_eyes_${a}_s$s.log 2>&1
  if [ $mask = 1 ]; then MM="--machines calc,eye"; else MM="--no_mm"; fi
  for sp in dev test; do python bbh_gen.py $M $S/eyes_${a}_s${s}_$sp.json --data data/eyes_$sp.jsonl $MM --bs 32 > $S/eyes_${a}_s${s}_$sp.log 2>&1
    grep MACRO $S/eyes_${a}_s${s}_$sp.log | sed "s|^|split=$sp |" | tag $a $s eyes; done
  rm -rf $M; }
fscale() { # arm n seed   focused scaling study (5 tasks where the machine should matter)
  local a=$1 n=$2 s=$3 M=runs/seeds/fs_$1_$2_s$3 mask=1; [ $a = notool ] && mask=0
  local ep=$(python -c "print(min(100, max(2, round(40960/(5*$n)))))")
  python ft.py --model Qwen/Qwen2.5-0.5B --fmt calc --mask_output $mask --lr 1e-5 --epochs $ep --gsm 0 --data data/fs_$n.jsonl --init_special calc+logic+dyck+sort+eye --seed $s --out $M > $S/ft_fs_${a}_${n}_s$s.log 2>&1
  if [ $a = tool ]; then MM="--machines calc,logic,dyck,sort,eye"; else MM="--no_mm"; fi
  python bbh_gen.py $M $S/fs_${a}_${n}_s${s}.json --data data/fs_dev.jsonl $MM --bs 32 > $S/fs_${a}_${n}_s${s}.log 2>&1; grep MACRO $S/fs_${a}_${n}_s${s}.log | tag ${a}_$n $s fscale
  rm -rf $M; }
comp0() { # model arm seed   an existing checkpoint on the composition set (seed 0 of the A0 table)
  local MM="--machines $ALL"; [ $2 = notool ] && MM="--no_mm"
  python bbh_gen.py $1 $S/comp0_${2}_s${3}.json --data data/compose_test.jsonl $MM > $S/comp0_${2}_s${3}.log 2>&1; grep MACRO $S/comp0_${2}_s${3}.log | tag $2 $3 comp0; }
"$@"
