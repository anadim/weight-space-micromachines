#!/usr/bin/env bash
# H100 GPU0 after h100_seq6: smoke of arm 4s -> [4c s0 if MOVE=1] -> 4s A s0 -> 4s B s0
cd "$(dirname "$0")"; export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
rm -rf runs/smoke4s
python train2.py --arm 4s --soft_init A --run_dir runs/smoke4s --updates 2 --prompts 8 --G 4 --dev_n 32 --dev_every 2 \
   --ckpt_every 1000 --B_max 64 --S_max 912 --max_new 700 --tok_budget 4096 --fused 1 > runs/smoke4s.log 2>&1
echo "$(date) smoke4s exit $?" >> runs/h100_seq300.log
[ "$1" = "1" ] && bash run4c.sh a4c_u300_s0 4c 0 512
run4s() {  # name init seed
  d=runs/long/$1; rm -rf $d; mkdir -p $d; echo "$(date) start $1" >> runs/h100_seq300.log
  python train2.py --arm 4s --soft_init $2 --seed $3 --run_dir $d --updates 300 --prompts 64 --G 8 --T 1.0 --lr 1e-6 \
     --dev_n 800 --dev_every 25 --ckpt_every 100 --fused 1 --max_new 700 --S_max 912 --tok_budget 8192 --det_lr 1e-4 \
     --det_T 1.0 --soft_lr 1e-3 > $d/log 2>&1
  echo "$(date) done $1" >> runs/h100_seq300.log
}
run4s a4sA_u300_s0 A 0
run4s a4sB_u300_s0 B 0
