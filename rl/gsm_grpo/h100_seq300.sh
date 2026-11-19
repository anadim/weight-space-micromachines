#!/usr/bin/env bash
# H100 GPU0: stop arm 4 s1 at u300 (checkpoint saved), detector audit, long arm 1 (300, ckpt 100), finals, arm-3 reruns.
# $1 = PID of the running arm-4 s1 trainer
cd "$(dirname "$0")"; export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
bash stop_at.sh $1 runs/long/a4_u500_s1 300 ckpt >> runs/h100_seq300.log
while kill -0 "$1" 2>/dev/null; do sleep 10; done
mkdir -p runs/det
for c in "base base" "a4s1_u100 runs/long/a4_u500_s1/step_000100" "a4s1_u200 runs/long/a4_u500_s1/step_000200" \
         "a4s1_u300 runs/long/a4_u500_s1/step_000300" "a2_u100 runs/long/a2_lr1e-6_s0/step_000100" \
         "a2_u200 runs/long/a2_lr1e-6_s0/step_000200" "a2_u300 runs/long/a2_lr1e-6_s0/step_000300"; do
  set -- $c; python det_audit.py --ckpt $2 --out runs/det/$1.json --B_max 512 > runs/det/$1.log 2>&1
done
echo "$(date) det audit done" >> runs/h100_seq300.log
d=runs/long/a1_lr1e-6_s0; rm -rf $d; mkdir -p $d
python train2.py --arm 1 --run_dir $d --updates 300 --prompts 64 --G 8 --T 1.0 --lr 1e-6 --dev_n 800 --dev_every 25 \
   --ckpt_every 100 --fused 1 --seed 0 > $d/log 2>&1
printf 'H_a2_s0_u300 runs/long/a2_lr1e-6_s0/step_000300 2\nH_a1_s0_u300 runs/long/a1_lr1e-6_s0/step_000300 1\nH_a4_s1_u300 runs/long/a4_u500_s1/step_000300 4\nH_base_calc base 2\nH_base_a4 base 4\n' > qf_h100_long.txt
bash final_pod.sh qf_h100_long.txt > runs/final_h100_long.log 2>&1
bash abl.sh q_abl_h100b.txt > runs/abl_h100c.log 2>&1
