#!/usr/bin/env bash
# A100: stop a2_u500_s0 after its u300 dev eval (its checkpoints are u250/u500), then 300-update a1 s0, a2 s1, a1 s1
# (checkpoint every 50), then the one-shot finals. $1 = PID of the running a2_u500_s0.
cd "$(dirname "$0")"; export HF_HUB_OFFLINE=1; source env_local.sh
bash stop_at.sh $1 runs/long/a2_u500_s0 300 >> runs/long_a100.log
while kill -0 "$1" 2>/dev/null; do sleep 10; done
for spec in "a1_u300_s0 1 0" "a2_u300_s1 2 1" "a1_u300_s1 1 1"; do
  set -- $spec; d=runs/long/$1; rm -rf $d; mkdir -p $d; echo "$(date) start $1" >> runs/long_a100.log
  python train2.py --arm $2 --seed $3 --run_dir $d --updates 300 --prompts 64 --G 8 --T 1.0 --lr 1e-6 --dev_n 800 \
     --dev_every 25 --ckpt_every 50 --fused 1 > $d/log 2>&1
  echo "$(date) done $1" >> runs/long_a100.log
done
printf 'L_a2_s0_u250 runs/long/a2_u500_s0/step_000250 2\nL_a1_s0_u300 runs/long/a1_u300_s0/step_000300 1\nL_a2_s1_u300 runs/long/a2_u300_s1/step_000300 2\nL_a1_s1_u300 runs/long/a1_u300_s1/step_000300 1\n' > qf_a100_long.txt
bash final_pod.sh qf_a100_long.txt > runs/final_a100_long.log 2>&1
