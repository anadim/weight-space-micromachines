#!/usr/bin/env bash
# H100 GPU0, after the running arm 1 s0 (PID $1): arm 1 s1 -> arm 2 s1 -> finals -> 4cRL s1 -> end-format check
cd "$(dirname "$0")"; export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
while kill -0 "$1" 2>/dev/null; do sleep 10; done
echo "$(date) done a1_lr1e-6_s0" >> runs/h100_seq300.log
run() {
  d=runs/long/$1; rm -rf $d; mkdir -p $d; echo "$(date) start $1" >> runs/h100_seq300.log
  python train2.py --arm $2 --seed $3 --run_dir $d --updates 300 --prompts 64 --G 8 --T 1.0 --lr 1e-6 --dev_n 800 \
     --dev_every 25 --ckpt_every 100 --fused 1 ${@:4} > $d/log 2>&1
  echo "$(date) done $1" >> runs/h100_seq300.log
}
run a1_u300_s1 1 1
run a2_u300_s1 2 1
cat > qf_h100_long.txt <<'Q'
H_a2_s0_u300 runs/long/a2_lr1e-6_s0/step_000300 2
H_a1_s0_u300 runs/long/a1_lr1e-6_s0/step_000300 1
H_a2_s1_u300 runs/long/a2_u300_s1/step_000300 2
H_a1_s1_u300 runs/long/a1_u300_s1/step_000300 1
H_a4_s0_u300 runs/long/a4_u300_s0/step_000300 4
H_a4_s1_u300 runs/long/a4_u500_s1/step_000300 4
H_base_calc base 2
H_base_a4 base 4
Q
bash final_pod.sh qf_h100_long.txt > runs/final_h100_long.log 2>&1
echo "$(date) finals done" >> runs/h100_seq300.log
bash run4c.sh a4cRL_u300_s1 4cRL 1 512
for c in base runs/long/a4_u500_s1/step_000300 runs/long/a2_lr1e-6_s0/step_000300 runs/long/a4_u500_s1/step_000100; do
  python end_check.py --ckpt $c --n 300 2>&1 | grep -E "^(base|runs)|next text|Error" >> runs/end_check.log
done
