#!/usr/bin/env bash
# H100 GPU0 (coordinator plan): after arm 4 s1 stops at u300 (PID $1): detector audit -> arm 4 s0 -> arm 1 s0 -> arm 1 s1
# -> arm 2 s1 (all 300 updates, ckpt every 100) -> one-shot finals at u300 -> arm-3 reruns.
cd "$(dirname "$0")"; export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
while kill -0 "$1" 2>/dev/null; do sleep 10; done
mkdir -p runs/det
for c in "base base" "a4s1_u100 runs/long/a4_u500_s1/step_000100" "a4s1_u200 runs/long/a4_u500_s1/step_000200" \
         "a4s1_u300 runs/long/a4_u500_s1/step_000300" "a2_u100 runs/long/a2_lr1e-6_s0/step_000100" \
         "a2_u200 runs/long/a2_lr1e-6_s0/step_000200" "a2_u300 runs/long/a2_lr1e-6_s0/step_000300"; do
  set -- $c; python det_audit.py --ckpt $2 --out runs/det/$1.json --B_max 512 > runs/det/$1.log 2>&1
done
echo "$(date) det audit done" >> runs/h100_seq300.log
run() {  # name arm seed [extra]
  d=runs/long/$1; rm -rf $d; mkdir -p $d; echo "$(date) start $1" >> runs/h100_seq300.log
  python train2.py --arm $2 --seed $3 --run_dir $d --updates 300 --prompts 64 --G 8 --T 1.0 --lr 1e-6 --dev_n 800 \
     --dev_every 25 --ckpt_every 100 --fused 1 ${@:4} > $d/log 2>&1
  echo "$(date) done $1" >> runs/h100_seq300.log
}
run a4_u300_s0 4 0 --max_new 700 --S_max 912 --tok_budget 8192
run a1_lr1e-6_s0 1 0
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
bash abl.sh q_abl_h100b.txt > runs/abl_h100c.log 2>&1
