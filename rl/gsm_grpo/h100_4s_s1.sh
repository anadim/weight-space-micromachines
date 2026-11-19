#!/usr/bin/env bash
# after h100_seq7 (PID $1): seed 1 of the better 4s variant (by dev at the last eval), H100 GPU0
cd "$(dirname "$0")"; export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
while kill -0 "$1" 2>/dev/null; do sleep 30; done
best=$(python3 - <<'PY'
import json
def last(run):
    d = None
    for l in open(f'runs/long/{run}/steps.jsonl'):
        r = json.loads(l)
        if 'dev' in r: d = r['dev']['dev_acc']
    return d or 0
a, b = last('a4sA_u300_s0'), last('a4sB_u300_s0')
print('A' if a >= b else 'B')
PY
)
d=runs/long/a4s${best}_u300_s1; rm -rf $d; mkdir -p $d; echo "$(date) start a4s${best}_u300_s1" >> runs/h100_seq300.log
python train2.py --arm 4s --soft_init $best --seed 1 --run_dir $d --updates 300 --prompts 64 --G 8 --T 1.0 --lr 1e-6 \
   --dev_n 800 --dev_every 25 --ckpt_every 100 --fused 1 --max_new 700 --S_max 912 --tok_budget 8192 --det_lr 1e-4 \
   --det_T 1.0 --soft_lr 1e-3 > $d/log 2>&1
echo "$(date) done a4s${best}_u300_s1" >> runs/h100_seq300.log
echo "$(date) H100 GPU0 queue empty" >> runs/h100_seq300.log
