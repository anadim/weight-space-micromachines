#!/bin/bash
# engine faithfulness along the anomalous seed-0 trajectory (runs/smoke25 = "B", deterministic):
# replay B for 6 updates saving the full trainable state after u3/u4/u5, then at each state compare OLD vs engine2 greedy
# (all fields bitwise) on the 32 reference prompts.
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python run2.py --run_dir runs/replayB --updates 6 --B_max 256 --dev_at 0 --save_host_at 3,4,5 > runs/replayB.out 2>&1
for u in 3 4 5; do python test_at_state.py --state runs/replayB/state_u$u.pt > runs/at_state_u$u.out 2>&1; done
echo CHAIN_DONE
