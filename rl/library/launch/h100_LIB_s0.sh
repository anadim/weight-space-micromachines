#!/bin/bash
# MEM_GB=50
# Tool-library GSM8K RL arm LIB_s0 (rl_library/NOTEBOOK.md): calc_s4 (CW-lean wiring) + 8 BBH tools + IFEval circuit, all frozen;
# per tool: in-wire veto (floor 0.05, lr 1e-3, eps 1e-8) + zero-init out-adapters at L14/final; host Dr.GRPO lr 1e-6, G8, 64 prompts,
# T 1.0, no KL, 300 updates, dev split B (800) greedy every 25 (+u64), LOO + test (GSM8K test + GSM-Hard) at u300.
# --deadline: at u5 the trainer cuts prompts/update (never updates) if the projected finish (incl. final evals) is after 13:15 UTC.
# CUDA_VISIBLE_DEVICES is set by the poller (our GPU = device 0). Saves only the u300 checkpoint.
cd "$(dirname "$0")/.."
python train_library.py --arm LIB --seed 0 --scramble 0 --run_dir runs/LIB_s0 --updates 300 --prompts 64 --G 8 --T 1.0 \
  --lr 1e-6 --kl 0 --loss drgrpo --dev_every 25 --dev_extra 64 --dev_n 800 --B_max 128 --tok_budget 4096 --ckpt_every 300 \
  --iw_eps 0.05 --link_lr 1e-3 --link_eps 1e-8 --deadline 13:15 > runs/LIB_s0.out 2>&1
