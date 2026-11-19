#!/bin/bash
# MEM_GB=42
# Mixed-task RL (m4_mixLIBnoisy): GSM-like / BBH-like / IFEval-like / combo (rl_library/mixed), host = Qwen3-0.6B INSTRUCT (chat,
# thinking off) after the format SFT warm-start (sft_host_noisy). Library: calc_s4 + 8 BBH tools + IFEval circuit, frozen; per-tool
# in-wire (floor 0.05) + out-adapters. scramble=0 plain=0. Dr.GRPO lr 1e-6, G8, 32 prompts/update, no KL, max_new 512,
# 300 updates; dev = 400 stratified test items every 50 (+u25); final: full test (999) tools on / off / host-only / LOO.
cd "$(dirname "$0")/.."
export RLENVS=$PWD/mixed WSM_INTEG_DIR=$PWD/../../bbh_generic/integrate/code TOKENIZERS_PARALLELISM=false
python train_library.py --arm LIB --task mixed --host mixed/sft_host_noisy --scramble 0 --plain 0 \
  --run_dir runs/m4_mixLIBnoisy --updates 300 --prompts 32 --G 8 --T 1.0 --lr 1e-6 --kl 0 --loss drgrpo \
  --B_max 128 --tok_budget 4096 --max_new 512 --S_max 896 --dev_mixed_n 400 --dev_every 50 --dev_extra 25 \
  --iw_eps 0.05 --link_lr 1e-3 --link_eps 1e-8 --ckpt_every 300 > runs/m4_mixLIBnoisy.out 2>&1
