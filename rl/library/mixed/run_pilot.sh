#!/bin/bash
# mixed-task pilot: LIB on Qwen3-0.6B (instruct, chat) from the SFT warm-start, 50 updates, 16 prompts x G8 (A40)
cd "$(dirname "$0")"/..
export WSM_INTEG_DIR=$PWD/../../bbh_generic/integrate/code RLW_DIR=$PWD/../wiring RLENVS=$PWD/mixed FLEX_SMALL=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
ARM=${1:-LIB}; EXTRA=${2:-}
python train_library.py --arm LIB --task mixed --host mixed/sft_host --fused 0 \
  --run_dir runs/mixed_pilot_$ARM --updates 50 --prompts 16 --G 8 --T 1.0 --lr 1e-6 --kl 0 --loss drgrpo \
  --B_max 128 --tok_budget 2048 --max_new 512 --S_max 896 --dev_mixed_n 200 --dev_every 25 --dev_extra "" \
  --iw_eps 0.05 --link_lr 1e-3 --link_eps 1e-8 --ckpt_every 1000 --no_test 1 $EXTRA > runs/mixed_pilot_$ARM.out 2>&1
