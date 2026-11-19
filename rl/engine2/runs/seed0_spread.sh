#!/bin/bash
# seed-0 spread on the A40: engine2 run #3, then the OLD engine (train_library.py, m1 flags, A40 settings), 25 updates each
cd <workdir>/rl_library/engine2
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run2.py --run_dir runs/smoke25_c --updates 25 --B_max 256 --dev_at 0,25 > runs/smoke25_c.out 2>&1
cd <workdir>/rl_library
export RLENVS=<workdir>/rl_envs WSM_INTEG_DIR=<workdir>/rl_library/bbh_integrate/code RLW_DIR=<workdir>/rl_wiring FLEX_SMALL=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=0 python train_library.py --fused 0 --arm LIB --task mixed --host <workdir>/rl_library/mixed/sft_host --scramble 0 --plain 0   --run_dir engine2/runs/old_a40_s0 --updates 25 --prompts 64 --G 8 --T 1.0 --lr 1e-6 --kl 0 --loss drgrpo   --B_max 128 --tok_budget 4096 --max_new 512 --S_max 896 --dev_mixed_n 400 --dev_every 50 --dev_extra 25   --iw_eps 0.05 --link_lr 1e-3 --link_eps 1e-8 --ckpt_every 1000 --hf_check 0 --no_test 1 > engine2/runs/old_a40_s0.out 2>&1
echo SPREAD_DONE
