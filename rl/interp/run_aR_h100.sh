#!/usr/bin/env bash
# H100 machine GPU 0: arm R after RL (deranged calc on / off) + base references on the same GPU
cd "$(dirname "$0")"; export RLDIR=../gsm_grpo HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
A=$RLDIR/runs/abl
python dump_traces.py --ckpt base --implant R --modes on,off --tag h_base --B_max 256
python dump_traces.py --ckpt $A/aR_lr1e-6_s0/step_000064 --implant R --modes on,off --tag aR0 --B_max 256
python dump_traces.py --ckpt $A/aR_lr1e-6_s0/step_000064 --implant calc --modes on --tag aR0calc --B_max 256
[ -f $A/aR_lr1e-6_s1/step_000064/DONE ] && python dump_traces.py --ckpt $A/aR_lr1e-6_s1/step_000064 --implant R --modes on,off --tag aR1 --B_max 256
echo ALLDONE
