#!/usr/bin/env bash
# probe: RL'd hosts with the DERANGED calculator attached (how much does each host trust the written digits?)
cd "$(dirname "$0")"; [ -f ../gsm_grpo/env_local.sh ] && source ../gsm_grpo/env_local.sh; export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
A=../gsm_grpo/runs/abl
python dump_traces.py --ckpt $A/a2_lr1e-6_s0/step_000064 --implant R --modes on --tag a2R
python dump_traces.py --ckpt $A/a1_lr1e-6_s1/step_000064 --implant R --modes on --tag a1R
echo ALLDONE
