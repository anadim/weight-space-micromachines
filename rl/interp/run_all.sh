#!/usr/bin/env bash
# A100 machine: all configs, sequential, A100 (same GPU type for every arm)
cd "$(dirname "$0")"; [ -f ../gsm_grpo/env_local.sh ] && source ../gsm_grpo/env_local.sh; export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
A=../gsm_grpo/runs/abl
python dump_traces.py --ckpt base --implant calc --modes on,off --tag base
python dump_traces.py --ckpt base --implant R --modes on --tag baseR
python dump_traces.py --ckpt $A/a2_lr1e-6_s0/step_000064 --implant calc --modes on,off --tag a2
python dump_traces.py --ckpt $A/a1_lr1e-6_s1/step_000064 --implant calc --modes off,on --tag a1
echo ALLDONE
