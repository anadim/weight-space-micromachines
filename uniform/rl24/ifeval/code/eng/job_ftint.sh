#!/bin/bash
# task [29]: IFEval-541 rows 1 (plain) and 3 (+14-rule implant, snapshot sd_inst) on the BBH call-FT host ft_int_v2
# (Qwen3-0.6B-Base + §6.1 tool-call full FT, REPRODUCE §6; $FT2_HOST, not released). Same format/decoding as the Base u0 rows.
cd $(dirname $0)/..; source env.sh
M=$FT2_HOST
ls -la $M; [ -e $M/model.safetensors ] || { echo "missing $M"; exit 9; }
sha256sum $M/model.safetensors
exec bash eng/job_eval.sh ftint $M inst '' chat
