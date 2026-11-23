#!/bin/bash
# C35 job list: prints one line per job (run from this directory, one GPU per job):
#   bash commands.sh                      # all lines
#   bash commands.sh | grep ' aX_s3 '     # one job
#   CUDA_VISIBLE_DEVICES=0 bash -c "$(bash commands.sh | grep ' aX_s3 ')"
# The original runs went one job per GPU through a cluster queue (eval submitted when its train finished), all on B200.
# Train ~24 / 33 / 30 min (arm 1 / 2 / X), eval ~2-3 min.
# The [37](a) re-evals e37a_Xs0-4 (cap on model tokens only) use the C36 code: see ../calcimp/tt/commands.sh.
j() { echo "$*"; }

j bash job.sh check                                       # -> out/check_tool.json (res/check_tool.json)
j bash job.sh eval 1 base ev_base1                        # row 0
j bash job.sh eval 2 base ev_base2                        # row 02
j bash job.sh eval X base ev_baseX --sync_every 4         # row 0X
for k in $(seq 0 11); do
  j bash job.sh train 1 $k a1_s$k "&&" bash job.sh eval 1 a1_s$k ev_a1s$k                                   # arm 1: plain RL
  j bash job.sh train 2 $k a2_s$k "&&" bash job.sh eval 2 a2_s$k ev_a2s$k                                   # arm 2: RL + calculator implant
  j bash job.sh train X $k aX_s$k --sync_every 4 "&&" bash job.sh eval X aX_s$k ev_Xs$k --sync_every 4      # arm X: RL + text tool
done
