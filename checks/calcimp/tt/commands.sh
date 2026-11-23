#!/bin/bash
# C36 job list: prints one line per job (run from this directory, one GPU per job):
#   bash commands.sh                      # all lines
#   bash commands.sh | grep ' fI_s3 '     # one job
#   CUDA_VISIBLE_DEVICES=0 bash -c "$(bash commands.sh | grep ' fI_s3 ')"
# The original runs went one job per GPU through a cluster queue. B200: ~40 min per train+eval line, 62.5 GB peak
# (packing two lines on one GPU gave only x0.92 throughput: not worth it).
# Train seeds 12-27 (T1: 12-19) ran on A100-SXM4 40 GB with smaller batching; the exact lines are the A100 block below
# (~1h20-2h05 per line there). Everything else ran on one B200.
A='--sync_every 4 --cap_extra 128'                       # tool / implant: model-token cap 320 + up to 128 tool tokens
I='--link_lr 1e-2 --iw_eps 0.05 --ci_adapters 1 --ci_std 1e-3'   # arm I wiring (the [41] dev-selected config)
A100T='--tok_budget 4096 --B_max 192 --ckpt_every 1000'; A100E='--B_max 192'
j() { echo "$*"; }

j bash job.sh check_ci                                   # -> out/check_ci.json (res/check_ci.json)
j bash job.sh eval X base eb_baseX $A                     # row 0X (no RL)
j bash job.sh eval I0 base eb_baseI0 $A                   # row 0I0 (no RL)

# ---- [41] dev-only sweep, 100 updates, dev 800 every 25 u (no eval). Seed 0 on B200; seed-1/2 replicates of the I grid
# and of swT were A100 jobs ($A100T added), the swIx / swT seed-1 runs B200.
for s in 0 1; do
  for ll in 1e-3 1e-2 3e-2; do for f in 0.05 0; do
    x=; [ $s = 1 ] && x=" $A100T"
    j bash job.sh train I $s swI_ll${ll}_f${f}_r_s$s --updates 100 --ckpt_every 1000 --link_lr $ll --iw_eps $f --ci_std 1e-3 $A$x
    j bash job.sh train I $s swI_ll${ll}_f${f}_tok_s$s --updates 100 --ckpt_every 1000 --link_lr $ll --iw_eps $f --ci_adapters 0 $A$x
    j bash job.sh train I $s swI_ll${ll}_f${f}_z_s$s --updates 100 --ckpt_every 1000 --link_lr $ll --iw_eps $f $A$x
  done; done
  for lr in 5e-7 1e-6 2e-6 4e-6 8e-6; do                  # swIx_lr1e-6_s0 = swI_ll1e-2_f0.05_r_s0 (not re-run)
    [ $s$lr = 01e-6 ] || j bash job.sh train I $s swIx_lr${lr}_s$s --updates 100 --ckpt_every 1000 --lr $lr $I $A
  done
  for lr in 5e-7 2e-6 4e-6 8e-6; do j bash job.sh train X $s swT_lr${lr}_s$s --updates 100 --ckpt_every 1000 --lr $lr $A; done
done
for lr in 5e-7 2e-6 4e-6; do j bash job.sh train X 2 swT_lr${lr}_s2 --updates 100 --ckpt_every 1000 --lr $lr $A $A100T; done
# (swT at 1e-6 = the first 100 u of the T_s* finals below)

# ---- finals, 300 updates + eval (test / GSM-Hard / GSM-Plus testmini, eval seeds 7/11/13). Row names as in c36_cfg.json.
for k in $(seq 0 11); do
  j bash job.sh train X $k T_s$k $A "&&" bash job.sh eval X T_s$k ev_Ts$k $A                                      # T1
  j bash job.sh train X $k fT2_s$k --lr 2e-6 $A "&&" bash job.sh eval X fT2_s$k ev_fT2_s$k $A                     # T2
  j bash job.sh train I $k fI_s$k --lr 2e-6 $I $A "&&" bash job.sh eval I fI_s$k ev_fI_s$k $A \
    "&&" bash job.sh eval I fI_s$k ev_fIe0_s$k $A --iw_eps_eval 0                                                 # I, Ie0
  j bash job.sh train I0 $k fI0_s$k --lr 2e-6 $A "&&" bash job.sh eval I0 fI0_s$k ev_fI0_s$k $A                  # I0
  j bash job.sh train Itok $k fItok_s$k --lr 2e-6 --link_lr 1e-2 $A "&&" bash job.sh eval Itok fItok_s$k ev_fItok_s$k $A   # Itok
done
for k in $(seq 0 7); do
  j bash job.sh train X $k fT4_s$k --lr 4e-6 $A "&&" bash job.sh eval X fT4_s$k ev_fT4_s$k $A                     # T (u100 pick)
  j bash job.sh train I $k fI1_s$k --lr 1e-6 $I $A "&&" bash job.sh eval I fI1_s$k ev_fI1_s$k $A                  # I1
  j bash job.sh train I0 $k fI01_s$k --lr 1e-6 $A "&&" bash job.sh eval I0 fI01_s$k ev_fI01_s$k $A               # I01
  j bash job.sh train Itok $k fItok1_s$k --lr 1e-6 --link_lr 1e-2 $A "&&" bash job.sh eval Itok fItok1_s$k ev_fItok1_s$k $A  # Itok1
done

# ---- A100-SXM4 40 GB block: train seeds 12-27 (T1: 12-19), same lines + the 40 GB batching flags
for k in $(seq 12 27); do
  [ $k -le 19 ] && j bash job.sh train X $k T_s$k $A $A100T "&&" bash job.sh eval X T_s$k ev_Ts$k $A $A100E
  j bash job.sh train X $k fT2_s$k --lr 2e-6 $A $A100T "&&" bash job.sh eval X fT2_s$k ev_fT2_s$k $A $A100E
  j bash job.sh train I $k fI_s$k --lr 2e-6 $I $A $A100T "&&" bash job.sh eval I fI_s$k ev_fI_s$k $A $A100E \
    "&&" bash job.sh eval I fI_s$k ev_fIe0_s$k $A $A100E --iw_eps_eval 0
  j bash job.sh train I0 $k fI0_s$k --lr 2e-6 $A $A100T "&&" bash job.sh eval I0 fI0_s$k ev_fI0_s$k $A $A100E
  j bash job.sh train Itok $k fItok_s$k --lr 2e-6 --link_lr 1e-2 $A $A100T "&&" bash job.sh eval Itok fItok_s$k ev_fItok_s$k $A $A100E
done

# ---- arm-I collapse diagnostic (B200; not pooled into any row): the two A100 seeds that collapsed, re-run with the
# B200 batching (fIr) and with the A100 batching on a B200 (fIa)
for k in 14 17; do
  j bash job.sh train I $k fIr_s$k --lr 2e-6 $I $A "&&" bash job.sh eval I fIr_s$k ev_fIr_s$k $A
  j bash job.sh train I $k fIa_s$k --lr 2e-6 $I --B_max 192 --tok_budget 4096 $A "&&" bash job.sh eval I fIa_s$k ev_fIa_s$k $A
done

# ---- C35 [37](a): the C35 arm-X u300 checkpoints re-evaluated with the cap on model tokens only (rows in ../../tooltext;
# checkpoints written by ../../tooltext/job.sh under its WSM_DATA, default $WSM_ROOT/runs/tooltext; override TT_RUNS)
TTR=${TT_RUNS:-$(cd "$(dirname "$0")/../../.." && pwd)/runs/tooltext/runs}
for k in 0 1 2 3 4; do j bash job.sh eval X $TTR/aX_s$k/step_000300 e37a_Xs$k $A; done
