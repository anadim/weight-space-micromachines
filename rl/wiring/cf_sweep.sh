#!/bin/bash
# 50-update sweep: A reference + B (in-wire lr x adapter lr) + C (module lr), 2 concurrent runs per GPU
G=$1; shift
cd "$(dirname "$0")"
run() { ./cf_run.sh $G "$@"; }
IW="--iw_eps 0.05 --iw_adam_eps 1e-8"
case "$1" in
  h100)        # one at a time: GPU0 is shared with a library run (35 GB)
    run swA --arm 2 --updates 50
    run swB_iw1e3_cw1e3 --arm CW --updates 50 $IW --iw_lr 1e-3 --cw_lr 1e-3
    run swB_iw1e3_cw1e2 --arm CW --updates 50 $IW --iw_lr 1e-3 --cw_lr 1e-2
    run swB_iw3e4_cw1e3 --arm CW --updates 50 $IW --iw_lr 3e-4 --cw_lr 1e-3 ;;
  a100)
    run swC_core1e3 --arm CWcore --updates 50 $IW --iw_lr 1e-3 --cw_lr 1e-3 --core_lr 1e-3 --reader_lr 1e-3 --soft_p 0.99 &
    run swC_core1e4 --arm CWcore --updates 50 $IW --iw_lr 1e-3 --cw_lr 1e-3 --core_lr 1e-4 --reader_lr 1e-4 --soft_p 0.99 &
    wait
    run swC_core1e2 --arm CWcore --updates 50 $IW --iw_lr 1e-3 --cw_lr 1e-3 --core_lr 1e-2 --reader_lr 1e-2 --soft_p 0.99 &
    wait ;;
esac
echo SWEEP_DONE > runs/cf/SWEEP_DONE_$1
