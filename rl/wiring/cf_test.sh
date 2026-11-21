#!/bin/bash
# usage: cf_test.sh GPU RUN ARGS...   test evals for a finished call-form run (u300 checkpoint):
#   (1) trained host + trained module/wiring | trained host + ORIGINAL module/wiring (cw_hand) | implant off  -- test, hard, dev
#   (2) ORIGINAL (base) host + trained module/wiring (and its hand reset / off)                               -- test, hard, dev
G=$1; N=$2; shift 2
cd "$(dirname "$0")"; [ -f env_local.sh ] && source env_local.sh
export CALC_MACHINE=calc_tag CALLFORM=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
CK=runs/cf/$N/step_000300
COMMON="--B_max 128 --tok_budget 4096 --fused 0 --calib_traces runs/cf/cf_calib.jsonl --test_sets test,hard,dev"
CUDA_VISIBLE_DEVICES=$G python train_wiring.py --run_dir runs/cf/${N}_test --eval_ckpt $CK $COMMON "$@" > runs/cf/${N}_test.out 2>&1
CUDA_VISIBLE_DEVICES=$G python train_wiring.py --run_dir runs/cf/${N}_basehost --eval_ckpt $CK --host_ckpt base $COMMON "$@" > runs/cf/${N}_basehost.out 2>&1
echo done > runs/cf/${N}_TESTED
