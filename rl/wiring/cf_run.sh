#!/bin/bash
# usage: cf_run.sh GPU NAME ARGS...   (call-form calculator ladder, 2026-11-19)
# - every run checkpoints (resumable) every 50 updates to /dev/shm and resumes from it when relaunched
# - on a non-zero exit without DONE_RUN the run is relaunched (resume) up to 3 times
# - memory cap 0.47 of the GPU by default (two runs per GPU); override with a later --mem_frac
G=$1; N=$2; shift 2
cd "$(dirname "$0")"; [ -f env_gcr.sh ] && source env_gcr.sh
export CALC_MACHINE=calc_tag CALLFORM=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
COMMON="--prompts 64 --G 8 --T 1.0 --lr 1e-6 --kl 0 --loss drgrpo --B_max 128 --tok_budget 4096 --fused 0 --dev_every 25 --dev_extra 0 --ckpt_every 100000 --calib_traces runs/cf/cf_calib.jsonl --mem_frac 0.47 --resume_every 50 --resume 1"
try=0
while :; do
  try=$((try + 1))
  CUDA_VISIBLE_DEVICES=$G python train_wiring.py --run_dir runs/cf/$N $COMMON "$@" >> runs/cf/$N.out 2>&1 &
  PID=$!
  ( for k in $(seq 1 60); do sleep 20; grep -q '^{"u": 5,' runs/cf/$N.out 2>/dev/null && break; kill -0 $PID 2>/dev/null || break; done; R5=$(grep '^{"u": 5,' runs/cf/$N.out | tail -1 | grep -o '"reward": [0-9.na]*'); echo "$(date -u +%H:%M) $N try $try u5 health: ${R5:-no u5}" >> runs/cf/LAUNCH_CHECK ) &
  ( sleep 75; if ! kill -0 $PID 2>/dev/null && ! grep -q DONE_RUN runs/cf/$N/steps.jsonl 2>/dev/null; then echo "$(date -u +%H:%M) $N try $try FAILED at start" >> runs/cf/LAUNCH_CHECK; else echo "$(date -u +%H:%M) $N try $try ok after 75 s" >> runs/cf/LAUNCH_CHECK; fi ) &
  wait $PID; rc=$?
  grep -q DONE_RUN runs/cf/$N/steps.jsonl 2>/dev/null && break
  echo "$(date -u +%H:%M) $N exited rc=$rc without DONE (try $try)" >> runs/cf/LAUNCH_CHECK
  [ $try -ge 4 ] && break
  sleep 30
done
