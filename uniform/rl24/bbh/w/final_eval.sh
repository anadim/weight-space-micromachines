#!/bin/bash
# Run the u300 BBH-27 evals for every finished run (out/RUN/DONE present, eval not yet in out/), one after the other on one GPU.
# Row 2 = ${PR}_sK u300 on Base, tools off (secondary: the same weights with the 8 tools attached = ${PR}_sK_t1).
# Row 4 = ${IR}_sK u300 on the call-FT host with the 8 tools + refit gate.
# Original runs: one job per eval via a cluster queue (B200, ~5 min each).
cd "$(dirname "$0")"; source ../../../env.sh
D=$RL24_DATA/bbh; PROBE=$D/cg/probe_ft2.pt
PR=${PR:-P0}; IR=${IR:-I0}; FR=${FR:-F0}   # run prefixes (e.g. PR=P2 IR=I2 FR=F2 for an lr 2e-6 rerun)
sub() { echo "== $1"; bash -c "$2"; }
for s in 0 1 2; do
  if [ -f out/${PR}_s$s/DONE ]; then
    [ -f out/eval_${PR}_s${s}_u300.json ] || sub ${PR}_s${s} "bash job.sh eval ${PR}_s${s}_u300 base 0 - $D/runs/${PR}_s$s/u300.pt --B_max 128"
    [ -f out/eval_${PR}_s${s}_u300_t1.json ] || [ -n "$NOSEC" ] || sub ${PR}_s${s}_t1 "bash job.sh eval ${PR}_s${s}_u300_t1 base 1 - $D/runs/${PR}_s$s/u300.pt --B_max 128"
  fi
  if [ -f out/${FR}_s$s/DONE ]; then
    [ -f out/eval_${FR}_s${s}_u300.json ] || sub ${FR}_s${s} "bash job.sh eval ${FR}_s${s}_u300 ft2 0 - $D/runs/${FR}_s$s/u300.pt --B_max 128"
  fi
  if [ -f out/${IR}_s$s/DONE ]; then
    [ -f out/eval_${IR}_s${s}_u300.json ] || sub ${IR}_s${s} "bash job.sh eval ${IR}_s${s}_u300 ft2 1 $PROBE $D/runs/${IR}_s$s/u300.pt --B_max 128"
  fi
done
# u0 control: Base + the 8 tools (only passive tools can fire on Base)
[ -f out/eval_base_t1.json ] || [ -n "$NOSEC" ] || sub base_t1 "bash job.sh eval base_t1 base 1 - - --B_max 128"
