#!/bin/bash
# [28] resume + eval driver (rl28/bbh/w): run a pack (pack.sh) that resumes the RL run(s) and then evaluates u300 in the
# same worker (job.sh rl is a no-op once runs/RUN/DONE exists). Repeats (up to 4 chains) until every eval json is in ./out.
# Original runs: each pack was one cluster-queue job on one B200 (4 h limit), started after the previous pack job OLDJOB
# ended (job labels I17b / I17c). Here: one GPU, each chain bounded by `timeout` (TMO, default 4 h).
# usage: bash drive.sh TAG RUN... ; log: out/drive_TAG.log
TAG=$1; shift; RUNS="$@"
cd "$(dirname "$0")"; source ../../../env.sh
D=$RL28_DATA/bbh; PR=$D/cg/probe_ft17.pt; mkdir -p out; LG=out/drive_$TAG.log
say() { echo "$(date -u +%FT%TZ) $*" >> $LG; }
for chain in 1 2 3 4; do
  todo=""; for r in $RUNS; do [ -f out/eval_${r}_u300.json ] || todo="$todo $r"; done
  [ -z "$todo" ] && { say "all evals present: $RUNS"; exit 0; }
  W=(); for r in $todo; do
    s=${r##*_s}; BM=64; [ $r = I17_s2 ] && BM=128
    W+=("bash job.sh rl $r ft17 1 $PR $s --B_max $BM --budget 8192 --mem_frac 0.46; bash job.sh eval ${r}_u300 ft17 1 $PR $D/runs/$r/u300.pt --B_max 64 --mem_frac 0.46")
  done
  say "chain $chain start for:$todo"
  timeout ${TMO:-14400} bash pack.sh ${TAG}c$chain 0.46 "${W[@]}"; rc=$?
  say "chain $chain ended rc=$rc"
done
