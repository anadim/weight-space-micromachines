#!/bin/bash
# chain.sh RUN HOST TOOLS GATE SEED [extra rl_bbh args] — resumable RL driver: runs `bash job.sh rl RUN HOST TOOLS GATE SEED ...`
# and restarts it after a crash / time limit until ./out/RUN/DONE exists. Every start resumes from $RL24_DATA/bbh/runs/RUN/resume.pt
# (written every 2 updates). Chain log: ./out/RUN.chain.log.
# Original runs: one job per GPU via a cluster queue (4 h limit per job, resubmitted). Here: one GPU, CUDA_VISIBLE_DEVICES=0.
RUN=$1; HOST=$2; TOOLS=$3; GATE=$4; SEED=$5; shift 5
[ -n "$SEED" ] || { echo "usage: $0 RUN HOST TOOLS GATE SEED [args]"; exit 64; }
HERE=$(cd "$(dirname "$0")" && pwd); cd $HERE
mkdir -p out; CL=$HERE/out/$RUN.chain.log
for chain in $(seq 1 ${MAXCHAIN:-12}); do
  echo "$(date -u +%FT%TZ) chain $chain start ($RUN)" >> $CL
  timeout ${TMO:-14400} bash job.sh rl $RUN $HOST $TOOLS $GATE $SEED "$@"; rc=$?
  echo "$(date -u +%FT%TZ) chain $chain ended rc=$rc" >> $CL
  if [ -f $HERE/out/$RUN/DONE ]; then echo "$(date -u +%FT%TZ) $RUN DONE" >> $CL; exit 0; fi
  [ -n "$NOCHAIN" ] && exit $rc
  sleep 30
done
exit 1
