#!/bin/bash
# queue.sh JOBFILE : run each line (a launch.sh argument list) when fewer than MAXJ of our python jobs are running
cd "$(dirname "$0")"; MAXJ=${MAXJ:-4}
while read -r line; do
  [ -z "$line" ] && continue
  while [ $(pgrep -f "python (run_gen|run_variants|test_tools|test_calls).py" | wc -l) -ge $MAXJ ]; do sleep 10; done
  ./launch.sh $line; sleep 20
done < "$1"
