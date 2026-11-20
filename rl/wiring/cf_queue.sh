#!/bin/bash
# usage: cf_queue.sh GPU WAITFILE QUEUEFILE   (overnight runner for one GPU slot; several runners may share a queue)
# queue line: NAME|TRAIN ARGS|TEST ARGS   (TEST ARGS empty -> no test). Lines starting with '#' are skipped.
G=$1; WAIT=$2; QF=$3
cd "$(dirname "$0")"
until [ -f "$WAIT" ]; do sleep 60; done
while :; do
  until mkdir runs/cf/.qlock 2>/dev/null; do sleep 2; done
  LINE=$(grep -v '^#' "$QF" | grep . | head -1)
  if [ -n "$LINE" ]; then grep -vxF "$LINE" "$QF" > "$QF.tmp"; mv "$QF.tmp" "$QF"; fi
  rmdir runs/cf/.qlock
  [ -z "$LINE" ] && { echo "$(date -u +%H:%M) slot GPU$G $(basename $WAIT): queue empty" >> runs/cf/LAUNCH_CHECK; exit 0; }
  N=$(echo "$LINE" | cut -d'|' -f1); TR=$(echo "$LINE" | cut -d'|' -f2); TE=$(echo "$LINE" | cut -d'|' -f3)
  while [ "$(df -BG --output=avail . | tail -1 | tr -dc 0-9)" -lt 6 ]; do echo "$(date -u +%H:%M) $N waits for disk (<6 GB)" >> runs/cf/LAUNCH_CHECK; sleep 300; done
  echo "$(date -u +%H:%M) START $N on GPU$G" >> runs/cf/LAUNCH_CHECK
  ./cf_run.sh $G $N $TR
  ( python3 - "$N" >> runs/cf/LAUNCH_CHECK <<'PY'
import json, sys, math
n = sys.argv[1]
try:
    L = [json.loads(l) for l in open(f'runs/cf/{n}/steps.jsonl')]
except Exception: print(f'{n}: no steps'); sys.exit()
u5 = [r for r in L if r.get('u') == 5 and 'reward' in r]
bad = [r for r in L if 'reward' in r and (math.isnan(r.get('loss', 0) or 0))]
print(f"{n}: u5 reward {u5[0]['reward'] if u5 else 'MISSING'}, nan steps {len(bad)}, done {any('DONE_RUN' in r for r in L)}")
PY
  )
  if grep -q DONE_RUN runs/cf/$N/steps.jsonl 2>/dev/null && [ -n "$TE" ]; then
    ./cf_test.sh $G $N $TE
    rm -f runs/cf/$N/step_000300/model.safetensors          # weights not needed after the tests (disk)
  fi
done
