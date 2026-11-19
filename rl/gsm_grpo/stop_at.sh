#!/usr/bin/env bash
# stop_at.sh PID RUN_DIR U : when RUN_DIR has the dev eval at update U logged (and, if it saves one, step_<U>/DONE),
# stop the training process PID. Cap of 300 updates requested by the user.
U=$3; S=$(printf "step_%06d" $U)
while kill -0 "$1" 2>/dev/null; do
  if grep -q "\"u\": $U, \"t\"" "$2/steps.jsonl" 2>/dev/null && python3 - "$2/steps.jsonl" "$U" <<'PY'
import sys, json
u = int(sys.argv[2]); ok = False
for l in open(sys.argv[1]):
    r = json.loads(l)
    if 'dev' in r and r['dev'].get('u') == u: ok = True
sys.exit(0 if ok else 1)
PY
  then
    if [ -z "$4" ] && sleep 150 || [ -f "$2/$S/DONE" ]; then sleep 5; kill "$1"; echo "$(date) stopped $2 at u$U"; exit 0; fi
  fi
  sleep 20
done
