#!/bin/bash
# Run a job file written by curve2.py (or the ones below) sequentially. Line format: MEM_GB|OUT|ENV VARS|COMMAND|DEPS
# (the original runs used a small GPU-memory scheduler to run several at once on one A100; outputs do not depend on it).
# Run from ifeval/heldout with PYTHONPATH=. (circuit.py, tables.py); run_t.py is taken from ../transfer.
JF=$1
while IFS='|' read -r MEM OUT ENVV CMD DEPS; do
  [ -z "$OUT" ] && continue; [[ "$MEM" == \#* ]] && continue
  [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
  echo "$(date +%T) $OUT"; env $ENVV $CMD > "$OUT.log" 2>&1 || echo "FAILED $OUT"
done < "$JF"
