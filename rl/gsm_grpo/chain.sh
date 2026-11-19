#!/usr/bin/env bash
# chain.sh PID QUEUE LOG [RUNNER]: wait for process PID to exit, then run RUNNER (default abl.sh) on QUEUE
while kill -0 "$1" 2>/dev/null; do sleep 30; done
cd "$(dirname "$0")"; bash "${4:-abl.sh}" "$2" > "$3" 2>&1
