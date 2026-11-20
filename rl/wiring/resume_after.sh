#!/bin/bash
# resume a SIGSTOPped run when the library process sharing its GPU exits
LIBPID=$1; MYPID=$2; TAG=$3
while kill -0 $LIBPID 2>/dev/null; do sleep 60; done
kill -CONT $MYPID && echo "$(date -u +%H:%M:%S) resumed $TAG (pid $MYPID) after library pid $LIBPID exited" >> "$(dirname "$0")/runs/resume.log"
