#!/bin/bash
# job_full.sh [SETS] : extract hidden states for SETS (default dev,dev2), then loader/probe sanity demos -> out/
SETS=${1:-dev,dev2}
set -e
python3 extract.py --sets $SETS
for S in ${SETS//,/ }; do
  python3 fit.py --set $S --kind hash --layers 14,18,22,27 --out out/fit_${S}_hash.json
  python3 fit.py --set $S --kind qend --layers 14,18,22 --out out/fit_${S}_qend.json
done
du -sh $WSM_DATA/implants/hs/*
