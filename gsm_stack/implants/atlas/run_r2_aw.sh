#!/bin/bash
# Rung-2 (fixed WPR + A+B+C+stop4) atlas: rows aw/out/r2f_{dev,dev2}/ (j23319 / j23340) -> atlas/data_aw/r2f.
set -e; cd gsm_stack/implants
D=atlas/data_aw/r2f; rm -rf $D; mkdir -p $D
cp aw/out/r2f_dev/*.gz $D/; [ -d aw/out/r2f_dev2 ] && cp aw/out/r2f_dev2/*.gz $D/
S=dev; [ -d aw/out/r2f_dev2 ] && S=dev,dev2
python3 atlas/parse.py --gen $D --out $D --notes $D/counts.md > /dev/null
python3 atlas/analysis_aw.py $D --label r2f --sets $S > $D/analysis.md
python3 atlas/diverge_aw.py $D --sets $S > $D/diverge.md 2> $D/diverge.err
python3 atlas/ceiling_aw.py $D > $D/ceiling.md
