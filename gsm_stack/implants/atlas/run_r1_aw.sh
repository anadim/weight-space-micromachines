#!/bin/bash
# Rung-1 (fixed WPR) atlas pipeline: rows aw/out/r1atl_{set}_s{seed}/ -> atlas/data_aw/r1 -> analysis/diverge/lists.
set -e; cd gsm_stack/implants
D=atlas/data_aw/r1; rm -rf $D; mkdir -p $D
cp aw/out/r1atl_*_s*/*.gz $D/
[ -e $D/dev_N8_R1_v1_s1_m0_pr0_wpr_T1.0_seed7.jsonl.gz ] || cp aw/out/r1f_dev/dev_*seed7.jsonl.gz $D/   # dev s7: fixed-code r1f_dev (j23319)
python3 atlas/parse.py --gen $D --out $D --notes $D/counts.md > /dev/null
python3 atlas/analysis_aw.py $D --label r1 > $D/analysis.md
python3 atlas/diverge_aw.py $D > $D/diverge.md
python3 atlas/lists_aw.py $D atlas/data_aw --hand atlas/data_aw/hand_labels_aw.tsv > $D/lists.md
