#!/bin/bash
# OLD (pre-fix, _wpr_ar permuted by compaction) code, ladder-r2 leak confirmation:
# (a) rung-2 dev2 seed 11 alone in a fresh process; (b) WPR only dev2 seeds 7,11 in one process.
env SAN_SMAX=8192 BMAX=128 bash aw.sh c_r2s11 AW_SETS=dev2 AW_SEEDS=11 SW_NS=16 SW_MAXN=7936 AW_ADAPT=A=0.7/16/4:S=4/0.5:B=3:C=400 WPR=1
bash aw.sh c_w711 AW_SETS=dev2 AW_SEEDS=7,11 WPR=1
# (c) leak test on the OLD code (expect NOT bitwise): seed 11 after seed 7 vs fresh, all graphs pre-captured
bash aw_leak.sh old 1
