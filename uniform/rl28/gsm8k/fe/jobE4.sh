#!/bin/bash
# usage: PROCS="cw17_s2_test+p17_s2 cw17_s2_hard" bash uniform/rl28/gsm8k/fe/jobE4.sh        (one GPU, B200 / H100 80 GB)
# Packed 1.7B eval job of REPRODUCE §18: PROCS = space-separated processes sharing the GPU (memory cap 0.45 each), each a
# '+'-chain of list names (fe/lists/) run in sequence. Settings by list family, all on the stk_r2q_17 overlay:
#   row 1 (a): fe.sh defaults;   rows 3/4 and their ablations (r3_*, cw17_*, q32c_r3_*, gp17_r3, gp17_cw_*, v17_*, a33_*, a33h_*, att17_*): B_max 64, S_max 8192
#   rows 1/2 (p17_*, q32c_h*, gp17_h*): B_max 128;   matched-compute SCH (q32b_*, q32b3_*): B_max 256 (S_max 1280)
# Runs the adaptive-tries CPU fuzz (test_aw.py) first. Logs -> fe/out/e4_<E4TAG><k>.txt, SM% -> fe/out/dmon_E4<E4TAG>.txt.
U="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$U/../../../env.sh"
mkdir -p $U/out
GI=${CUDA_VISIBLE_DEVICES%%,*}
nvidia-smi dmon -s u -d 10 -i $GI > $U/out/dmon_E4${E4TAG}.txt 2>&1 &
DM=$!
cd "$WSM_ROOT"; H=$(gs_tree "$U/../trees/stk_r2q_17" fuzz17_$$) || { kill $DM; exit 2; }
(cd $H/stk && STK_DIR=$PWD python3 test_aw.py) > $U/out/test_aw.txt 2>&1 || { echo "test_aw FAILED"; tail $U/out/test_aw.txt; kill $DM; exit 3; }
tail -1 $U/out/test_aw.txt
pids=""; k=0
for p in $PROCS; do
  k=$((k + 1))
  (
    for l in ${p//+/ }; do
      case $l in
        r3_*|cw17_*|q32c_r3_*|gp17_r3|gp17_cw_*|v17_*|a33_*|a33h_*|att17_*) FE_MEMFRAC=0.45 FE_BMAX=64 FE_SMAX=8192 bash $U/fe.sh $l ;;
        p17_*|q32c_h*|gp17_h*) FE_MEMFRAC=0.45 FE_BMAX=128 bash $U/fe.sh $l ;;
        q32b_*|q32b3_*) FE_MEMFRAC=0.45 FE_BMAX=256 bash $U/fe.sh $l ;;
        a) bash $U/fe.sh a ;;   # row 1 + idle check, as the original job A (B_max 384, no memory cap)
        *) echo "unknown list $l"; exit 2 ;;
      esac
    done
  ) > $U/out/e4_${E4TAG}$k.txt 2>&1 &
  pids="$pids $!"; sleep 30
done
rc=0; for q in $pids; do wait $q; r=$?; [ $r -gt $rc ] && rc=$r; done
cat $U/out/e4_${E4TAG}*.txt
grep -h "audit\|writer_steps\|sampled_ne" $WSM_WORK/out/fe17_*/log.txt | cut -c1-300 | head -20
kill $DM
awk '$1 ~ /^[0-9]+$/ {s += $2; n++} END {if (n) printf "DMON mean SM%% %.1f over %d samples\n", s / n, n}' $U/out/dmon_E4${E4TAG}.txt
exit $rc
