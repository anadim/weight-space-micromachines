#!/bin/bash
# run.sh SETS SEEDS VARIANTS [NLIM] : implants/reread job. CPU test, then S3-serial rollouts (N=8, R=1, T=1) per variant.
# VARIANTS: comma list of full | ask | phrase | base   (full/ask: SER_RR, re-read queue with question copy; phrase: SER_WAIT
# control phrase; base: original writer at the raised token budget). Every (variant, set) runs in a FRESH process with seeds
# in the given order (sampler seed counter alignment with the gen/out base files, see implants/adapt/job/run_adapt.sh).
H=$PWD
export STK_MAXN=7936 SAN_SMAX=8192
( source $WSM_DATA/venv/bin/activate; HF_HOME=$WSM_DATA/hf HF_HUB_CACHE=$WSM_DATA/hf/hub HF_HUB_OFFLINE=1 \
  python3 test_reread.py qs.jsonl.gz ) 2>&1 | tail -3
NL=${4:-0}; TG=""; [ "$NL" != 0 ] && TG=_n$NL
for V in ${3//,/ }; do
  for S in ${1//,/ }; do
    case $V in
      full|ask) env SER_RR=$V STK_TAG=_rr$V$TG STK_N=$NL bash gen.sh $S $2 "8:1:1:1:0:T1.0" 128;;
      phrase) env SER_WAIT=$'Wait, let me read the question carefully and solve it again.\n' STK_TAG=_rrphrase$TG STK_N=$NL bash gen.sh $S $2 "8:1:1:1:0:T1.0" 128;;
      base) env STK_TAG=_rrbase$TG STK_N=$NL bash gen.sh $S $2 "8:1:1:1:0:T1.0" 128;;
    esac
  done
done
echo RUNDONE
