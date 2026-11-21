#!/bin/bash
# task [6] test runs. usage (gq job, cwd = snapshot of gsm_stack/eval): bash fe.sh LISTFILE
# LISTFILE lines: NAME HOST SET KIND N PR VM SAN RM [EXTRA_ENV...]   (HOST: u0 | base | ARM -> $ROOT/gsm_final/rl/ARM/step_000300)
# Every run: T = 1, seed 7, fastroll as the arms' dev evals, rel/v4.2 engine code. Rows -> out/fe/<SET>_<NAME>.jsonl.
ROOT=$WSM_DATA
F="RL_FASTROLL=1 FAST_TK=1 FAST_COMPACT=1 FAST_OVL=1 FAST_ATT3=1 FLEX_SMALL=1"
grep -v '^#' $1 | while read NAME HOST SET KIND N PR VM SAN RM EXTRA; do
  [ -z "$NAME" ] && continue
  case $HOST in u0) XA="";; base) XA="--host_ckpt base";; *) XA="--host_ckpt $ROOT/gsm_final/rl/$HOST/step_000300";; esac
  if [ $KIND = B ] || [ $KIND = SC ]; then SM=1280; BM=384; else if [ $N -gt 4 ]; then SM=5696; BM=192; else SM=2944; BM=384; fi; fi
  if [ -f out/fe/${SET}_${NAME}.jsonl ]; then echo "skip $NAME (exists)"; continue; fi
  XARGS="$XA" BMAX=$BM SAN_SMAX=$SM bash run_hook.sh fe_${SET}_$NAME stk fe_hook.py $F SW_NMAX=$N FE_OUT=$PWD/out/fe FE_SET=$SET \
    FE_KIND=$KIND FE_N=$N FE_PR=$PR FE_VM=$VM FE_SAN=$SAN FE_RM=$RM FE_TAG=$NAME FE_SEED=7 $EXTRA < /dev/null \
    | grep -E "^FE |rror|Traceback|host" | cut -c1-400
done
