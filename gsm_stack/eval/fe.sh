#!/bin/bash
# usage: bash gsm_stack/eval/fe.sh LISTFILE        (one GPU; configs run one after the other)
# LISTFILE lines: NAME HOST SET KIND N PR VM SAN RM [EXTRA_ENV...]
#   HOST: u0 (the RL+calc host a2_u300_s2 = $CKPT_U0) | base (Qwen3-0.6B-Base) | ARM ($ARMDIR/rl/ARM/step_000300)
#   SET: test | hard | dev | dev2;  KIND: S serial (N tries) | B (S2F, one attempt) | SC (N independent B samples +
#   plain majority: programmatic control) | H / C4 / D (ladder: host only / + calc_s4 / + calc_s4 + chain2)
#   PR 1 = programmatic position reset (NON-COMPLIANT reference); VM 1 probe voter / 0 plain majority; SAN sanity filter
#   in the vote; RM 1 = repair only missing answers. Columns are documented in trees/v42/stk/fe_hook.py.
# Env: FE_TREE (code tree, default v42 = the frozen rel/v4.2 copy every test number used; r2 / r3 = rung 2 / rung 3),
#   FE_BMAX / FE_SMAX override B_max / SAN_SMAX, FE_OUT_DIR (default $WSM_WORK/fe).
# Every run: T = 1, seed 7 (FE_SEED overrides), fastroll as the arms' dev evals. Rows -> $FE_OUT_DIR/<SET>_<NAME>.jsonl
# (one record per item: gold, prediction, correctness, slot registers; no text). Summary lines start with "FE ".
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
FE_OUT_DIR=${FE_OUT_DIR:-$WSM_WORK/fe}; mkdir -p $FE_OUT_DIR
F="RL_FASTROLL=1 FAST_TK=1 FAST_COMPACT=1 FAST_OVL=1 FAST_ATT3=1 FLEX_SMALL=1"
grep -v '^#' $1 | while read NAME HOST SET KIND N PR VM SAN RM EXTRA; do
  [ -z "$NAME" ] && continue
  case $HOST in u0) XA="";; base) XA="--host_ckpt base";; *) XA="--host_ckpt $ARMDIR/rl/$HOST/step_000300";; esac
  if [ $KIND = B ] || [ $KIND = SC ]; then SM=1280; BM=384; else if [ $N -gt 4 ]; then SM=5696; BM=192; else SM=2944; BM=384; fi; fi
  BM=${FE_BMAX:-$BM}; SM=${FE_SMAX:-$SM}   # job-level overrides (rung 2 / 3: FE_BMAX=128 FE_SMAX=8192)
  if [ -f $FE_OUT_DIR/${SET}_${NAME}.jsonl ]; then echo "skip $NAME (exists)"; continue; fi
  XARGS="$XA" BMAX=$BM SAN_SMAX=$SM bash $GS_DIR/run_hook.sh fe_${SET}_$NAME ${FE_TREE:-v42} fe_hook.py $F SW_NMAX=$N FE_OUT=$FE_OUT_DIR FE_SET=$SET \
    FE_KIND=$KIND FE_N=$N FE_PR=$PR FE_VM=$VM FE_SAN=$SAN FE_RM=$RM FE_TAG=$NAME FE_SEED=7 $EXTRA < /dev/null \
    | grep -E "^FE |rror|Traceback|host" | cut -c1-400
done
