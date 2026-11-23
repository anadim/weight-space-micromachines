#!/bin/bash
# usage: bash uniform/rl28/gsm8k/fe/fe.sh LIST        (one GPU; configs run one after the other)
# uniform/final_eval/fe.sh for the Qwen3-1.7B-Base host (REPRODUCE §18), via fe/run_hook.sh. LIST: a path or a name in fe/lists/.
# LIST lines: NAME HOST SET KIND N PR VM SAN RM [EXTRA_ENV...]
#   HOST: base (Qwen3-1.7B-Base) | ARM ($ARMDIR17/rl/ARM/step_000300) | ARM@U ($ARMDIR17/rl/ARM/step_<U, 6 digits>) | /abs/host_dir
#   SET / KIND / columns: as uniform/final_eval/fe.sh.
# Env: FE_TREE (default: the 1.7B overlay uniform/rl28/gsm8k/trees/stk_r2q_17, used by every 1.7B number), FE_BMAX / FE_SMAX,
#      FE_MEMFRAC (per-process GPU memory cap; jobE4.sh sets 0.45), FE_OUT_DIR (default uniform/rl28/gsm8k/fe/out/fe; not tracked).
# Every run: T = 1, seed 7 (FE_SEED=.. in a line overrides), fastroll. Rows -> $FE_OUT_DIR/<SET>_<NAME>.jsonl.
U="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$U/../../../env.sh"
FE_OUT_DIR=${FE_OUT_DIR:-$U/out/fe}; mkdir -p $FE_OUT_DIR
L=$1; [ -f "$L" ] || L=$U/lists/$1; [ -f "$L" ] || L=$U/lists/$1.list; [ -f "$L" ] || { echo "no list $1"; exit 2; }
T=${FE_TREE:-$U/../trees/stk_r2q_17}; [ -d "$T" ] && T=$(cd "$T" && pwd)
F="RL_FASTROLL=1 FAST_TK=1 FAST_COMPACT=1 FAST_OVL=1 FAST_ATT3=1 FLEX_SMALL=1"
grep -v '^#' $L | while read NAME HOST SET KIND N PR VM SAN RM EXTRA; do
  [ -z "$NAME" ] && continue
  case $HOST in
    base) XA="--host_ckpt base";; /*) XA="--host_ckpt $HOST";;
    *@*) XA="--host_ckpt $ARMDIR17/rl/${HOST%@*}/step_$(printf %06d ${HOST#*@})";; *) XA="--host_ckpt $ARMDIR17/rl/$HOST/step_000300";;
  esac
  [ -n "$FE_MEMFRAC" ] && XA="$XA --mem_frac $FE_MEMFRAC"
  if [ $KIND = B ] || [ $KIND = SC ] || [ $KIND = SCH ]; then SM=1280; BM=384; else if [ $N -gt 4 ]; then SM=5696; BM=192; else SM=2944; BM=384; fi; fi
  BM=${FE_BMAX:-$BM}; SM=${FE_SMAX:-$SM}
  if [ -f $FE_OUT_DIR/${SET}_${NAME}.jsonl ]; then echo "skip $NAME (exists)"; continue; fi
  XARGS="$XA" BMAX=$BM SAN_SMAX=$SM bash $U/run_hook.sh fe17_${SET}_$NAME "$T" fe_hook.py $F SW_NMAX=$N FE_OUT=$FE_OUT_DIR FE_SET=$SET \
    FE_KIND=$KIND FE_N=$N FE_PR=$PR FE_VM=$VM FE_SAN=$SAN FE_RM=$RM FE_TAG=$NAME FE_SEED=7 $EXTRA < /dev/null \
    | grep --line-buffered -E "^FE |rror|Traceback|host" | cut -c1-400
done
