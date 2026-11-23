#!/bin/bash
# usage: bash uniform/final_eval/fe.sh LIST        (one GPU; configs run one after the other)
# The uniform-row eval harness (REPRODUCE §17): gsm_stack/eval/fe.sh plus the host forms, the SCH kind and the memory cap
# of the uniform runs. LIST: a path, or a name looked up in uniform/final_eval/lists/, then gsm_stack/eval/lists/.
# LIST lines: NAME HOST SET KIND N PR VM SAN RM [EXTRA_ENV...]
#   HOST: u0 ($CKPT_U0) | base (Qwen3-0.6B-Base) | ARM ($ARMDIR/rl/ARM/step_000300) | ARM@U ($ARMDIR/rl/ARM/step_<U, 6 digits>)
#         | ft_int_v2 ($FT2_HOST: the BBH call-FT host, REPRODUCE §6; uniform/env.sh) | /abs/host_dir
#   SET: test | hard | gsmplus (GSM-Plus testmini, data/prepare_gsmplus.py) | dev | dev2
#   KIND: S serial (N tries) | B (S2F, one attempt) | SC / SCH (N independent samples + plain majority; SCH = host only)
#         | H / C4 / D (ladder: host only / + calc_s4 / + calc_s4 + chain2). Columns: trees/stk/stk/fe_hook.py.
# Env: FE_TREE  code tree: stk | stk_r2 | stk_r2q | stk_r3 | stk_r4 (overlays in uniform/final_eval/trees/), any gsm_stack
#               tree name (v42 | r2 | r3) or an overlay path (default stk)
#      FE_BMAX / FE_SMAX override B_max / SAN_SMAX; FE_MEMFRAC caps the per-process GPU memory (packed jobs, fepack.sh);
#      FE_OUT_DIR (default uniform/final_eval/out/fe, as the original layout; not tracked). Every run: T = 1, seed 7, fastroll. Rows -> $FE_OUT_DIR/<SET>_<NAME>.jsonl.
U="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$U/../env.sh"
FE_OUT_DIR=${FE_OUT_DIR:-$U/out/fe}; mkdir -p $FE_OUT_DIR
L=$1; [ -f "$L" ] || L=$U/lists/$1; [ -f "$L" ] || L=$GS_DIR/eval/lists/$1; [ -f "$L" ] || { echo "no list $1"; exit 2; }
T=${FE_TREE:-stk}
if [ -d "$U/trees/$T" ]; then T=$U/trees/$T; elif [ -d "$T" ]; then T=$(cd "$T" && pwd); fi
F="RL_FASTROLL=1 FAST_TK=1 FAST_COMPACT=1 FAST_OVL=1 FAST_ATT3=1 FLEX_SMALL=1"
grep -v '^#' $L | while read NAME HOST SET KIND N PR VM SAN RM EXTRA; do
  [ -z "$NAME" ] && continue
  case $HOST in
    u0) XA="";; base) XA="--host_ckpt base";; ft_int_v2) XA="--host_ckpt $FT2_HOST";; /*) XA="--host_ckpt $HOST";;
    *@*) XA="--host_ckpt $ARMDIR/rl/${HOST%@*}/step_$(printf %06d ${HOST#*@})";; *) XA="--host_ckpt $ARMDIR/rl/$HOST/step_000300";;
  esac
  [ -n "$FE_MEMFRAC" ] && XA="$XA --mem_frac $FE_MEMFRAC"
  if [ $KIND = B ] || [ $KIND = SC ] || [ $KIND = SCH ]; then SM=1280; BM=384; else if [ $N -gt 4 ]; then SM=5696; BM=192; else SM=2944; BM=384; fi; fi
  BM=${FE_BMAX:-$BM}; SM=${FE_SMAX:-$SM}   # job-level overrides (rung 2 / 3 / 4: FE_BMAX=128 FE_SMAX=8192)
  if [ -f $FE_OUT_DIR/${SET}_${NAME}.jsonl ]; then echo "skip $NAME (exists)"; continue; fi
  XARGS="$XA" BMAX=$BM SAN_SMAX=$SM bash $GS_DIR/run_hook.sh fe_${SET}_$NAME "$T" fe_hook.py $F SW_NMAX=$N FE_OUT=$FE_OUT_DIR FE_SET=$SET \
    FE_KIND=$KIND FE_N=$N FE_PR=$PR FE_VM=$VM FE_SAN=$SAN FE_RM=$RM FE_TAG=$NAME FE_SEED=7 $EXTRA < /dev/null \
    | grep --line-buffered -E "^FE |rror|Traceback|host" | cut -c1-400
done
