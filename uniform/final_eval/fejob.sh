#!/bin/bash
# usage: bash uniform/final_eval/fejob.sh TREE LIST [BMAX SMAX]       (one GPU)
# Generic eval job. TREE as FE_TREE of fe.sh. Every tree but stk runs the adaptive-tries CPU fuzz (test_aw.py) first and
# evaluates at B_max BMAX (default 128) / S_max SMAX (default 8192), the settings of every rung-2/3/4 number.
U="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$U/../env.sh"
S=$1; L=$2
if [ "$S" != stk ]; then
  T=$S; [ -d "$U/trees/$S" ] && T=$U/trees/$S
  H=$(gs_tree "$T" fuzz_$(basename $S)_$$) || exit 2
  (cd $H/stk && STK_DIR=$PWD python3 test_aw.py) || { echo "test_aw FAILED"; exit 3; }
  FE_TREE=$S FE_BMAX=${3:-128} FE_SMAX=${4:-8192} bash $U/fe.sh $L
else
  FE_TREE=stk bash $U/fe.sh $L
fi
