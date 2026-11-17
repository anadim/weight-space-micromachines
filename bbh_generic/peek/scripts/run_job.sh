#!/bin/bash
# run_job.sh NAME  -- one arm of the real-BBH peek; writes $OUT/NAME.json
# Env: BBH_DIR (official BBH task json files; default <repo>/data/bbh/data, from data/download_bbh.sh),
#      TEAM_A_FT (Team A host fine-tuned on teamA/data/train3 = ft_v3; NOT released, recreate with teamA/ft.py),
#      TEAM_B_FT (Team B host ft3; NOT released, recreate with teamB/ft_B.py), VETO (calc veto, scripts/solve_veto.py).
set -e
P="$(cd "$(dirname "$0")/.." && pwd)"; REPO="$(cd "$P/../.." && pwd)"; C=$REPO/bbh_generic
D=${BBH_DIR:-$REPO/data/bbh/data}; O=${OUT:-$P/out}; VETO=${VETO:-$REPO/runs/veto_qwen3.pt}
TEAM_A_FT=${TEAM_A_FT:-$REPO/runs/teamA_ft_v3}; TEAM_B_FT=${TEAM_B_FT:-$REPO/runs/teamB_ft3}
export PYTHONPATH=$REPO; mkdir -p "$O"; cd "$P"
[ -f "$P/bbh_A.jsonl" ] || python scripts/make_A_jsonl.py "$D" "$P/bbh_A.jsonl"   # Team-A-format copy of the 6 BBH task files (not shipped)
n=$1
case $n in
  C_base)  python $C/teamC/run_gen.py $O/$n.json --data $D --tasks multistep_arithmetic_two,boolean_expressions,object_counting --machines none --cot $C/teamC/cot --veto $VETO ;;
  C_tools) python $C/teamC/run_gen.py $O/$n.json --data $D --tasks multistep_arithmetic_two,boolean_expressions,object_counting --machines expr_gen,calc_gen,bool_gen --cot $C/teamC/cot --veto $VETO ;;
  B_base)        python scripts/peek_B.py $O/$n.json --sets temporal,navigate,lies --tools none ;;
  B_ft3_notools) python scripts/peek_B.py $O/$n.json --sets temporal,navigate,lies --tools none --model $TEAM_B_FT ;;
  B_ft3_tools)   python scripts/peek_B.py $O/$n.json --sets temporal,navigate,lies --tools all --model $TEAM_B_FT ;;
  A_base)         python $C/teamA/run_eval.py $P/bbh_A.jsonl $O/$n.json ;;
  A_ftv3_notools) python $C/teamA/run_eval.py $P/bbh_A.jsonl $O/$n.json --model $TEAM_A_FT ;;
  A_ftv3_tools)   python $C/teamA/run_eval.py $P/bbh_A.jsonl $O/$n.json --model $TEAM_A_FT --tools track_gen,order_gen ;;
esac
