#!/bin/bash
# BBH transfer to Qwen3-1.7B-Base / Qwen3-4B-Base (integration v2 setup) + the 0.6B add-one-tool curve.
# Run from the repo root on a GPU machine (the original runs: 2x H100 80GB and 2x A40 48GB). Paths are repo-relative.
# usage: bash bbh_generic/transfer/code/run_transfer.sh <step>
set -e
R=$(pwd); C=$R/bbh_generic/integrate/code; O=$R/bbh_generic/transfer/out; RUNS=$R/bbh_generic/transfer/runs; mkdir -p $O $RUNS
cd $C
case $1 in
  vetoes)   # calc_gen veto per host, solved on GSM8K-train number endings (generic data, never BBH)
    python $R/scripts/solve_veto.py --model Qwen/Qwen3-1.7B-Base --out $RUNS/veto_q17.pt --device cuda
    python $R/scripts/solve_veto.py --model Qwen/Qwen3-4B-Base   --out $RUNS/veto_q4.pt  --device cuda
    python $R/scripts/solve_veto.py --model Qwen/Qwen3-0.6B-Base --out $RUNS/veto_qwen3.pt ;;
  ft)       # same 432 rows and recipe as the 0.6B v2 host (lr 2e-5, 6 epochs, tok_budget 8000); 74 s / 243 s on one H100
    cd $R/bbh_generic/teamA
    python ft.py --model Qwen/Qwen3-1.7B-Base --data ../integrate/data/train_union_v2.jsonl --out $RUNS/ft_v2_17 --lr 2e-5 --epochs 6 --tok_budget 8000 --seed 0
    python ft.py --model Qwen/Qwen3-4B-Base   --data ../integrate/data/train_union_v2.jsonl --out $RUNS/ft_v2_4b --lr 2e-5 --epochs 6 --tok_budget 8000 --seed 0 ;;
  checks)   # host checks before BBH: silence = bit-identity, per-tool e2e, passive C
    VETO=$RUNS/veto_q17.pt python test_integ.py $O/check_ft17_fast.json --model $RUNS/ft_v2_17 --v2 --skip1
    VETO=$RUNS/veto_q4.pt  python test_integ.py $O/check_ft4_fast.json  --model $RUNS/ft_v2_4b --v2 --skip1 ;;
  q17)      # 4 arms on Qwen3-1.7B-Base (bs 32)
    python run27_integ.py $O/q17_base.json       --model Qwen/Qwen3-1.7B-Base --tools none --max_new 600 --bs 32
    python run27_integ.py $O/q17_frozenC.json    --model Qwen/Qwen3-1.7B-Base --tools C  --veto $RUNS/veto_q17.pt --max_new 600 --bs 32
    python run27_integ.py $O/q17_ft_notools.json --model $RUNS/ft_v2_17 --tools none --max_new 600 --bs 32
    python run27_integ.py $O/q17_ft_tools.json   --model $RUNS/ft_v2_17 --tools v2 --veto $RUNS/veto_q17.pt --max_new 600 --bs 32 ;;
  q4)       # 4 arms on Qwen3-4B-Base (the original FT+tools arm used bs 48, the others bs 32)
    python run27_integ.py $O/q4_base.json        --model Qwen/Qwen3-4B-Base --tools none --max_new 600 --bs 32
    python run27_integ.py $O/q4_frozenC.json     --model Qwen/Qwen3-4B-Base --tools C  --veto $RUNS/veto_q4.pt --max_new 600 --bs 32
    python run27_integ.py $O/q4_ft_notools.json  --model $RUNS/ft_v2_4b --tools none --max_new 600 --bs 32
    python run27_integ.py $O/q4_ft_tools.json    --model $RUNS/ft_v2_4b --tools v2 --veto $RUNS/veto_q4.pt --max_new 600 --bs 48 ;;
  curve)    # cumulative add-one-tool curve on the 0.6B v2 host (k0 = FT no tools, k6 = FT + all 8: integrate/README.md)
    M=${FT_HOST:-$R/bbh_generic/integrate/runs/ft_int_v2}
    NAMES=(k1_interval k2_track k3_order k4_truth k5_walk)
    SETS=("interval_gen" "interval_gen,track_gen" "interval_gen,track_gen,order_gen" "interval_gen,track_gen,order_gen,truth_gen" "interval_gen,track_gen,order_gen,truth_gen,turtle_gen")
    for i in ${!NAMES[@]}; do
      python run27_integ.py $O/curve27_${NAMES[$i]}.json --model $M --tools subset --machines "${SETS[$i]}" --veto $RUNS/veto_qwen3.pt --max_new 600 --bs 16
      cp $O/curve27_${NAMES[$i]}.json $O/curve_k$((i+1)).json
    done ;;
  report)
    python $R/bbh_generic/transfer/code/report_transfer.py $O q17_base q17_frozenC q17_ft_notools q17_ft_tools q4_base q4_frozenC q4_ft_notools q4_ft_tools
    python $R/bbh_generic/transfer/code/curve_table.py $O ;;   # curve_table needs q06_ft_notools / q06_ft_tools in $O (integrate v2 runs)
esac
