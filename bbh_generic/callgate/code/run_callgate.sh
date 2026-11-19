#!/bin/bash
# Call gate for the v2 integration host. Run from the repo root on a GPU (original: one A40 48GB + one H100 80GB).
# Needs: the v2 host (bbh_generic/integrate/README.md "v2 host"; set FT_HOST), the 0.6B calc veto (VETO, scripts/solve_veto.py),
# BBH data (data/download_bbh.sh). usage: bash bbh_generic/callgate/code/run_callgate.sh <step>
set -e
R=$(pwd); C=$R/bbh_generic/callgate/code; S=${CG_RESULTS:-$R/bbh_generic/callgate/results}; mkdir -p $S
export CG_RESULTS=$S; cd $C
case $1 in
  states)   # opener-step hidden states on OWN data only (collections 2, 3, 4 are the ones probe v3 uses; 1 = tools-off pilot)
    python cg_collect_states2.py $S/states2.pt
    python cg_collect_states3.py $S/states3.pt
    python cg_collect_states4.py $S/states4.pt ;;
  probe)    # CPU, seconds. Shipped states reproduce the shipped probe_v3.pt to 5e-6 in w.
    python cg_select.py $S/states2.pt $S/states3.pt $S/states4.pt          # own-data model selection (lambda, threshold rule)
    python cg_loso.py   $S/states2.pt $S/states3.pt $S/states4.pt          # leave-one-style-out positives
    python cg_logo.py   $S/states2.pt $S/states3.pt $S/states4.pt          # leave-one-genre-out negatives
    CG_LAM=0.1 python cg_train_probe.py $S/probe_v3.pt 0.5 $S/states2.pt $S/states3.pt $S/states4.pt ;;
  bbh)      # real BBH, 27 tasks, ONE look per arm
    python cg_run27_gate.py $S/bbh_gate.json    --mode gate    --probe $S/probe_v3.pt --bs 32 --max_new 600
    python cg_run27_gate.py $S/bbh_breaker.json --mode breaker                        --bs 16 --max_new 600 ;;
  sets)     # own held-out sets (A_oos2, B_oos, C_ood)
    for m in none gate breaker; do
      python cg_sets_gate.py $S/sets_${m}_heldout.json --mode $m --probe $S/probe_v3.pt --sets A_oos2,B_oos,C_ood --bs 16; done ;;
  arc)      # ARC-Easy (400 val, 3-shot CoT), never used for training; needs the HF datasets library
    for m in none gate breaker; do python cg_arc.py $S/arc_$m.json --mode $m --probe $S/probe_v3.pt --n 400; done ;;
  report)   # CPU; baseline = the published v2+tools transcripts (integrate/results)
    python cg_report.py $R/bbh_generic/integrate/results "v2 + GATE=$S/bbh_gate.json" "v2 + BREAKER=$S/bbh_breaker.json"
    python cg_silence.py $R/bbh_generic/integrate/results $S ;;
esac
