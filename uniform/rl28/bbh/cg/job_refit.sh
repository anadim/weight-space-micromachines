#!/bin/bash
# Call-gate probe REFIT (task [24] recipe, see uniform/rl24/bbh/cg/job_refit.sh): the ORIGINAL callgate recipe
# (bbh_generic/callgate/code/run_callgate.sh states + probe), own data only (teams' sets, nocall prompts), never BBH.
# [28]: the same recipe on the 1.7B call-FT host $RL28_DATA/bbh/ft17 with the 1.7B calc veto -> probe_ft17.pt
# usage: bash uniform/rl28/bbh/cg/job_refit.sh states|probe|sets|all     (one GPU; results -> ./out, states -> $RL28_DATA/bbh/cg)
#   PROBE_OUT=probe_ft17_rederive.pt  write the probe to a side file (setup2.sh); NOSETS=1  skip the sets step
set -eo pipefail
X="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$X"
source "$X/../../../env.sh"                     # uniform/env.sh: WSM_ROOT, RL28_DATA, BBH_DIR, BBH_COT
D=$RL28_DATA/bbh/cg; mkdir -p $D $D/tmp out; export TMPDIR=$D/tmp
R=$WSM_ROOT; C=$R/bbh_generic/callgate/code
CC=$X/../../../rl24/bbh/cg/code                 # cg_sets_gate_c.py / cg_compiled.py (mode compiled), shared with rl24
export WSM_REPO=$R FT_HOST=$RL28_DATA/bbh/ft17 VETO=$WSM_ROOT/results/gsm_bigger/veto_q17.pt CG_RESULTS=$D
export PYTHONPATH=$C:$CC${PYTHONPATH:+:$PYTHONPATH}
O=$(pwd)/out
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader || true
sha256sum $FT_HOST/*.safetensors | tee $O/host_sha.txt
st_states() { cd $C; for k in 2 3 4; do [ -f $D/states$k.pt ] || python cg_collect_states$k.py $D/states$k.pt 2>&1 | grep -v Warning; done; cd -; }
st_probe() {
  cd $C
  { python cg_select.py $D/states2.pt $D/states3.pt $D/states4.pt
    python cg_loso.py   $D/states2.pt $D/states3.pt $D/states4.pt
    python cg_logo.py   $D/states2.pt $D/states3.pt $D/states4.pt
    CG_LAM=0.1 python cg_train_probe.py $D/${PROBE_OUT:-probe_ft17.pt} 0.5 $D/states2.pt $D/states3.pt $D/states4.pt; } 2>&1 | grep -v Warning | tee $O/probe_log.txt
  cp $D/${PROBE_OUT:-probe_ft17.pt} $O/; sha256sum $D/probe_ft17*.pt | tee -a $O/probe_log.txt; cd -
}
st_sets() {
  cd $C
  for m in none gate compiled; do [ -f $O/sets_$m.json ] || [ -n "$NOSETS" ] || python $CC/cg_sets_gate_c.py $O/sets_$m.json --mode $m --probe $D/probe_ft17.pt --sets A_oos2,B_oos,C_ood --bs 16 2>&1 | grep -v Warning; done
  cd -
}
case $1 in
  states) st_states;; probe) st_probe;; sets) st_sets;;
  all) st_states; st_probe; st_sets;;
esac
