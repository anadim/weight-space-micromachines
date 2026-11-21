#!/bin/bash
# lad_job1.sh : WPR_K equivalence. (1) wprk_fuzz (real host kernel, q.k vs programmatic reset / exact);
# (2) greedy, first 64 dev items, seed 7, rung-2 config: pr1 (programmatic reset, reference), WPR K=0 (+_wpr_ar fix), K=1, K=2;
# (3) N=8 sampled dev seed 7 (600), same config: pr1, K=1, K=2 (K=2 must be token-identical to pr1). One process per run.
source $WSM_DATA/venv/bin/activate
for v in $(env | grep -o '^TRITON_[A-Z_]*PATH'); do unset $v; done
TB=$WSM_DATA/venv/lib/python3.12/site-packages/triton/backends/nvidia/bin
export TRITON_PTXAS_PATH=$TB/ptxas TRITON_CUOBJDUMP_PATH=$TB/cuobjdump TRITON_NVDISASM_PATH=$TB/nvdisasm
mkdir -p $WSM_DATA/implants/triton_cache_aw; export TRITON_CACHE_DIR=$WSM_DATA/implants/triton_cache_aw
export WIRING_DIR=$PWD/wiring STK_DIR=$PWD/stk
STK_DIR=$PWD/stk python3 test_aw.py || exit 3
python3 stk/wprk_fuzz.py || echo "WPRK fuzz rc=$? (continuing)"
STK_DIR=$PWD/stk python3 test_vf.py || echo "VF fuzz rc=$? (continuing)"
R2="SW_NS=16 SW_MAXN=7936 AW_ADAPT=A=0.7/16/4:S=4/0.5:B=3:C=400"
run() { env SAN_SMAX=8192 BMAX=128 bash aw.sh "$@"; }
run eq_gr_pr1 AW_SETS=dev AW_SEEDS=7 AW_NLIM=64 AW_GREEDY=1 AW_PR=1 $R2
for k in 0 1 2; do run eq_gr_k$k AW_SETS=dev AW_SEEDS=7 AW_NLIM=64 AW_GREEDY=1 WPR=1 WPR_K=$k $R2; done
run eq_s7_pr1 AW_SETS=dev AW_SEEDS=7 AW_PR=1 $R2
for k in 2 1; do run eq_s7_k$k AW_SETS=dev AW_SEEDS=7 WPR=1 WPR_K=$k $R2; done
# VF smoke: 64 dev items, sampled seed 7, rung-2 config (WPR K=0 fixed) + AW_VF=1, then the offline register / pick check
run vf_smoke AW_SETS=dev AW_SEEDS=7 AW_NLIM=64 WPR=1 AW_VF=1 $R2
(export HF_HOME=$WSM_DATA/hf HF_HUB_CACHE=$WSM_DATA/hf/hub HF_HUB_OFFLINE=1; python3 vf_check.py out/vf_smoke/*.jsonl.gz)
python3 eq_cmp.py out/eq_gr_pr1 out/eq_gr_k0 out/eq_gr_k1 out/eq_gr_k2
python3 eq_cmp.py out/eq_s7_pr1 out/eq_s7_k2 out/eq_s7_k1
echo LADJOB1DONE
