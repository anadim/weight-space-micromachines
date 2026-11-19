cd "$(dirname "$0")/.."   # run from rl/wiring
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1
S="--dev_extra '' --dev_n 40 --B_max 16 --tok_budget 2048 --fused 0 --eval_all 0"
rm -rf runs/smk7_*
for A in WOn WOc; do timeout 900 python train_wiring.py --arm $A --run_dir runs/smk7_$A --updates 2 --dev_every 2 --dev_extra "" --dev_n 40 --B_max 16 --tok_budget 2048 --fused 0 --eval_all 0 --iw_eps 0.05 > runs/smk7_$A.out 2>&1; done
timeout 900 python train_wiring.py --arm CWinB --run_dir runs/smk7_CWinB --updates 2 --dev_every 2 --dev_extra "" --dev_n 40 --B_max 16 --tok_budget 2048 --fused 0 --eval_all 0 --freeze_host 1 --break_at 1 --break_dir on2off --iw_eps 0.05 --cw_lr 1e-3 --link_eps 1e-8 > runs/smk7_CWinB.out 2>&1
timeout 900 python train_wiring.py --arm W3e --run_dir runs/smk7_TP --eval_ckpt runs/L300_W3d_s0_initlinksrunssft3W3elr1e2linksfinalpt/step_000300 --host_ckpt ../gsm_grpo/runs/long/a1_u300_s1/step_000300 --test_sets dev,hard --test_n 30 --dev_n 40 --B_max 16 --tok_budget 2048 --fused 0 > runs/smk7_TP.out 2>&1
echo done > runs/smk7.done
