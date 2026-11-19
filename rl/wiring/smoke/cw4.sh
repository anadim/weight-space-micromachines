cd "$(dirname "$0")/.."   # run from rl/wiring
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1
C="--dev_extra '' --dev_n 40 --B_max 32 --tok_budget 2048 --eval_all 0"
rm -rf runs/smk4_*
timeout 900 python train_wiring.py --arm CW --run_dir runs/smk4_CW --updates 2 --dev_every 2 --ckpt_every 2 --kl 0.01 --dev_extra "" --dev_n 40 --B_max 32 --tok_budget 2048 --eval_all 0 > runs/smk4_CW.out 2>&1
timeout 900 python train_wiring.py --arm CW --run_dir runs/smk4_CWeval --eval_ckpt runs/smk4_CW/step_000002 --interp 1 --test_n 40 --dev_n 40 --B_max 32 --tok_budget 2048 > runs/smk4_CWeval.out 2>&1
for A in CWinR CWinP CWinC; do timeout 900 python train_wiring.py --arm $A --run_dir runs/smk4_$A --updates 1 --dev_every 1 --dev_extra "" --dev_n 40 --B_max 32 --tok_budget 2048 --eval_all 0 > runs/smk4_$A.out 2>&1; done
echo done > runs/cw4.done
