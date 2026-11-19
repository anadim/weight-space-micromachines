cd "$(dirname "$0")/.."   # run from rl/wiring
for LR in 1e-3 1e-2; do
  rm -rf runs/sft_W3e_lr$LR
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 python train_wiring.py --arm W3e --run_dir runs/sft_W3e_lr$LR \
    --sft 200 --sft_n 2000 --sft_kind scaled --prompts 64 --link_lr $LR --freeze_host 1 --dev_every 50 --dev_n 800 \
    --B_max 16 --tok_budget 2048 --fused 0 --diag 1 > runs/sft_W3e_lr$LR.out 2>&1
done
