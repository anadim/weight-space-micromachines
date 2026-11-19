cd "$(dirname "$0")/.."   # run from rl/wiring
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1
python gen_traces.py --out runs/traces_A_scaled.jsonl --n_scaled 6000 --B_max 64 > runs/gen_traces.out 2>&1
for LR in 1e-3 1e-2; do
  rm -rf runs/sft2_W3e_lr$LR
  python train_wiring.py --arm W3e --run_dir runs/sft2_W3e_lr$LR --sft 200 --sft_n 2000 --sft_kind scaled \
    --sft_traces runs/traces_A_scaled.jsonl --prompts 64 --link_lr $LR --freeze_host 1 --dev_every 50 --dev_n 800 \
    --B_max 16 --tok_budget 2048 --fused 0 --diag 1 > runs/sft2_W3e_lr$LR.out 2>&1
done
echo ALLDONE > runs/sft2.done
