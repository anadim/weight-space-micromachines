cd "$(dirname "$0")/.."   # run from rl/wiring
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1
rm -rf runs/cwsmk2_CW; timeout 600 python train_wiring.py --arm CW --run_dir runs/cwsmk2_CW --updates 2 --dev_every 2 --dev_extra "" --dev_n 40 --B_max 32 --tok_budget 2048 --kl 0.01 > runs/cwsmk2_CW.out 2>&1
rm -rf runs/sft3_W3e_lr1e-2; python train_wiring.py --arm W3e --run_dir runs/sft3_W3e_lr1e-2 --sft 200 --sft_n 2000 --sft_kind scaled \
    --sft_traces runs/traces_A_scaled.jsonl --prompts 64 --link_lr 1e-2 --freeze_host 1 --dev_every 200 --dev_n 800 \
    --B_max 16 --tok_budget 2048 --fused 0 > runs/sft3_W3e_lr1e-2.out 2>&1
echo done > runs/cw2.done
