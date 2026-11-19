cd "$(dirname "$0")/.."   # run from rl/wiring
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1
for A in W2ws CW; do rm -rf runs/smk3_$A; timeout 900 python train_wiring.py --arm $A --run_dir runs/smk3_$A --updates 1 --dev_every 1 --dev_extra "" --dev_n 40 --B_max 32 --tok_budget 2048 --kl 0.01 --eval_all 0 --ckpt_every 1 > runs/smk3_$A.out 2>&1; done
echo done > runs/cw3.done
