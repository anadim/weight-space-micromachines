cd "$(dirname "$0")/.."   # run from rl/wiring
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1
for A in W2w CW CWlean CWout CWin; do
  rm -rf runs/cwsmk_$A
  timeout 600 python train_wiring.py --arm $A --run_dir runs/cwsmk_$A --updates 2 --dev_every 2 --dev_extra "" --dev_n 40 \
     --B_max 32 --tok_budget 2048 --kl 0.01 > runs/cwsmk_$A.out 2>&1
done
echo done > runs/cwsmk.done
