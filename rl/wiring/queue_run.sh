#!/bin/bash
# Sequential RL-wiring runner for H100 machine GPU 0, shared with the ablation agent.
# Gates each launch on free GPU memory (never exceed ~35 GB; our peak ~13 GB), retries on OOM, never kills other jobs.
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
NEED=16000           # MiB free required before we launch a run (our peak ~13 GB + margin)
LOG=runs/queue.log
mkdir -p runs
echo "=== queue start $(date) ===" >> $LOG

# arm  seed  extra-args
RUNS=(
  "2    0  "
  "1    0  "
  "W2   0  "
  "W3z  0  "
  "W1   0  --impl_lr 1e-5"
  "W3r  0  "
  "W4   0  --impl_lr 1e-5"
  "W1   0  --impl_lr 1e-6"
)

wait_mem () {
  while true; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n 1p)
    if [ "$FREE" -ge "$NEED" ]; then
      # confirm it stays free for ~20s (avoid launching into a transient dip of the neighbor)
      sleep 20
      FREE2=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n 1p)
      if [ "$FREE2" -ge "$NEED" ]; then return 0; fi
    fi
    echo "$(date +%H:%M:%S) waiting for GPU0 mem (free=${FREE}MiB need=${NEED})" >> $LOG
    sleep 30
  done
}

for r in "${RUNS[@]}"; do
  set -- $r
  ARM=$1; SEED=$2; shift 2; EXTRA="$@"
  TAG="${ARM}_s${SEED}"
  [ -n "$EXTRA" ] && TAG="${ARM}_s${SEED}_$(echo $EXTRA | tr -d ' -' | tr -c 'a-zA-Z0-9' '_')"
  DIR=runs/$TAG
  if [ -f "$DIR/step_000064/DONE" ]; then echo "skip $TAG (done)" >> $LOG; continue; fi
  for attempt in 1 2 3 4 5; do
    wait_mem
    echo "$(date +%H:%M:%S) launch $TAG attempt $attempt (free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|sed -n 1p))" >> $LOG
    python train_wiring.py --arm $ARM --seed $SEED --run_dir $DIR \
       --updates 64 --dev_every 16 --dev_n 800 --B_max 64 --tok_budget 4096 $EXTRA \
       > $DIR.out 2>&1
    if [ -f "$DIR/step_000064/DONE" ]; then echo "$(date +%H:%M:%S) OK $TAG" >> $LOG; break; fi
    if grep -q "OutOfMemory" "$DIR.out"; then
      echo "$(date +%H:%M:%S) OOM $TAG attempt $attempt, retrying" >> $LOG; sleep 60; continue
    fi
    echo "$(date +%H:%M:%S) FAIL $TAG (non-OOM) attempt $attempt" >> $LOG
    tail -3 "$DIR.out" >> $LOG; sleep 30
  done
done
echo "=== queue done $(date) ===" >> $LOG
