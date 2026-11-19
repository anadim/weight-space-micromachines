#!/bin/bash
# Sequential RL-wiring runner for A40 machine GPU 0 (A40, free). GPU 1 runs another job: never touch it.
# A40 needs FLEX_SMALL=1 (smaller flex tiles) and fused=0 (persistent implant kernel exceeds A40 shared mem).
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export FLEX_SMALL=1
NEED=22000           # MiB free on GPU0 before launching (our peak ~15 GB + margin)
LOG=runs/queue.log
mkdir -p runs
echo "=== A40 machine queue start $(date) ===" >> $LOG

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
  "2    1  "
  "W2   1  "
  "W3z  1  "
)

wait_mem () {
  while true; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | sed -n 1p)
    if [ "$FREE" -ge "$NEED" ]; then
      sleep 15
      FREE2=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | sed -n 1p)
      if [ "$FREE2" -ge "$NEED" ]; then return 0; fi
    fi
    echo "$(date +%H:%M:%S) waiting GPU0 mem (free=${FREE} need=${NEED})" >> $LOG
    sleep 30
  done
}

for r in "${RUNS[@]}"; do
  set -- $r
  ARM=$1; SEED=$2; shift 2; EXTRA="$@"
  TAG="${ARM}_s${SEED}"
  [ -n "$EXTRA" ] && TAG="${ARM}_s${SEED}_$(echo $EXTRA | tr -cd 'a-zA-Z0-9')"
  DIR=runs/$TAG
  if [ -f "$DIR/step_000064/DONE" ]; then echo "skip $TAG (done)" >> $LOG; continue; fi
  for attempt in 1 2 3; do
    wait_mem
    echo "$(date +%H:%M:%S) launch $TAG attempt $attempt free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0|sed -n 1p)" >> $LOG
    python train_wiring.py --arm $ARM --seed $SEED --run_dir $DIR \
       --updates 64 --dev_every 16 --dev_n 800 --B_max 128 --tok_budget 8192 --fused 0 $EXTRA \
       > $DIR.out 2>&1
    if [ -f "$DIR/step_000064/DONE" ]; then echo "$(date +%H:%M:%S) OK $TAG" >> $LOG; break; fi
    if grep -q "OutOfMemory\|out of resource" "$DIR.out"; then
      echo "$(date +%H:%M:%S) OOM $TAG attempt $attempt" >> $LOG; sleep 60; continue
    fi
    echo "$(date +%H:%M:%S) FAIL $TAG non-OOM" >> $LOG; tail -4 "$DIR.out" >> $LOG; break
  done
done
echo "=== A40 machine queue done $(date) ===" >> $LOG
