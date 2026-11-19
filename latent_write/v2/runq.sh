# usage: bash runq.sh "<python args>" logfile  -- waits for a GPU with >=5GB free, runs; retries on OOM
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for attempt in 1 2 3 4 5 6 7 8 9 10; do
  while true; do
    g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1)
    gi=${g%%,*}; free=${g##*, }
    if [ "$free" -gt 5000 ]; then break; fi
    sleep 30
  done
  LW_DEV=cuda:$gi python -u $1 > $2 2>&1
  if grep -q "OutOfMemoryError\|CUDA out of memory" $2; then echo "OOM retry $attempt" >> $2.oom; sleep 60; continue; fi
  exit
done
