cd "$(dirname "$0")/.."   # run from topic_fence/ (needs res/ = fitted/ + test data)
F=$1
for i in $(seq 1 60); do
  for G in 1 0; do
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$G python3 inweights/s6_small.py q17c res/test_q17c_gen_$F.json > s6c_$F.log 2>&1
    if grep -q "silent (checked" s6c_$F.log; then echo "JUDGE_OK $F gpu$G try$i" >> s6c_$F.log; exit 0; fi
    sleep 30
  done
done
echo JUDGE_FAIL >> s6c_$F.log
