cd "$(dirname "$0")"
while [ ! -f logs/queue.done ]; do sleep 30; done
for t in bool yn wk cmp cmpE; do bash runq.sh "q3_types.py $t" logs/q3_$t.txt; done
LW_MODEL=Qwen/Qwen2.5-0.5B bash runq.sh "q4_scale.py mid" logs/q4_mid_q25-0.5B.txt
LW_MODEL=Qwen/Qwen2.5-1.5B bash runq.sh "q4_scale.py mid" logs/q4_mid_q25-1.5B.txt
echo done > logs/queue2.done
