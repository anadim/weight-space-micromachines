cd "$(dirname "$0")"
bash runq.sh "q1_multi.py sum2 0,4,8,12,14,16" logs/q1_sum2.txt
bash runq.sh "q2_word.py lost 0,4,8,12,14,16" logs/q2_lost.txt
for t in bool yn wk cmp cmpE; do bash runq.sh "q3_types.py $t" logs/q3_$t.txt; done
bash runq.sh "q1_multi.py chain 0,4,8,12,14,16" logs/q1_chain.txt
LW_MODEL=Qwen/Qwen3-1.7B-Base bash runq.sh "q4_scale.py mid" logs/q4_mid_1.7B.txt
bash runq.sh "q2_word.py rows 0,4,8,12,14,16" logs/q2_rows.txt
bash runq.sh "q1_multi.py sum3 0,4,8,12,14" logs/q1_sum3.txt
LW_MODEL=Qwen/Qwen3-1.7B-Base bash runq.sh "q4_scale.py hard" logs/q4_hard_1.7B.txt
bash runq.sh "q4_scale.py mid" logs/q4_mid_0.6B.txt
echo done > logs/queue1.done
