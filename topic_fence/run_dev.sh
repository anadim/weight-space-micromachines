#!/bin/sh
# dev pipeline for the 1.7B host (run from topic_fence/): merge dev runs a+b, states, probe sweeps, monitors.
# Needs data/dev.jsonl and data/dev_b.jsonl from gen_dev.py (see README).
cd "$(dirname "$0")"; mkdir -p res
HOST=${HOST:-Qwen/Qwen3-1.7B}
python -c "
import json
R=[json.loads(l) for f in ['data/dev.jsonl','data/dev_b.jsonl'] for l in open(f)]
seen=set(); out=open('data/dev_all.jsonl','w')
for r in R:
    k=r['text'].lower()[:200]
    if k in seen: continue
    seen.add(k); out.write(json.dumps(r,ensure_ascii=False)+'\n')
print(len(seen))"
python s1_states.py $HOST data/dev_all.jsonl res/dev_states_q17.pt
python s2_sweep.py math res/dev_states_q17.pt data/dev_all.jsonl res/probe_math_q17.json > s2_math.log 2>&1
python s2_sweep.py ml res/dev_states_q17.pt data/dev_all.jsonl res/probe_ml_q17.json > s2_ml.log 2>&1
python s4_monitor.py math res/mon_math_q17.json > s4_math.log 2>&1
python s4_monitor.py ml res/mon_ml_q17.json > s4_ml.log 2>&1
