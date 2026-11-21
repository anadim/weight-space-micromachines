#!/bin/bash
# aw_job3.sh SET [OFF8=1] : after the _wpr_ar compaction fix (ladder-r2). CPU fuzz -> [OFF8 bitwise vs out/val] ->
# rung 2 (WPR + A+B+C+stop4) 4 seeds in ONE process -> rung 2 seed 11 ALONE in a fresh process (leak check: must be
# bitwise equal to the multi-seed run's seed 11) -> rung 1 (WPR only) 4 seeds in one process.
SET=$1
source $WSM_DATA/venv/bin/activate
STK_DIR=$PWD/stk python3 test_aw.py || exit 3
if [ "${2:-0}" = 1 ]; then
  bash aw.sh off8f AW_SETS=dev AW_SEEDS=4242
  python3 - <<'PY'
import gzip, json
a = [json.loads(l) for l in gzip.open('out/val/dev_N8_R1_v1_s1_m0_pr0_T1.0_seed4242.jsonl.gz')]
b = [json.loads(l) for l in gzip.open('out/off8f/dev_N8_R1_v1_s1_m0_pr0_T1.0_seed4242.jsonl.gz')]
same = sum(x['tokens'] == y['tokens'] and x['wf'] == y['wf'] and x['logp'] == y['logp'] for x, y in zip(a, b))
print('OFF8 BITWISE', same, '/', len(a), 'acc', sum(x['ok'] for x in b) / len(b))
PY
fi
R2="SW_NS=16 SW_MAXN=7936 AW_ADAPT=A=0.7/16/4:S=4/0.5:B=3:C=400 WPR=1"
env SAN_SMAX=8192 BMAX=128 bash aw.sh r2f_$SET AW_SETS=$SET AW_SEEDS=7,11,13,17 $R2
env SAN_SMAX=8192 BMAX=128 bash aw.sh r2f1_$SET AW_SETS=$SET AW_SEEDS=11 $R2
SET=$SET python3 - <<'PY'
import gzip, json, glob, os
s = os.environ['SET']
f = lambda d: [json.loads(l) for l in gzip.open(glob.glob(f'out/{d}/{s}_*seed11.jsonl.gz')[0])]
a, b = f(f'r2f_{s}'), f(f'r2f1_{s}')
same = sum(x['tokens'] == y['tokens'] and x['logp'] == y['logp'] for x, y in zip(a, b))
print('LEAKCHECK seed11 multi-vs-fresh bitwise', same, '/', len(a))
PY
bash aw.sh r1f_$SET AW_SETS=$SET AW_SEEDS=7,11,13,17 WPR=1
