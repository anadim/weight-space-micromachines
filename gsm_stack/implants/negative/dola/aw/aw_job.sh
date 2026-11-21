#!/bin/bash
# aw_job.sh SETS [OFF8=1]: CPU fuzz (test_aw.py) -> [all-off NS=8 val run, compare bitwise to out/val] -> A+B+C+stop4 run on SETS.
SETS=$1
source $WSM_DATA/venv/bin/activate
STK_DIR=$PWD/stk python3 test_aw.py || exit 3
if [ "${2:-0}" = 1 ]; then
  bash aw.sh off8 AW_SETS=dev AW_SEEDS=4242
  python3 - <<'PY'
import gzip, json
a = [json.loads(l) for l in gzip.open('out/val/dev_N8_R1_v1_s1_m0_pr0_T1.0_seed4242.jsonl.gz')]
b = [json.loads(l) for l in gzip.open('out/off8/dev_N8_R1_v1_s1_m0_pr0_T1.0_seed4242.jsonl.gz')]
same = sum(x['tokens'] == y['tokens'] and x['wf'] == y['wf'] and x['logp'] == y['logp'] for x, y in zip(a, b))
print('OFF8 BITWISE', same, '/', len(a), 'acc', sum(x['ok'] for x in b) / len(b))
PY
fi
env SAN_SMAX=8192 BMAX=128 bash aw.sh abcs_$(echo $SETS | tr , _) AW_SETS=$SETS AW_SEEDS=7,11,13,17 SW_NS=16 SW_MAXN=7936 AW_ADAPT=A=0.7/16/4:S=4/0.5:B=3:C=400
