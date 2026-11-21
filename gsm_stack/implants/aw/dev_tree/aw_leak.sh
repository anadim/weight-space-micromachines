#!/bin/bash
# aw_leak.sh TAG [WPR=1|0] : multi-run state-leak test (ladder-r2). AW_PRECAP=1 captures every graph bucket up front, so the
# only difference between "seed 11 after seed 7 in one process" and "seed 11 alone in a fresh process" is state left by the
# earlier run: rows must be bitwise equal if there is no leak. Rung-2 config (adapt on), dev2.
TAG=$1; W=${2:-1}
source $WSM_DATA/venv/bin/activate
R2="SW_NS=16 SW_MAXN=7936 AW_ADAPT=A=0.7/16/4:S=4/0.5:B=3:C=400 WPR=$W AW_PRECAP=1"
env SAN_SMAX=8192 BMAX=128 bash aw.sh lk${TAG}_m AW_SETS=dev2 AW_SEEDS=7,11 $R2
env SAN_SMAX=8192 BMAX=128 bash aw.sh lk${TAG}_f AW_SETS=dev2 AW_SEEDS=11 $R2
T=$TAG python3 - <<'PY'
import gzip, json, glob, os
t = os.environ['T']
f = lambda d: [json.loads(l) for l in gzip.open(glob.glob(f'out/{d}/dev2_*seed11.jsonl.gz')[0])]
a, b = f(f'lk{t}_m'), f(f'lk{t}_f')
same = sum(x['tokens'] == y['tokens'] and x['logp'] == y['logp'] for x, y in zip(a, b))
print('LEAKTEST', t, 'seed11 after-seed7 vs fresh bitwise', same, '/', len(a), 'acc', sum(x['ok'] for x in a) / len(a), sum(x['ok'] for x in b) / len(b))
PY
