#!/bin/bash
# rl28 [28]: fast-engine exactness check on Qwen3-1.7B-Base with the CHOSEN export (HF + reference Executor vs IFEngine).
# bash uniform/rl28/ifeval/code/eng/job_check17.sh [N]   (the setup job's packed check OOMed at the engine build under --mem_frac 0.45)
cd $(dirname $0)/..; C=$PWD; source env.sh
R=$DATA/setup; O=$C/../out/setup; mkdir -p $R $O; exec > >(tee -a $R/check17.log) 2>&1
JID=$(date -u +%H%M%S); CV=${CUDA_VISIBLE_DEVICES:-0}; CV=${CV%%,*}
case $CV in GPU-*) GI=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v u=$CV '$2==u{print $1}');; *) GI=$CV;; esac
nvidia-smi dmon -s um -d 10 -i $GI -o DT > $O/dmon_chk_$JID.txt 2>&1 & DM=$!
(cd heldout; build_tables)
SD=$DATA/eng/sd_$(cat $DATA/eng/CHOSEN).pt; echo "$(date -u +%FT%TZ) check with $SD"
rm -rf $R/chk2; cd $C/eng
python check.py --model Qwen/Qwen3-1.7B-Base --sd $SD --data $C/gen/dev.jsonl --n ${1:-64} --max_new 1024 --out $R/chk2 --fmt chat
rc=$?; echo "$(date -u +%FT%TZ) check rc=$rc"
python - $R/chk2/check.jsonl > $O/check_summary.txt <<'PY'
import json,sys
for l in open(sys.argv[1]):
    d=json.loads(l); d.pop('divs',None) if 'divs' in d and len(json.dumps(d['divs']))>4000 else None; print(json.dumps(d))
PY
cat $O/check_summary.txt | cut -c1-300
kill $DM; awk '$1 !~ /^#/ {s+=$4; n++} END {if (n) printf "mean SM%% %.1f over %d samples\n", s/n, n}' $O/dmon_chk_$JID.txt
exit $rc
