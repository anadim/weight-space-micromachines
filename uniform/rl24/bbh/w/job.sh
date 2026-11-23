#!/bin/bash
# rl24/bbh work jobs (REPRODUCE §17.8), one GPU each. usage (from uniform/rl24/bbh/w):
#   bash job.sh data                  : train/dev look-alike data (CPU) -> $D/data + out/data
#   bash job.sh hc NAME HOST TOOLS    : host check on dev (hostcheck.py) -> out/hc_NAME.json
#   bash job.sh chk NAME HOST TOOLS GATE [args]  : engine exactness/speed checks (rl_bbh.py check) -> out/chk_NAME.json
#   bash job.sh rl RUN HOST TOOLS GATE SEED [args] : GRPO run (resumable; re-run to continue) -> $D/runs/RUN, out/RUN
#   bash job.sh eval NAME HOST TOOLS GATE WEIGHTS [args] : BBH-27 (WEIGHTS '-' = the host itself) -> out/eval_NAME.json
#   bash job.sh ana OUT.md --row LABEL=a.json,.. --cmp X-Y : table + paired bootstrap CIs (analyze.py; CPU)
set -eo pipefail
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$H"
source "$H/../../../env.sh"                     # uniform/env.sh: WSM_ROOT, RL24_DATA, FT2_HOST, BBH_DIR, BBH_COT, WSM_VETO
D=$RL24_DATA/bbh; mkdir -p $D/tmp out; export TMPDIR=$D/tmp
export WSM_REPO=$WSM_ROOT WSM_REPO_ROOT=$WSM_ROOT VETO=$WSM_VETO FT_HOST=$FT2_HOST
# engine files shared unchanged with gsm_stack/engine (not duplicated here)
for f in wiring/fastqwen.py wiring/hostkern.py stk/tattn.py stk/tk.py stk/tk_compile.py stk/lmhead.cu; do
  [ -f "$(basename $f)" ] || cp "$WSM_ROOT/gsm_stack/engine/$f" .; done
host_of() { case $1 in base) echo Qwen/Qwen3-0.6B-Base;; inst) echo Qwen/Qwen3-0.6B;; ft2) echo $FT2_HOST;; *) echo $1;; esac; }
case $1 in
  data)
    python bbhdata.py $D/data $BBH_DIR ${2:-1000} ${3:-34} | tee out/data_log.txt
    mkdir -p out/data; cp $D/data/*.jsonl out/data/; sha256sum $D/data/*.jsonl | tee -a out/data_log.txt ;;
  hc)
    nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader || true
    G=${5:-}; [ "$G" = "-" ] && G=""
    python hostcheck.py out/hc_$2.json --host $(host_of $3) --tools $4 --gate "$G" --data $D/data/dev.jsonl 2>&1 | grep --line-buffered -v Warning ;;
  chk)
    nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader || true
    HFJ=""; [ -f out/hc_$2.json ] && HFJ="--hf_json out/hc_$2.json"
    G=$5; [ "$G" = "-" ] && G=""
    python rl_bbh.py check --host $(host_of $3) --tools $4 --gate "$G" --dev $D/data/dev.jsonl --out out/chk_$2.json $HFJ "${@:6}" 2>&1 | grep --line-buffered -v Warning ;;
  rl)
    nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader || true
    G=$5; [ "$G" = "-" ] && G=""
    python rl_bbh.py rl --runs_dir $D/runs --run $2 --host $(host_of $3) --tools $4 --gate "$G" --seed $6 --train $D/data/train.jsonl --dev $D/data/dev.jsonl "${@:7}" 2>&1 | grep --line-buffered -v Warning ;;
  eval)
    nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader || true
    G=$5; [ "$G" = "-" ] && G=""; WT=$6; [ "$WT" = "-" ] && WT=""
    python rl_bbh.py eval --host $(host_of $3) --tools $4 --gate "$G" --weights "$WT" --out out/eval_$2.json "${@:7}" 2>&1 | grep --line-buffered -v Warning ;;
  ana)
    python analyze.py "${@:2}" ;;
esac
