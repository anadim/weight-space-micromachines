#!/bin/bash
# rl28/bbh work jobs ([28] BBH on Qwen3-1.7B-Base; REPRODUCE §18.3), one GPU each. usage (from uniform/rl28/bbh/w):
#   bash job.sh ft                    : 1.7B call-FT host -> $D/ft17
#   bash job.sh idle NAME HOST GATE [args] : idle bit-identity (implants attached vs not, greedy dev) -> out/idle_NAME.json
#   bash job.sh hc NAME HOST TOOLS    : host check on dev (hostcheck.py) -> out/hc_NAME.json
#   bash job.sh chk NAME HOST TOOLS GATE [args]  : engine exactness/speed checks (rl_bbh.py check) -> out/chk_NAME.json
#   bash job.sh rl RUN HOST TOOLS GATE SEED [args] : GRPO run (resumable; re-run to continue) -> $D/runs/RUN, out/RUN
#   bash job.sh eval NAME HOST TOOLS GATE WEIGHTS [args] : BBH-27 (WEIGHTS '-' = the host itself) -> out/eval_NAME.json
#   bash job.sh ana OUT.md --row LABEL=a.json,.. --cmp X-Y : table + paired bootstrap CIs (analyze.py; CPU)
# The RL train / dev data are the 0.6B ones ($RL24_DATA/bbh/data; build them with uniform/rl24/bbh/w/job.sh data).
set -eo pipefail
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$H"
source "$H/../../../env.sh"                     # uniform/env.sh: WSM_ROOT, RL24_DATA, RL28_DATA, BBH_DIR, BBH_COT
D=$RL28_DATA/bbh; DD=$RL24_DATA/bbh/data; mkdir -p $D/tmp out; export TMPDIR=$D/tmp
export HOST_SIZE=${HOST_SIZE:-1.7B}     # fastqwen shapes; all hosts here are 1.7B
export WSM_REPO=$WSM_ROOT WSM_REPO_ROOT=$WSM_ROOT VETO=$WSM_ROOT/results/gsm_bigger/veto_q17.pt FT_HOST=$D/ft17
# files shared unchanged with gsm_stack/engine and uniform/rl24/bbh/w (not duplicated here); fastqwen.py (HOST_SIZE) ships here
for f in wiring/hostkern.py stk/tattn.py stk/tk.py stk/tk_compile.py stk/lmhead.cu; do
  [ -f "$(basename $f)" ] || cp "$WSM_ROOT/gsm_stack/engine/$f" .; done
for f in bbhdata.py bbheng.py devcurves.py envs_ref.py fasttrain.py hostcheck.py; do
  [ -f $f ] || cp "$H/../../../rl24/bbh/w/$f" .; done
host_of() { case $1 in base17) echo Qwen/Qwen3-1.7B-Base;; ft17) echo $D/ft17;; *) echo $1;; esac; }
case $1 in
  data)
    echo 'reuse the 0.6B data: '$DD; sha256sum $DD/*.jsonl ;;
  ft)   # [28] call-FT of 1.7B-Base with the ft_int_v2 recipe (teamA/ft.py, train_union_v2, lr 2e-5, 6 ep, tok_budget 8000, seed 0)
    nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader || true
    [ -f $D/ft17/model.safetensors ] || [ -f $D/ft17/model.safetensors.index.json ] || { cd $WSM_ROOT/bbh_generic/teamA; T0=$(date +%s)
      python ft.py --model Qwen/Qwen3-1.7B-Base --data ../integrate/data/train_union_v2.jsonl --out $D/ft17 --lr 2e-5 --epochs 6 --tok_budget 8000 --seed 0 2>&1 | grep --line-buffered -v Warning
      echo "FT_SECONDS $(( $(date +%s) - T0 ))"; cd $H; }
    ls -la $D/ft17; sha256sum $D/ft17/*.safetensors | tee out/ft17_sha.txt ;;
  idle)
    G=$4; [ "$G" = "-" ] && G=""
    python rl_bbh.py idle --host $(host_of $3) --tools 1 --gate "$G" --dev $DD/dev.jsonl --out out/idle_$2.json "${@:5}" 2>&1 | grep --line-buffered -v Warning ;;
  hc)
    nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader || true
    G=${5:-}; [ "$G" = "-" ] && G=""
    python hostcheck.py out/hc_$2.json --host $(host_of $3) --tools $4 --gate "$G" --data $DD/dev.jsonl 2>&1 | grep --line-buffered -v Warning ;;
  chk)
    nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader || true
    HFJ=""; [ -f out/hc_$2.json ] && HFJ="--hf_json out/hc_$2.json"
    G=$5; [ "$G" = "-" ] && G=""
    python rl_bbh.py check --host $(host_of $3) --tools $4 --gate "$G" --dev $DD/dev.jsonl --out out/chk_$2.json $HFJ "${@:6}" 2>&1 | grep --line-buffered -v Warning ;;
  rl)
    [ -f $D/runs/$2/DONE ] && { echo "$2 already DONE ($D/runs/$2/DONE): nothing to do"; exit 0; }
    nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader || true
    G=$5; [ "$G" = "-" ] && G=""
    python rl_bbh.py rl --runs_dir $D/runs --run $2 --host $(host_of $3) --tools $4 --gate "$G" --seed $6 --train $DD/train.jsonl --dev $DD/dev.jsonl "${@:7}" 2>&1 | grep --line-buffered -v Warning ;;
  eval)
    nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader || true
    G=$5; [ "$G" = "-" ] && G=""; WT=$6; [ "$WT" = "-" ] && WT=""
    python rl_bbh.py eval --host $(host_of $3) --tools $4 --gate "$G" --weights "$WT" --out out/eval_$2.json "${@:7}" 2>&1 | grep --line-buffered -v Warning ;;
  ana)
    python analyze.py "${@:2}" ;;
esac
