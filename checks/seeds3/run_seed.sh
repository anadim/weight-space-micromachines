#!/bin/bash
# usage: CUDA_VISIBLE_DEVICES=0 bash checks/seeds3/run_seed.sh NAME
#   NAME = m4_sK | m2_sK | LIB_sK | LIBscr_sK | m1m_sK | aNL_sK | m3_sK | W3d_sK
# One seeds3 RL run on one GPU (blog §5.2 three-seed checks; README.md):
#   m4_sK      (b) tool library + IFEval circuit on the mixed task (GSM8K / BBH / IFEval), from the noisy-routing SFT host
#              (sft_warm.py --noisy 0.3), 150 updates; test 400 + per-tool leave-one-out at u150
#   m2_sK      (c) the same from the clean SFT host with every implant but the calculator scrambled (--scramble 1), 150 u
#   LIB_sK     (d) tool library on GSM8K, 300 u, dev 800 every 25 (+u64), GSM8K test + GSM-Hard at u300
#   LIBscr_sK  (d) LIB with every implant but the calculator scrambled
#   m1m_sK     (e) all implants on the mixed task (m1 with --cw_mask 1), from the clean SFT host, 150 u; test 400 + LOO
#   aNL_sK     (e) plain RL (no implants) from the worked-solution SFT host (sft_warm.py on union_nl.jsonl), 150 u
#   m3_sK      (e) plain RL from the tagged SFT host (m3), 150 u
#   W3d_sK     (f) wiring arm W3d: links initialised from the SFT-distilled W3e links, 300 u, dev 800 every 25; then a
#              separate test call (GSM8K test + GSM-Hard, links on / host only) into $WSM_DATA/runs/TEST_W3d_sK
# All 64 prompts x G8, lr 1e-6, Dr.GRPO, no KL, T 1.0. The original runs were cluster-queue jobs with a 3.5 h (m4 / m2) or
# 4 h (LIB / LIBscr) limit that resumed from the rolling resume state (every 10 updates); here one call runs to the end (up
# to 2 tries, B_max halves on OOM; a rerun of the same NAME resumes). Wall time on one B200: m4 / m2 / m1m about 2 h 50 min,
# LIB / LIBscr about 4 h 15 min, aNL / m3 about 21 min (+ a few min for the SFT host on first use), W3d about 1 h incl. test.
# Needs: the seeds3 code tree (built on first use with rebuild_code.sh into $S3_CODE, default checks/seeds3/code), its GSM8K
# and mixed-task data (prepared here), the calculator veto $WSM_VETO (python scripts/solve_veto.py --device cuda, from the
# release root), a Python env with torch / transformers (REPRODUCE §2), network for the first Qwen/Qwen3-0.6B download.
# Outputs: $WSM_DATA/runs/NAME/{steps.jsonl,test_eval.json,config.json,...} (WSM_DATA default <release>/runs/seeds3);
# reduce them into res/ with: python3 checks/seeds3/reduce_run.py $WSM_DATA/runs/NAME NAME
NAME=$1
[ -n "$NAME" ] || { echo "usage: $0 m4_sK|m2_sK|LIB_sK|LIBscr_sK|m1m_sK|aNL_sK|m3_sK|W3d_sK"; exit 64; }
HERE=$(cd "$(dirname "$0")" && pwd); REL=$(cd $HERE/../.. && pwd)
B=${WSM_DATA:-$REL/runs/seeds3}; RUN=$B/runs/$NAME; mkdir -p $B/runs $B/hosts $B/logs $B/resume $RUN
exec 9> $B/.prep.lock; flock 9      # runs started in parallel build the shared code tree / data / hosts / libs one at a time
CODE=${S3_CODE:-$HERE/code}
[ -d $CODE ] || bash $HERE/rebuild_code.sh $CODE || exit 2
LOGF=$B/logs/$NAME.out
say() { echo "$(date -u +%FT%TZ) [$NAME] $*" | tee -a $LOGF; }
source $CODE/rl/gsm_grpo/env_local.sh
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FLEX_SMALL=${FLEX_SMALL:-1}
export WSM_ROOT=$CODE WSM_REPO_ROOT=$CODE WSM_VETO=${WSM_VETO:-$REL/runs/veto_qwen3.pt}
export WSM_INTEG_DIR=$CODE/bbh_generic/integrate/code WSM_BBH_GENERIC=$CODE/bbh_generic RLENVS=$CODE/rl/library/mixed \
  RLW_DIR=$CODE/rl/wiring IFDIR=$CODE/ifeval/heldout IF_HOST=Qwen/Qwen3-0.6B WSM_RESUME_DIR=$B/resume
[ -f $WSM_VETO ] || { say "no veto at $WSM_VETO: run python scripts/solve_veto.py --device cuda in the release root"; exit 2; }
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader >> $LOGF

# ---- prerequisites (idempotent)
python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-0.6B')" >> $LOGF 2>&1 || { say "Qwen3-0.6B download failed"; exit 2; }
[ -f $CODE/data/gsm8k/shots_v1.json ] || (cd $CODE/data && python prepare_gsm.py) >> $LOGF 2>&1 || { say "prepare_gsm failed"; exit 2; }
(cd $CODE/rl/library/mixed && { grep -E ' (train|test)\.jsonl$' DATA_SHA256.txt | sha256sum -c --quiet 2>/dev/null || \
   { python mixed_data.py && grep -E ' (train|test)\.jsonl$' DATA_SHA256.txt | sha256sum -c --quiet; }; }) >> $LOGF 2>&1 \
  || { say "mixed-task data does not match DATA_SHA256.txt"; exit 3; }
prep_host() {     # prep_host sft_host|sft_host_noisy|sft_host_nl: the SFT warm-start host of the mixed task (one GPU, minutes)
  local hn=$1
  [ -f $B/hosts/$hn/sft_config.json ] && return 0
  say "building $hn"; rm -rf $B/hosts/$hn.tmp
  local UN=""; [ $hn = sft_host_noisy ] && UN="--noisy 0.3"
  if [ $hn = sft_host_nl ]; then   # the 432 tagged union rows with every tool call rewritten as a worked solution (CPU, seconds)
    local NLU=$CODE/rl/library/mixed/union_nl.jsonl
    echo "bb894058308fed93f45f5ba1655d99eba3ebea7295a21357dd45c261d512baa6  $NLU" | sha256sum -c --quiet 2>/dev/null || \
      { (cd $CODE/rl/library/mixed && python make_nl_sft.py $CODE/bbh_generic/integrate/data/train_union_v2.jsonl $NLU) >> $B/logs/$hn.out 2>&1
        echo "bb894058308fed93f45f5ba1655d99eba3ebea7295a21357dd45c261d512baa6  $NLU" | sha256sum -c --quiet; } || { say "union_nl.jsonl build / SHA failed"; exit 3; }
    UN="--union $NLU"
  fi
  (cd $CODE/rl/library/mixed && python sft_warm.py --out $B/hosts/$hn.tmp $UN) >> $B/logs/$hn.out 2>&1 || { say "SFT $hn failed"; exit 3; }
  mv $B/hosts/$hn.tmp $B/hosts/$hn
}
prep_pylib() {    # IFEval checkers for the mixed task: lm_eval 0.4.13 + langdetect 1.0.9, --no-deps into $B/pylib; NLTK punkt
  export PYTHONPATH=$B/pylib${PYTHONPATH:+:$PYTHONPATH} NLTK_DATA=$B/nltk_data
  python -c "from lm_eval.tasks.ifeval import utils; import langdetect" 2>/dev/null || {
    python -c "import nltk" 2>/dev/null || pip install -q --no-deps --target $B/pylib nltk regex click joblib tqdm
    pip install -q --no-deps --target $B/pylib lm_eval==0.4.13 langdetect==1.0.9 immutabledict; } >> $B/logs/pylib.out 2>&1
  # NLTK >= 3.9.3 silently skips downloads into group/world-writable dirs: make it private and fail loudly
  mkdir -p $B/nltk_data; chmod go-w $B/nltk_data
  python -c "import nltk; [nltk.download(p, download_dir='$B/nltk_data', quiet=True, raise_on_error=True) for p in ('punkt', 'punkt_tab')]; nltk.data.find('tokenizers/punkt_tab/english/')" >> $B/logs/pylib.out 2>&1 \
    && python -c "from lm_eval.tasks.ifeval import utils; import langdetect" || { say "IFEval checker libs / NLTK punkt_tab missing (see $B/logs/pylib.out)"; exit 3; }
}

MIX="--arm LIB --task mixed --scramble 0 --updates 150 --prompts 64 --G 8 --T 1.0 --lr 1e-6 --kl 0 --loss drgrpo --B_max 128 \
 --tok_budget 4096 --max_new 512 --S_max 896 --dev_mixed_n 400 --dev_every 50 --dev_extra 25 --iw_eps 0.05 --link_lr 1e-3 \
 --link_eps 1e-8 --ckpt_every 150 --test_n 400 --resume_every 10"
LIBG="--arm LIB --updates 300 --prompts 64 --G 8 --T 1.0 --lr 1e-6 --kl 0 --loss drgrpo --dev_every 25 --dev_extra 64 \
 --dev_n 800 --B_max 128 --tok_budget 4096 --ckpt_every 300 --iw_eps 0.05 --link_lr 1e-3 --link_eps 1e-8 --resume_every 10"
S=${NAME##*_s}
case $NAME in
  m4_s*)     prep_host sft_host_noisy; prep_pylib; ARGS="$MIX --host $B/hosts/sft_host_noisy --plain 0 --seed $S";;
  m2_s*)     prep_host sft_host;       prep_pylib; ARGS="$MIX --host $B/hosts/sft_host --plain 0 --scramble 1 --seed $S";;
  LIB_s*)    ARGS="$LIBG --scramble 0 --seed $S";;
  LIBscr_s*) ARGS="$LIBG --scramble 1 --seed $S";;
  m1m_s*)    prep_host sft_host;       prep_pylib; ARGS="$MIX --host $B/hosts/sft_host --plain 0 --cw_mask 1 --seed $S";;
  aNL_s*)    prep_host sft_host_nl;    prep_pylib; ARGS="$MIX --host $B/hosts/sft_host_nl --plain 1 --seed $S";;
  m3_s*)     prep_host sft_host;       prep_pylib; ARGS="$MIX --host $B/hosts/sft_host --plain 1 --seed $S";;
  W3d_s*)    KIND=w; ARGS="--arm W3d --init_links runs/sft3_W3e_lr1e-2/links_final.pt --updates 300 --dev_every 25 --dev_extra 64 \
 --dev_n 800 --B_max 128 --tok_budget 4096 --ckpt_every 300 --eval_all 1 --resume_every 25 --seed $S";;
  *) say "unknown run $NAME"; exit 64;;
esac
exec 9>&-                           # prerequisites done: release the lock
TEST=$B/runs/TEST_$NAME
done_ok() { if [ "$KIND" = w ]; then [ -f $TEST/test_eval.json ]; else grep -q DONE_RUN $RUN/steps.jsonl 2>/dev/null; fi; }
BMAX=${BMAX:-}
for try in 1 2; do
  BM=""; [ -n "$BMAX" ] && BM="--B_max $BMAX"
  say "start try$try: $ARGS $BM"; T0=$(date +%s)
  if [ "$KIND" = w ]; then
    (cd $CODE/rl/wiring && python train_wiring.py $ARGS $BM --run_dir $RUN) >> $LOGF 2>&1; rc=$?
    LAST=$(ls -d $RUN/step_* 2>/dev/null | tail -1)
    if [ $rc = 0 ] && grep -q DONE_RUN $RUN/steps.jsonl && [ -n "$LAST" ] && [ ! -f $TEST/test_eval.json ]; then
      say "test eval $LAST"; mkdir -p $TEST
      (cd $CODE/rl/wiring && python train_wiring.py --arm W3e --eval_all 0 --B_max 128 --tok_budget 4096 --test_sets test,hard \
         --eval_ckpt $LAST --run_dir $TEST) >> $LOGF 2>&1; rc=$?
    fi
  else
    (cd $CODE/rl/library && python train_library.py $ARGS $BM --run_dir $RUN) >> $LOGF 2>&1; rc=$?
  fi
  say "exit rc=$rc after $(( $(date +%s) - T0 )) s"
  if done_ok; then say DONE; exit 0; fi
  if tail -n 200 $LOGF | grep -q "CUDA out of memory"; then BMAX=$(( ${BMAX:-128} / 2 )); say "OOM -> B_max $BMAX"; fi
  sleep 5
done
exit 3
