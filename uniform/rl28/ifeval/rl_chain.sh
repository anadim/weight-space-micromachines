#!/bin/bash
# rl_chain.sh NAME [--inner] -- IFEval RL arms of task [28] (Qwen3-1.7B-Base) on the fast stack (copy of uniform/rl24/ifeval/rl_chain.sh).
#   NAME = {P,I}_s{0,1,2}[_TAG]  (optional suffix for variants, e.g. P_s0_lr2e-6 with RL_EXTRA="--lr 2e-6"; P: plain RL, host alone = table row 2; I: RL with the 14-rule implant attached = row 4)
#   EVAL_EXTRA: extra eval541.py args (packed jobs: --B_max 128 --mem_frac F); RL_EXTRA: extra rl_if.py args
#   HOST: env MODEL (default Qwen/Qwen3-1.7B-Base; HOST_MODEL = MODEL) + TAG (implant export $DATA/eng/sd_TAG.pt made by the
#   setup job; default = the dev-chosen variant in $DATA/eng/CHOSEN)
# No --inner: CHAIN DRIVER: runs the job on this machine's GPU (CUDA_VISIBLE_DEVICES; 4 h limit per start), restarts after a timeout /
#   crash until out/NAME/DONE exists. Every start resumes from $DATA/rl/NAME/resume.pt (saved every RESUME_EVERY=10 updates, non-blocking).
# Inside the job: prep (flock'd), rl_if.py (300 u), then the 541 eval of step_000300 (implant on and off + idle identity),
#   seeded lm_eval scoring; results -> out/NAME/.
NAME=$1; MODE=$2
[ -n "$NAME" ] || { echo "usage: $0 NAME"; exit 64; }
HERE=$(cd "$(dirname "$0")" && pwd)
if [ "$MODE" != "--inner" ]; then
  # chain driver: run the job (--inner) in this folder on one GPU, restart after a crash / time limit until out/NAME/DONE exists.
  # Original runs: one cluster-queue job per start (4 h limit), on a B200.
  mkdir -p $HERE/out; CL=$HERE/out/$NAME.chain.log
  cd $HERE
  for chain in $(seq 1 ${MAXCHAIN:-12}); do
    mkdir -p envs; : > envs/$NAME.env
    for v in MODEL TAG RL_EXTRA EVAL_EXTRA FMT MAXNEW CKPT_EVERY RESUME_EVERY; do [ -n "${!v}" ] && echo "export $v=\"${!v}\"" >> envs/$NAME.env; done
    echo "$(date -u +%FT%TZ) chain $chain start ($NAME)" >> $CL
    timeout ${TMO:-14400} bash rl_chain.sh $NAME --inner; rc=$?
    echo "$(date -u +%FT%TZ) chain $chain ended rc=$rc" >> $CL
    if [ -f $HERE/out/$NAME/DONE ]; then echo "$(date -u +%FT%TZ) $NAME RLDONE" >> $CL; exit 0; fi
    case $rc in 64) echo "  bad name: stop" >> $CL; exit 64;; 3) echo "  2 in-job failures: stop (out/$NAME/log_tail.txt)" >> $CL; exit 3;; esac
    [ -n "$NOCHAIN" ] && exit $rc
    sleep 30
  done
  exit 1
fi

# ------------------------------------------------------------------------------- inside the job (cwd = this folder)
H=$PWD; C=$H/code
[ -f $H/envs/$NAME.env ] && source $H/envs/$NAME.env
source $C/env.sh
MODEL=${MODEL:-Qwen/Qwen3-1.7B-Base}; export HOST_MODEL=$MODEL; TAG=${TAG:-$(cat $DATA/eng/CHOSEN 2>/dev/null)}; FMT=${FMT:-chat}
ARM=${NAME:0:1}; S=$(echo $NAME | sed -E 's/^[PI]_s([0-9]+).*/\1/')
case $ARM in P|I) ;; *) echo "bad arm $ARM"; exit 64;; esac
case $S in [0-9]*) ;; *) echo "bad seed in $NAME"; exit 64;; esac
R=$DATA/rl/$NAME; O=$H/out/$NAME; mkdir -p $R $O; LOGF=$R/job.log
exec > >(tee -a $LOGF) 2>&1
say() { echo "$(date -u +%FT%TZ) [$NAME] $*"; }
nvidia-smi -i ${CUDA_VISIBLE_DEVICES:-0} --query-gpu=index,name,memory.used --format=csv,noheader
prep_pylib > $DATA/pylib.log 2>&1
(cd $C/heldout; build_tables)
cp $DATA/tables.pt $DATA/vocab_strs.json $C/heldout/ 2>/dev/null
SD=$DATA/eng/sd_$TAG.pt   # P arms need it only for the u300 eval (row 2b); I arms from u0
[ $ARM = P ] || { [ -n "$TAG" ] && [ -e $SD ]; } || { echo "implant export missing (TAG=$TAG; run the setup job first)"; exit 64; }
X=""; [ $ARM = I ] && X="--sd $SD"
report() { cp $R/steps.jsonl $O/ 2>/dev/null; tail -n 80 $LOGF > $O/log_tail.txt; }
for try in 1 2; do
  if ! grep -q DONE_RUN $R/steps.jsonl 2>/dev/null; then
    say "train try$try: arm $ARM seed $S model $MODEL tag $TAG fmt $FMT $RL_EXTRA"
    (cd $C/eng && python rl_if.py --model $MODEL --arm $ARM $X --run $R --seed $S --updates 300 --prompts 64 --G 8 --lr 1e-6 \
        --T 1.0 --fmt $FMT --max_new ${MAXNEW:-1024} --dev_every 25 --resume_every ${RESUME_EVERY:-10} --ckpt_every ${CKPT_EVERY:-300} $RL_EXTRA); rc=$?
    say "train rc=$rc"; report
  fi
  if grep -q DONE_RUN $R/steps.jsonl 2>/dev/null; then
    CK=$R/step_000300/model.safetensors
    TAG=${TAG:-$(cat $DATA/eng/CHOSEN 2>/dev/null)}; SD=$DATA/eng/sd_$TAG.pt
    [ -n "$TAG" ] && [ -e $SD ] || { say "eval: implant export missing (TAG=$TAG)"; report; exit 3; }
    if [ ! -f $O/eval_u300.summary.json ]; then
      say "eval541 $CK"
      (cd $C/eng && python eval541.py --model $MODEL --ckpt $CK --sd $SD --modes imp,plain --fmt $FMT --out $R/eval_u300 $EVAL_EXTRA) && \
      for m in imp plain; do
        PYTHONPATH=$PYL python $C/score_reduce.py $C/ifdata/ifeval.jsonl $R/eval_u300.$m.jsonl $O/eval_u300.$m.score.json \
          --command "rl28/ifeval $NAME u300 $m sd_$TAG" > /dev/null || exit 3
      done && cp $R/eval_u300.summary.json $O/
    fi
    [ -f $O/eval_u300.summary.json ] && { report; say DONE; touch $O/DONE; exit 0; }
  fi
  sleep 5
done
report; exit 3
