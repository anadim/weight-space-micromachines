#!/bin/bash
# usage: CUDA_VISIBLE_DEVICES=0 bash checks/seeds35/ifeval_seed.sh ARM SEED [OUT]      (ARM = P | I, SEED = 3 | 4 | 5)
# One IFEval RL seed of the uniform rows (blog §5.1) on Qwen3-0.6B-Base, as the seeds 3-5 ran: P = host + RL, no implant
# (row 2; its implant-attached eval = row 2b), I = host + RL with the 14-rule implant attached (row 4). Same recipe as the
# seeds 0-2 driver uniform/rl24/ifeval/rl_chain.sh (GRPO / Dr.GRPO 300 u, 64 prompts x G8, lr 1e-6, T 1, max_new 1024, chat
# format with thinking off), then the IFEval-541 eval of step_000300 (greedy, max_new 2048, implant on and off + idle
# identity, seeded lm_eval scorer). Code: uniform/rl24/ifeval/code, used unchanged.
# Seeds 3-5 ran on A100-SXM4-40GB GPUs with this script: B_max 128 / tok_budget 8192 for RL (auto-retry from the last
# resume state with 96 / 4096 after a crash or OOM), eval B_max 96, dev every 100 updates. The B200 seeds 0-2 used
# B_max 512|256 / tok_budget 32768, eval 256|128 and dev every 25: batch scheduling and monitoring only, same recipe.
# Outputs: OUT (default checks/seeds35/res_new/<ARM>_s<SEED>): eval_u300.{imp,plain}.score.json, eval_u300.summary.json,
# steps.jsonl, gpu.txt and start/train_start/train_end/end.txt. Scratch (HF cache, code copy, 7 GB resume state) in OUT/.work.
# Runtime on one A100 40 GB: I about 3 h 15 min train + 8 min eval, P about 1 h 30 min + 4 min (peak memory 33.6 / 32.2 GB).
ARM=$1; S=$2; [ "$ARM" = P -o "$ARM" = I ] && [ -n "$S" ] || { echo "usage: $0 P|I SEED [OUT]"; exit 64; }
HERE=$(cd "$(dirname "$0")" && pwd); WSM_ROOT=${WSM_ROOT:-$(cd $HERE/../.. && pwd)}
NAME=${ARM}_s$S; O=${3:-$HERE/res_new/$NAME}; mkdir -p $O; O=$(cd $O && pwd)
W=$O/.work; R=$W/run; C=$W/code; mkdir -p $R
exec > >(tee -a $O/run.log) 2>&1
say() { echo "$(date -u +%FT%TZ) [$NAME] $*"; }
fail() { say "FAIL $1 (setup)"; exit $1; }
say start; date -u +%FT%TZ > $O/start.txt
nvidia-smi --query-gpu=name,memory.total --format=csv > $O/gpu.txt 2>&1; cat $O/gpu.txt
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=${HF_HOME:-$W/hf} TRITON_CACHE_DIR=$W/triton TORCHINDUCTOR_CACHE_DIR=$W/inductor
export PYL=$W/pylib NLTK_DATA=$W/nltk TFX=$W/tfx
# the shipped code dir holds only the files not elsewhere in the release: setup.sh (CPU, ~2 min, idempotent) completes it in
# place (shared modules, IFEval-541 prompts via ifeval/data/prepare_ifeval.py, which needs `datasets`; RL train / dev data, sha-checked)
bash $WSM_ROOT/uniform/rl24/ifeval/code/setup.sh > $O/setup.log 2>&1 || { tail -20 $O/setup.log; fail 4; }
rm -rf $C; cp -r $WSM_ROOT/uniform/rl24/ifeval/code $C

# ---- python deps (inside OUT/.work): transformers pinned to 4.56.2 (the implant export loads Qwen3-0.6B with dtype=bf16,
#      a kwarg older transformers ignore); IFEval checkers = lm_eval 0.4.13 without deps.
if ! python3 -c "import transformers as t, sys; sys.exit(t.__version__ != '4.56.2')" 2>/dev/null; then
  python3 -m pip install -q --target $TFX --no-deps transformers==4.56.2 'tokenizers>=0.22,<=0.23' 'huggingface_hub>=0.34,<1.0' 'safetensors>=0.4.3'
  export PYTHONPATH=$TFX${PYTHONPATH:+:$PYTHONPATH}
  python3 -c "import transformers, tokenizers, huggingface_hub, safetensors, regex, yaml, requests, tqdm, filelock, packaging" 2>/dev/null || \
    { rm -rf $TFX; python3 -m pip install -q --target $TFX transformers==4.56.2; }
fi
python3 -c "import transformers, torch, triton; print('transformers', transformers.__version__, 'torch', torch.__version__, 'triton', triton.__version__)" || fail 5
python3 -m pip install -q --target $PYL --no-deps lm_eval==0.4.13 langdetect==1.0.9 && python3 -m pip install -q --target $PYL nltk immutabledict absl-py six || fail 6
# NLTK >= 3.9.3 silently refuses to download into a group/world-writable dir (quiet=True hid it in the first seeds 3-5
# start): without punkt_tab, count_sentences raises and every number_sentences prompt scores 0. Make the dir private and assert.
mkdir -p $NLTK_DATA; chmod -R go-w $W
PYTHONPATH=$PYL:$PYTHONPATH python3 -c "import nltk; [nltk.download(p, download_dir='$NLTK_DATA', quiet=True, raise_on_error=True) for p in ('punkt_tab', 'punkt')]" || fail 6
(cd $C/eng && PYTHONPATH=$PYL:$PYTHONPATH python3 -c "
import nltk, importlib, ifreward; nltk.data.find('tokenizers/punkt_tab/english/'); ifreward._load()
n = importlib.import_module('lm_eval.tasks.ifeval.instructions_util').count_sentences('A b. C d.'); assert n == 2, n
doc = dict(key=1, prompt='x', instruction_id_list=['length_constraints:number_sentences'], kwargs=[dict(num_sentences=2, relation='at least')])
r = ifreward.score(doc, 'A b. C d. E f.'); assert r[0] == 1, r; print('nltk punkt_tab ok: count_sentences + number_sentences reward work')") \
  || { say 'NLTK punkt_tab / number_sentences check FAILED'; fail 6; }
python3 -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-0.6B-Base'); s('Qwen/Qwen3-0.6B')" > $O/hf.log 2>&1 || { cat $O/hf.log; fail 7; }

# ---- implant export (sd_inst = 14-rule full14 export with the frozen Qwen3-0.6B LM-head snapshot; the seeds 0-2 sd_inst.pt)
SD=$W/sd_inst.pt
(cd $C/heldout && PYTHONPATH=.:$PYTHONPATH python3 -c "
from transformers import AutoTokenizer
from common import vocab_strings
from tables import build
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B'); build(tok, vocab_strings(tok), device='cpu'); print('tables built')" \
  && PYTHONPATH=$C/heldout:$C/f14:$PYTHONPATH python3 ../hostchk/mk_sd.py $SD Qwen/Qwen3-0.6B) > $O/sd.log 2>&1 || { tail -30 $O/sd.log; fail 8; }
X=""; [ $ARM = I ] && X="--sd $SD"

# ---- RL (resumes from $R/resume.pt after a crash; up to 3 tries, smaller batch / plain LP head on later tries)
date -u +%FT%TZ > $O/train_start.txt
for try in 1 2 3; do
  grep -q DONE_RUN $R/steps.jsonl 2>/dev/null && break
  BM=${BMAX:-128}; TB=${TOKB:-8192}; [ $try -ge 2 ] && { BM=96; TB=4096; }
  [ $try -ge 2 ] && cat $O/train_try*.log | grep -qi "nvrtc\|cuModule\|lmhead\|cuda_runtime" && export FT_LP=0
  [ $try -ge 2 ] && cat $O/train_try*.log | grep -qi "OutOfResources\|out of resource" && export FT_ATTN=flex
  say "train try$try arm $ARM seed $S B_max $BM tok_budget $TB FT_LP=${FT_LP:-1} FT_ATTN=${FT_ATTN:-triton}"
  (cd $C/eng && python3 rl_if.py --model Qwen/Qwen3-0.6B-Base --arm $ARM $X --run $R --seed $S --updates 300 --prompts 64 --G 8 \
      --lr 1e-6 --T 1.0 --fmt chat --max_new 1024 --dev_every 100 --resume_every 25 --ckpt_every 300 \
      --B_max $BM --tok_budget $TB --workers 8) > $O/train_try$try.log 2>&1
  say "train try$try rc=$?"; tail -5 $O/train_try$try.log
done
date -u +%FT%TZ > $O/train_end.txt; cp $R/steps.jsonl $O/steps.jsonl 2>/dev/null
grep -q DONE_RUN $R/steps.jsonl 2>/dev/null || { say "TRAIN FAILED"; exit 3; }

# ---- IFEval-541 eval of step_000300: imp (implant attached: row 4 for I, row 2b for P) and plain (row 2 for P), idle identity
CK=$R/step_000300/model.safetensors; EV=$O/eval; mkdir -p $EV; EVRC=1
for try in 1 2; do
  BM=96; [ $try = 2 ] && BM=64
  say "eval541 try$try B_max $BM"
  (cd $C/eng && python3 eval541.py --model Qwen/Qwen3-0.6B-Base --ckpt $CK --sd $SD --modes imp,plain --fmt chat --out $EV/eval_u300 \
      --B_max $BM) > $O/eval_try$try.log 2>&1 && { EVRC=0; break; }
  tail -5 $O/eval_try$try.log
done
if [ $EVRC = 0 ]; then
  for m in imp plain; do
    PYTHONPATH=$PYL:$PYTHONPATH python3 $C/score_reduce.py $C/ifdata/ifeval.jsonl $EV/eval_u300.$m.jsonl $O/eval_u300.$m.score.json \
      --command "seeds35 ifeval $NAME u300 $m" > $O/score_$m.log 2>&1 || EVRC=4
  done
  cp $EV/eval_u300.summary.json $O/ 2>/dev/null
fi
date -u +%FT%TZ > $O/end.txt
[ $EVRC = 0 ] && touch $O/DONE
say "end rc=$EVRC (scratch $W kept: delete it when done)"; exit $EVRC
