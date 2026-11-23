#!/bin/bash
# rl28 [28] IFEval on Qwen3-1.7B-Base: SETUP job (one GPU, packed lanes). Run on one GPU (after code/setup.sh):
#   bash uniform/rl28/ifeval/code/eng/job_setup17.sh
# 0. HF cache: Qwen3-1.7B-Base + Qwen3-1.7B (chat; its LM head = the "snapshot" readout, as Qwen3-0.6B's was for 0.6B-Base);
#    tokenizer / chat-template identity vs Qwen3-0.6B.
# 1. lane A: the 14-rule implant's only host-tied rows (force_fix) re-solved on Qwen3-1.7B with the transfer recipe
#    (ifeval/transfer/run_all.sh step 1: the host's own plain HF answers to heldout/data/dev.jsonl, 2 shards, bs 24;
#    solve_forcefix.py) -> vecs_q17.pt; exports sd_snap17 (E,g,F from Qwen3-1.7B) and sd_base17 (E,g,F from 1.7B-Base).
#    lane C (meanwhile): RL memory / speed smoke of a P arm (3 updates, B_max 256, cap 0.46).
# 2. after A: dev choice (hostdev, IFEval-like dev 415, greedy 1024: none / snap17 / base17) || u0 541 evals (rows 1 + 3,
#    + idle identity) with each export.  CHOSEN = the export with the higher dev prompt-strict (the 0.6B rule).
# 3. check.py exactness (HF + reference Executor vs fast engine, 64 dev prompts) || RL smoke of an I arm (memory / speed).
cd $(dirname $0)/..; C=$PWD; source env.sh
R=$DATA/setup; O=$C/../out/setup; mkdir -p $R $O $DATA/eng; exec > >(tee -a $R/job.log) 2>&1
JID=$(date -u +%H%M%S)
CV=${CUDA_VISIBLE_DEVICES:-0}; CV=${CV%%,*}
case $CV in GPU-*) GI=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v u=$CV '$2==u{print $1}');; *) GI=$CV;; esac
nvidia-smi dmon -s um -d 10 -i $GI -o DT > $O/dmon_$JID.txt 2>&1 & DM=$!
say() { echo "$(date -u +%FT%TZ) [setup] $*"; }
say "gpu $CV (index $GI); $DATA free $(df -BG --output=avail $DATA | tail -1)"
B=Qwen/Qwen3-1.7B-Base; I=Qwen/Qwen3-1.7B

HF_HUB_OFFLINE=0 python - > $O/hf.txt 2>&1 <<'EOF'
from huggingface_hub import snapshot_download
for m in ['Qwen/Qwen3-1.7B-Base', 'Qwen/Qwen3-1.7B']:
    try: p = snapshot_download(m, local_files_only=True); src = 'cache'
    except Exception: p = snapshot_download(m, allow_patterns=['*.json', '*.safetensors', '*.txt', 'merges.txt', 'vocab.json']); src = 'fetched'
    print(m, src, p)
from transformers import AutoTokenizer
t0 = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
for m in ['Qwen/Qwen3-1.7B', 'Qwen/Qwen3-1.7B-Base']:
    t = AutoTokenizer.from_pretrained(m); msg = [{'role': 'user', 'content': 'Write two lines, no commas.'}]
    same_t = t.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False) == \
        t0.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    print(m, 'vocab identical to Qwen3-0.6B:', t.get_vocab() == t0.get_vocab(), 'chat template (thinking off) identical:', same_t)
EOF
cat $O/hf.txt | grep -v Warn; prep_pylib > $R/pylib.log 2>&1; (cd heldout; build_tables); cp $DATA/tables.pt $DATA/vocab_strs.json heldout/

# ---------------------------------------------------------------- lane A: force_fix on Qwen3-1.7B -> exports
laneA() {
  cd $C/heldout; mkdir -p $R/runs
  VQ=$DATA/eng/vecs_q17.pt
  if [ ! -e $VQ ]; then
    for K in 0 1; do
      [ -s $R/runs/q17_dev_plain.s$K.jsonl ] || WSM_MODEL=$I python ../transfer/run_t.py data/dev.jsonl $R/runs/q17_dev_plain.s$K.jsonl \
          --mode plain --shard $K/2 --bs 24 > $R/run_t.s$K.log 2>&1 &
    done; wait
    say "q17 dev plain answers: $(cat $R/runs/q17_dev_plain.s?.jsonl | wc -l)"
    WSM_MODEL=$I python ../transfer/solve_forcefix.py $VQ.tmp $R/runs/q17_dev_plain.s0.jsonl $R/runs/q17_dev_plain.s1.jsonl > $R/solve_forcefix.log 2>&1 \
      && mv $VQ.tmp $VQ
    grep -v -i warn $R/solve_forcefix.log | tail -8; cp $R/solve_forcefix.log $O/
  fi
  [ -e $VQ ] || { say "force_fix FAILED"; return 1; }
  sha256sum $VQ | tee $O/vecs_q17.sha256
  for t in snap17:$I base17:$B; do n=${t%%:*}; m=${t#*:}
    [ -e $DATA/eng/sd_$n.pt ] || PYTHONPATH=$C/heldout:$C/f14 python ../hostchk/mk_sd.py $DATA/eng/sd_$n.pt $m $VQ 2>&1 | grep -v -i warn | tail -3
  done
  ls -la $DATA/eng/
}
smoke() {   # smoke NAME ARM [SD]: 3 RL updates at the packed shape, memory + time per update
  local X=""; [ -n "$3" ] && X="--sd $3"; rm -rf $R/smoke_$1
  (cd $C/eng && python rl_if.py --model $B --arm $2 $X --run $R/smoke_$1 --seed 0 --updates 300 --prompts 64 --G 8 --lr 1e-6 --T 1.0 \
     --fmt chat --max_new 1024 --dev_every 25 --resume_every 10 --ckpt_every 300 --B_max 256 --mem_frac 0.46 --workers 6 \
     --stop_after 3 --skip_eval0 1) > $R/smoke_$1.log 2>&1
  local rc=$?; say "smoke $1 rc=$rc"; [ $rc = 0 ] && touch $R/smoke_$1.done; grep -a '^RL' $R/smoke_$1.log | cut -c1-400 | tail -4; cp $R/smoke_$1/steps.jsonl $O/smoke_$1.steps.jsonl 2>/dev/null
}
laneA & PA=$!
grep -q "STOP_AFTER 3" $R/smoke_P.log 2>/dev/null && touch $R/smoke_P.done
PS=""; [ -e $R/smoke_P.done ] || { smoke P P & PS=$!; }
# never a bare `wait` in this shell: it would also wait on the dmon child (j23827 hung here 00:17-00:40Z)
wait $PA $PS
[ -e $DATA/eng/sd_snap17.pt ] && [ -e $DATA/eng/sd_base17.pt ] || { say "exports missing: stop"; kill $DM; exit 2; }

# ---------------------------------------------------------------- 2. dev choice || u0 541 evals
cd $C/eng
rm -f $R/hostdev.jsonl
python hostdev.py --model $B --fmts chat --sds snap17=$DATA/eng/sd_snap17.pt,base17=$DATA/eng/sd_base17.pt --out $R/hostdev.jsonl \
  --mem_frac 0.3 --B_max 256 > $R/hostdev.log 2>&1 & W=$!
for n in snap17 base17; do
  (python eval541.py --model $B --sd $DATA/eng/sd_$n.pt --modes imp,plain --fmt chat --out $R/eval_u0_$n --B_max 128 --mem_frac 0.3 > $R/eval_u0_$n.log 2>&1 &&
   for m in imp plain; do PYTHONPATH=$PYL python $C/score_reduce.py $C/ifdata/ifeval.jsonl $R/eval_u0_$n.$m.jsonl $O/eval_u0_$n.$m.score.json \
       --command "rl28/ifeval u0 1.7B-Base sd_$n $m" > /dev/null; done && cp $R/eval_u0_$n.summary.json $O/; say "u0 eval $n done") & W="$W $!"
done
wait $W
cp $R/hostdev.jsonl $O/; cat $R/hostdev.jsonl
CH=$(python - $R/hostdev.jsonl <<'EOF'
import json, sys
r = {d['implant']: d['prompt_strict'] for d in map(json.loads, open(sys.argv[1]))}
print(max(['snap17', 'base17'], key=lambda k: r.get(k, -1)))
EOF
)
[ -n "$CH" ] && echo $CH > $DATA/eng/CHOSEN.tmp && mv $DATA/eng/CHOSEN.tmp $DATA/eng/CHOSEN && cp $DATA/eng/CHOSEN $O/
say "CHOSEN export: $(cat $DATA/eng/CHOSEN)"

# ---------------------------------------------------------------- 3. exactness || I smoke
SD=$DATA/eng/sd_$(cat $DATA/eng/CHOSEN).pt
(python check.py --model $B --sd $SD --data $C/gen/dev.jsonl --n 64 --max_new 1024 --out $R/chk --fmt chat --mem_frac 0.45 > $R/chk.log 2>&1;
 say "check rc=$?"; cp $R/chk/check.jsonl $O/check.jsonl 2>/dev/null) & W=$!
smoke I I $SD & W="$W $!"
wait $W
rm -rf $R/smoke_P $R/smoke_I
kill $DM 2>/dev/null
awk '$1 !~ /^#/ {s+=$4; n++} END {if (n) printf "mean SM%% %.1f over %d samples\n", s/n, n}' $O/dmon_$JID.txt | tee -a $O/setup.txt
say "done; $DATA free $(df -BG --output=avail $DATA | tail -1)"
tail -n 60 $R/job.log > $O/log_tail.txt
