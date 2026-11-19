#!/bin/bash
# IFEval: the frozen 14-rule implant on Qwen3-0.6B / 1.7B / 4B (chat models, thinking off) + the add-one-rule curve.
# Run from the repo root. One GPU (an A100 80GB was used). Only the force_fix rows are re-solved per host.
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd); PY=${PY:-python}; T=../transfer
cd "$ROOT/ifeval/heldout"; mkdir -p runs jobs
export PYTHONPATH="$ROOT/ifeval/heldout:$PYTHONPATH"
[ -e data/ifeval.jsonl ] || ln -s ../../data/ifeval.jsonl data/ifeval.jsonl
declare -A HOST=([q06]=Qwen/Qwen3-0.6B [q17]=Qwen/Qwen3-1.7B [q4]=Qwen/Qwen3-4B)
declare -A VEC=([q06]=vecs2_forcefix.pt [q17]=vecs_q17.pt [q4]=vecs_q4.pt)

# 1. force_fix rows for the bigger hosts, from each host's OWN plain answers to our dev prompts (no IFEval prompt used).
for t in q17 q4; do
  for K in 0 1; do WSM_MODEL=${HOST[$t]} $PY $T/run_t.py data/dev.jsonl runs/${t}_dev_plain.s$K.jsonl --mode plain --shard $K/2 --bs 24; done
  WSM_MODEL=${HOST[$t]} $PY $T/solve_forcefix.py ${VEC[$t]} runs/${t}_dev_plain.s0.jsonl runs/${t}_dev_plain.s1.jsonl
done

# 2. IFEval, plain and with the 14 rules, 4 shards each (same bs-24 batches for both modes).
for t in q06 q17 q4; do for K in 0 1 2 3; do
  WSM_MODEL=${HOST[$t]} $PY $T/run_t.py data/ifeval.jsonl runs/${t}_ifeval_plain.s$K.jsonl --mode plain --shard $K/4 --bs 24
  WSM_MODEL=${HOST[$t]} VECS=${VEC[$t]} $PY $T/run_t.py data/ifeval.jsonl runs/${t}_ifeval_imp.s$K.jsonl --mode implant --shard $K/4 --bs 24
done; $PY $T/score_t.py runs/${t}_score.json runs/${t}_ifeval_plain runs/${t}_ifeval_imp | grep -v -i warn > runs/${t}_score.txt; done
$PY $T/fmt_transfer.py > runs/transfer_tables.md                      # markdown tables (reads runs/{q06,q17,q4}_score.json)

# 3. add-one-rule curve. Order = per-rule gain on Team D's FRESH set (needs heldout runs/fresh_{plain,imp}.s*.jsonl, see
#    ../README.md); it is hard-coded in curve2.py (ORDER) and recorded in results/order_fresh_seeded.txt.
WSM_MODEL=Qwen/Qwen3-0.6B DUMP_BATCHES=runs/batches_ifeval_bs24.json $PY $T/run_t.py data/ifeval.jsonl /dev/null --bs 24
for t in q06 q17 q4; do c=c${t#q}
  $PY $T/curve2.py plan $c runs/${t}_ifeval_imp "WSM_MODEL=${HOST[$t]}" ${VEC[$t]}
  $T/run_jobs.sh jobs/curve_$c.txt                                   # 26 batch reruns per host
  $PY $T/curve2.py merge $c runs/${t}_ifeval_plain runs/${t}_ifeval_imp
  $PY $T/score_t.py runs/${c}_curve_score.json $(for k in $(seq 0 14); do echo -n "runs/${c}_k$k.jsonl "; done) | grep -v -i warn > runs/${c}_curve_score.txt
  $PY $T/fmt_curve.py $c > runs/curve_$c.md
done

# 4. text-free per-prompt records (as in results/reduced/)
for t in q06 q17 q4; do for m in plain imp; do
  cat runs/${t}_ifeval_$m.s?.jsonl > runs/${t}_$m.jsonl
  $PY $T/reduce_seeded.py data/ifeval.jsonl runs/${t}_$m.jsonl $T/results/reduced/${t}_ifeval_$m.json --command "ifeval/transfer/run_all.sh"
done; done
