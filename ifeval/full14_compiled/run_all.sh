#!/bin/bash
# Build the 14-rule IFEval implant as plain weight tensors and check it against circuit.py. Run from the repo root.
# Needs: one GPU (A100 80GB used; peak < 10 GB), ifeval/data/ifeval.jsonl (python ifeval/data/prepare_ifeval.py),
# ifeval/heldout/vecs2_forcefix.pt (in the repo). Work dir = ifeval/heldout (circuit.py, tables.py, common.py, tables.pt cache).
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd); PY=${PY:-python}; F14="$ROOT/ifeval/full14_compiled"; TR="$ROOT/ifeval/transfer"
cd "$ROOT/ifeval/heldout"; mkdir -p runs_full14
export PYTHONPATH="$ROOT/ifeval/heldout:$PYTHONPATH" HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
[ -e data/ifeval.jsonl ] || ln -s ../../data/ifeval.jsonl data/ifeval.jsonl
ln -sf data/ifeval.jsonl data_ifeval.jsonl                     # default --data of lockstep.py

# 1. compile + export (~20 s build; tables.pt is built on first use by tables.py, ~118 MB cache)
$PY "$F14/build_test.py"                                      # prints ReLU units / live wires / exactness bound per netlist
$PY "$F14/export.py" implant14_sd.pt --device cuda --dtype bf16   # 1.23 GB; sha256 in ifeval/full14_compiled/README.md

# 2. reference circuit.py run and compiled run, same batches (bs 24, max_new 2048, greedy). ~1.5 h each on an A100.
WSM_MODEL=Qwen/Qwen3-0.6B VECS=vecs2_forcefix.pt $PY "$TR/run_t.py" data/ifeval.jsonl runs_full14/ref_imp.jsonl --mode implant --shard 0/1 --bs 24
$PY "$TR/run_t.py" data/ifeval.jsonl runs_full14/plain.jsonl --mode plain --shard 0/1 --bs 24
$PY "$F14/run_c.py" data/ifeval.jsonl runs_full14/comp_imp.jsonl --sd implant14_sd.pt --bs 24 --shadow 1   # --shadow 0 is ~2x faster
$PY "$F14/compare.py" runs_full14/comp_imp.jsonl runs_full14/ref_imp.jsonl --plain runs_full14/plain.jsonl
for r in comp_imp ref_imp plain; do
  $PY "$TR/reduce_seeded.py" data/ifeval.jsonl runs_full14/$r.jsonl runs_full14/$r.reduced.json --command "ifeval/full14_compiled/run_all.sh"
done

# 3. hash / lockstep stress tests (CPU; hours). The stress token sets used are in results/stress/ (gzip).
$PY "$F14/hash_state_test.py" --trials 400 --device cpu
for k in forbid mixed random; do
  gunzip -c "$F14/results/stress/stress_$k.jsonl.gz" > runs_full14/stress_$k.jsonl
  $PY "$F14/lockstep.py" --device cpu --seqs runs_full14/stress_$k.jsonl --batches all --log runs_full14/stress_$k.log
done
