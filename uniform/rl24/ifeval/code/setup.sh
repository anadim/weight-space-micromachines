#!/bin/bash
# usage: bash uniform/rl24/ifeval/code/setup.sh      (CPU, ~2 min; idempotent; also used by rl28/ifeval and checks/seeds35)
# Completes this code dir in place: copies the files it shares unchanged with the rest of the release (the original job
# snapshot held copies), the IFEval-541 prompts, and regenerates the RL train / dev data (gen_data seeds 200 / 201 + filter.py;
# the sha256 of the filtered files must match gen/filter.log).
set -e
C="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; R=${WSM_ROOT:-$(cd "$C/../../../.." && pwd)}
cd "$C"; mkdir -p f14 heldout transfer stk ifdata gen
cpn() { local d=$2; [ -d "$d" ] && d="$d/$(basename "$1")"; [ -e "$d" ] || cp "$1" "$d"; }
for f in build_test check_shared circ14 compare execu export lockstep_util run_c sym; do cpn $R/ifeval/full14_compiled/$f.py f14/; done
cpn $R/micromachines/hdl.py f14/wsm_hdl.py
for f in circuit.py common.py gen_data.py tables.py vecs2_forcefix.pt; do cpn $R/ifeval/heldout/$f heldout/; done
for f in common run_t reduce_seeded; do cpn $R/ifeval/transfer/$f.py transfer/; done
cpn $R/gsm_stack/engine/wiring/hostkern.py stk/
for f in lmhead.cu tattn.py tk.py tk_compile.py; do cpn $R/gsm_stack/engine/stk/$f stk/; done
if [ ! -f ifdata/ifeval.jsonl ]; then
  [ -f $R/ifeval/data/ifeval.jsonl ] || python3 $R/ifeval/data/prepare_ifeval.py   # needs the `datasets` package (HF google/IFEval)
  cp $R/ifeval/data/ifeval.jsonl ifdata/
fi
echo "77a7897eda0eac35809310f5ffb655dd  ifdata/ifeval.jsonl" | md5sum -c --quiet || echo "WARNING: ifdata/ifeval.jsonl differs from the file of the original runs"
if [ ! -f gen/train.jsonl ] || [ ! -f gen/dev.jsonl ]; then
  (cd heldout && python3 gen_data.py --seed 200 --per 1450 --out ../gen/train_s200.jsonl && python3 gen_data.py --seed 201 --per 30 --out ../gen/dev_s201.jsonl)
  cp gen/filter.log gen/filter.log.orig; cp gen/filter_report.json gen/filter_report.json.orig
  python3 gen/filter.py > gen/filter.log.new
  mv gen/filter.log.orig gen/filter.log; mv gen/filter_report.json.orig gen/filter_report.json
fi
sha256sum gen/train.jsonl gen/dev.jsonl
echo "expected: fdbec3d19f4497cb... gen/train.jsonl (18219 rows), 2f314da1fe2b90bd... gen/dev.jsonl (415 rows)"
