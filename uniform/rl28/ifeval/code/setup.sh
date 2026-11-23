#!/bin/bash
# usage: bash uniform/rl28/ifeval/code/setup.sh      (CPU, ~2 min; idempotent)
# Completes this code dir in place. The original rl28 job snapshot was a copy of rl24/ifeval/code with the files shipped
# here changed for the 1.7B host (env.sh, eng/{check,hostdev,rl_if}.py, hostchk/mk_sd.py, stk/fastqwen.py) plus the setup /
# check jobs (eng/job_setup17.sh, eng/job_check17.sh). Everything else is copied from uniform/rl24/ifeval/code (completed
# first by its own setup.sh: shared release files, IFEval-541 prompts, regenerated RL train / dev data), plus
# ifeval/transfer/solve_forcefix.py and ifeval/heldout/data/{dev,train2}.jsonl (force_fix re-solve on the 1.7B host).
set -e
C="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; R=${WSM_ROOT:-$(cd "$C/../../../.." && pwd)}
C24=$R/uniform/rl24/ifeval/code
bash $C24/setup.sh
cd "$C"
(cd $C24 && find . -type f ! -name setup.sh ! -name '*.pyc' ! -path './gen/train_s200.jsonl' ! -path './gen/dev_s201.jsonl' ! -name '*.new') |
  while read -r f; do [ -e "$f" ] || { mkdir -p "$(dirname "$f")"; cp "$C24/$f" "$f"; }; done
mkdir -p transfer heldout/data
[ -e transfer/solve_forcefix.py ] || cp $R/ifeval/transfer/solve_forcefix.py transfer/
for f in dev train2; do [ -e heldout/data/$f.jsonl ] || cp $R/ifeval/heldout/data/$f.jsonl heldout/data/; done
echo "rl28/ifeval/code complete"
