#!/bin/bash
# CPU, ~1 min: fetch the five off-task benchmarks from the HF Hub (parquet) and write the jsonl files ot_hook.py reads.
#   bash checks/offtask/dl.sh [DIR]     DIR default $WSM_WORK/offtask/bench (= ot_job.sh's OT_DATA)
# Needs curl and pyarrow (pip install pyarrow; the original runs used pyarrow 26.0.1). Then check the files:
#   (cd DIR && sha256sum -c $WSM_ROOT/checks/offtask/bench.sha256)
# Files: mmlu_test (14042) + mmlu_dev (285, the 5-shot pool), arcc_test (1172), hs_val (10042), wg_val (1267),
# nq_val (3610) + nq_train (first 2000 rows, the 5-shot pool).
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$D/../../gsm_stack/env.sh"
O=${1:-$WSM_WORK/offtask/bench}; mkdir -p $O/raw
dl() { local n=$1; shift
  for t in 1 2 3 4 5 6; do for u in "$@"; do
    curl -sL --fail --max-time 300 -o $O/raw/$n.parquet "https://huggingface.co/datasets/$u" && [ -s $O/raw/$n.parquet ] &&
      python3 -c "import pyarrow.parquet as pq; pq.read_table('$O/raw/$n.parquet')" 2>/dev/null && { echo "ok $n"; return; }
  done; sleep 10; done; echo "FAIL $n"; }
dl mmlu_test "cais/mmlu/resolve/main/all/test-00000-of-00001.parquet"
dl mmlu_dev "cais/mmlu/resolve/main/all/dev-00000-of-00001.parquet"
dl arcc_test "allenai/ai2_arc/resolve/refs%2Fconvert%2Fparquet/ARC-Challenge/test/0000.parquet"
dl hs_val "Rowan/hellaswag/resolve/main/data/validation-00000-of-00001.parquet" \
          "Rowan/hellaswag/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet"
dl wg_val "allenai/winogrande/resolve/refs%2Fconvert%2Fparquet/winogrande_xl/validation/0000.parquet"
dl nq_val "google-research-datasets/nq_open/resolve/refs%2Fconvert%2Fparquet/nq_open/validation/0000.parquet"
dl nq_train "google-research-datasets/nq_open/resolve/refs%2Fconvert%2Fparquet/nq_open/train/0000.parquet"
O=$O python3 - <<'PY'
import pyarrow.parquet as pq, json, glob, os
O = os.environ['O']
for f in sorted(glob.glob(f'{O}/raw/*.parquet')):
    t = pq.read_table(f).to_pylist()
    n = os.path.basename(f)[:-8]
    if n == 'nq_train': t = t[:2000]
    with open(f'{O}/{n}.jsonl', 'w') as fh:
        for r in t: fh.write(json.dumps(r, default=str) + '\n')
    print(n, len(t), list(t[0].keys()))
PY
rm -rf $O/raw
