#!/bin/bash
# IFEval baselines for the 14-rule implant (Qwen3-0.6B chat, frozen). Run from the repo root; needs ifeval/data/ifeval.jsonl
# (python ifeval/data/prepare_ifeval.py) and one GPU. The scripts import circuit.py / common.py from ifeval/heldout.
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd); PY=${PY:-python}
cd "$ROOT/ifeval/heldout"; mkdir -p runs_base
export PYTHONPATH="$ROOT/ifeval/heldout:$PYTHONPATH"
[ -e data/ifeval.jsonl ] || ln -s ../../data/ifeval.jsonl data/ifeval.jsonl
# (a) thinking ON, plain host, same budget as the implant (max_new 2048), greedy, bs 16 (~2.5 h on one A40)
$PY ../baselines/run_base.py data/ifeval.jsonl runs_base/think_plain.jsonl --mode plain --bs 16 --max_new 2048 --think 1
$PY ../baselines/strip_think.py runs_base/think_plain.jsonl          # -> runs_base/think_plain.ans.jsonl (answer after </think>)
$PY ../data/reduce_ifeval_run.py data/ifeval.jsonl runs_base/think_plain.ans.jsonl ../baselines/results/ifeval_think_plain.ans.json \
    --command "run_base.py --think 1 --bs 16 --max_new 2048; strip_think.py"
# (b) silence at scale: 1000 wikitext-103 passages + 600 PyTorch source files as user messages; count rule latches (CPU, ~30 min)
$PY ../baselines/ifeval_silence.py                                    # -> runs_base/ifeval_silence.json
