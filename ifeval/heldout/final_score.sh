#!/bin/bash
# Score the final runs (after queue.sh finished the ifeval_* and ifbench_* jobs). Run from ifeval/heldout.
# Needs data/ifeval.jsonl (link ../data/ifeval.jsonl), data/ifbench.jsonl (../data/prep_transfer.py), and for IFBench the
# allenai/IFBench repo cloned into ./IFBench (commit 488918665d53f867fa908a293769ec6e71e74aec; its data/IFBench_test.jsonl copied
# to data/ifbench_orig.jsonl).
cd "$(dirname "$0")"; PY=${PY:-python}
echo "== IFEval ODD (held-out headline)"; $PY score.py data/ifeval.jsonl runs/ifeval_plain runs/ifeval_imp --keys odd 2>&1 | grep -v -i warn
echo "== IFEval EVEN (seen in reworded form)"; $PY score.py data/ifeval.jsonl runs/ifeval_plain runs/ifeval_imp --keys even 2>&1 | grep -v -i warn
echo "== IFEval ALL 541"; $PY score.py data/ifeval.jsonl runs/ifeval_plain runs/ifeval_imp 2>&1 | grep -v -i warn
cat runs/ifbench_plain.s*.jsonl > runs/ifbench_plain_all.jsonl; cat runs/ifbench_imp.s*.jsonl > runs/ifbench_imp_all.jsonl
echo "== IFBench"; $PY ifbench_prep.py
$PY score_ifbench.py runs/ifbench_plain_p.jsonl runs/ifbench_imp_p.jsonl 2>&1 | grep -v -i warn
echo FINALDONE
