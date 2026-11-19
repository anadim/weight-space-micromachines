#!/bin/bash
# The frozen Team D 14-rule IFEval implant ported to SmolLM2-1.7B-Instruct (ChatML host). Run from the repo root; one GPU
# (an A40 was used). The port is a deterministic text transform of ifeval/heldout/{circuit,common,run,tables}.py:
#   port_ifeval.py (v1: host constants only)  ->  port_v2.py (v2.1: fixed emitter strings as host token sequences; post-diagnosis)
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd); PY=${PY:-python}; X="$ROOT/ifeval/xfamily"; W=${WORK:-$ROOT/ifeval/xfamily/work}
export IFE_MODEL=HuggingFaceTB/SmolLM2-1.7B-Instruct
$PY "$X/port_ifeval.py" "$ROOT/ifeval/heldout" "$W/v1"         # copies + edits circuit.py, common.py, run.py, tables.py
cp -r "$W/v1" "$W/v21" && $PY "$X/port_v2.py" "$W/v21"
for v in v1 v21; do
  cd "$W/$v"; mkdir -p runs data; cp "$X/solve_force.py" "$ROOT/ifeval/heldout/score.py" .
  ln -sf "$ROOT/ifeval/heldout/data/dev.jsonl" data/dev.jsonl; ln -sf "$ROOT/ifeval/data/ifeval.jsonl" data/ifeval.jsonl
  [ $v = v21 ] && cp -n ../v1/vecs_sm2i.pt ../v1/runs/odd_plain.jsonl runs/ 2>/dev/null && mv -n runs/vecs_sm2i.pt . ; true
  ODD=$($PY -c "import json;print(','.join(str(json.loads(l)['key']) for l in open('data/ifeval.jsonl') if json.loads(l)['key']%2))")
  # force_fix rows from the host's own plain answers to OUR dev prompts (39 tokens; ifeval/xfamily/vecs_sm2i.pt is the original,
  # sha256 c5577724...; copy it here to skip the solve)
  [ -f runs/dev_plain.jsonl ] || $PY run.py data/dev.jsonl runs/dev_plain.jsonl --mode plain --bs 32 --limit 250 --max_new 700
  [ -f vecs_sm2i.pt ] || $PY solve_force.py vecs_sm2i.pt runs/dev_plain.jsonl
  [ -f runs/odd_plain.jsonl ] || $PY run.py data/ifeval.jsonl runs/odd_plain.jsonl --mode plain --bs 32 --keys $ODD
  VECS=vecs_sm2i.pt $PY run.py data/ifeval.jsonl runs/odd_imp.jsonl --mode implant --bs 32 --keys $ODD
  $PY score.py data/ifeval.jsonl runs/odd_plain.jsonl runs/odd_imp.jsonl --keys odd 2>&1 | grep -v -i warn > runs/score_odd.txt
  cat runs/score_odd.txt
done
