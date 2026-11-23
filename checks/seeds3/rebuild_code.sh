#!/bin/bash
# usage: bash checks/seeds3/rebuild_code.sh TARGET
# Rebuilds the frozen code tree the seeds3 runs (and the LIB / LIBscr control pairs) used: the release's bbh_generic, data,
# ifeval, micromachines and rl dirs, plus code_overlay/ (3 trainers with the resume / per-item patches and
# rl/library/mixed/make_nl_sft.py, the worked-solution SFT rows of arm aNL; see README.md), with
# the four stale rl/library/{common,engine,fastqwen,implant}.py moved to rl/library/_repo_stale/ so that train_library.py
# picks up the rl/wiring versions through RLW_DIR. Then run `python3 data/prepare_gsm.py` in TARGET (GSM8K jsonl + shots)
# and `python3 mixed_data.py` in TARGET/rl/library/mixed (mixed-task data; checked against DATA_SHA256.txt); run_seed.sh
# does both. The W3d init links (rl/results/wiring/H100/sft3_W3e_lr1e-2/links_final.pt) go to TARGET/rl/wiring/runs/.
set -e
T=${1:?usage: $0 TARGET}
HERE=$(cd "$(dirname "$0")" && pwd); WSM_ROOT=${WSM_ROOT:-$(cd $HERE/../.. && pwd)}
[ -e "$T" ] && { echo "$T exists; give a new dir"; exit 2; }
mkdir -p "$T"
for d in bbh_generic data ifeval micromachines rl; do cp -r "$WSM_ROOT/$d" "$T/"; done
mkdir -p "$T/rl/library/_repo_stale"
for f in common engine fastqwen implant; do mv "$T/rl/library/$f.py" "$T/rl/library/_repo_stale/"; done
cat > "$T/rl/library/_repo_stale/README.txt" <<'TXT'
These four files are older copies of rl/wiring/{implant,engine,fastqwen,common}.py. train_library.py / lib_engine.py need
the rl/wiring versions (lib_engine calls CalcImplant.step(emb, B, tok) and uses the wiring Engine), which they get via
RLW_DIR on sys.path once these stale copies no longer shadow them. With the stale copies in place the trainer dies at the
first generate: "CalcImplant.step() takes 3 positional arguments but 4 were given".
TXT
cp -r "$HERE/code_overlay/." "$T/"
mkdir -p "$T/rl/wiring/runs/sft3_W3e_lr1e-2"
cp "$WSM_ROOT/rl/results/wiring/H100/sft3_W3e_lr1e-2/links_final.pt" "$T/rl/wiring/runs/sft3_W3e_lr1e-2/"
find "$T" -name __pycache__ -prune -exec rm -rf {} +
echo "rebuilt $T"
