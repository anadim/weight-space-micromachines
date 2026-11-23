# rl28/ifeval common env (task [28], IFEval on Qwen3-1.7B-Base; = uniform/rl24/ifeval/code/env.sh with the rl28 changes).
# Run `bash setup.sh` once first (it completes this code dir from uniform/rl24/ifeval/code and the rest of the release).
_C="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSM_ROOT=${WSM_ROOT:-$(cd "$_C/../../../.." && pwd)}
source "$WSM_ROOT/uniform/env.sh"                 # WSM_ROOT, WSM_DATA, RL24_DATA, RL28_DATA, triton / caches, your venv
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} PYTHONUNBUFFERED=1
export DATA=${IF_DATA:-$RL28_DATA/ifeval}; mkdir -p $DATA
# host: fastqwen.py takes (D, FF) from HOST_MODEL; every script passes the same name as --model
export HOST_MODEL=${HOST_MODEL:-Qwen/Qwen3-1.7B-Base}
# IFEval checkers (lm_eval 0.4.13 + langdetect 1.0.9): shared with rl24/ifeval (same --target dir)
R24=${IF24_DATA:-$RL24_DATA/ifeval}; mkdir -p $R24
export PYL=$R24/pylib NLTK_DATA=$R24/nltk_data
prep_pylib() { ( flock 9; _prep_pylib ) 9> $R24/pylib.lock; }
_prep_pylib() {
  PYTHONPATH=$PYL python -c "import langdetect, nltk, immutabledict; import importlib.util as u; assert u.find_spec('lm_eval')" 2>/dev/null && return 0
  pip install -q --target $PYL --no-deps lm_eval==0.4.13 langdetect==1.0.9 && pip install -q --target $PYL nltk immutabledict absl-py six
  PYTHONPATH=$PYL python -c "import nltk; nltk.download('punkt_tab', download_dir='$NLTK_DATA', quiet=True); nltk.download('punkt', download_dir='$NLTK_DATA', quiet=True)"
}
build_tables() { ( flock 9; _build_tables ) 9> $DATA/tables.lock; }
_build_tables() {   # in heldout/: vocab tables for the Qwen3 tokenizer (cached under $DATA; tokenizer-only, same as 0.6B: reuse rl24's)
  [ -e $DATA/tables.pt ] || [ ! -e $R24/tables.pt ] || { cp $R24/tables.pt $DATA/tables.pt.tmp && cp $R24/vocab_strs.json $DATA/vocab_strs.json && mv $DATA/tables.pt.tmp $DATA/tables.pt; }
  if [ -e $DATA/tables.pt ]; then cp $DATA/tables.pt $DATA/vocab_strs.json . 2>/dev/null; return 0; fi
  PYTHONPATH=. python -c "
from transformers import AutoTokenizer
from common import vocab_strings
from tables import build
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B'); build(tok, vocab_strings(tok), device='cpu'); print('tables built')" && cp tables.pt $DATA/tables.pt.tmp && cp vocab_strs.json $DATA/vocab_strs.json.tmp && mv $DATA/vocab_strs.json.tmp $DATA/vocab_strs.json && mv $DATA/tables.pt.tmp $DATA/tables.pt
}
