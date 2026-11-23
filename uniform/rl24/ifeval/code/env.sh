# rl24/ifeval common env (sourced by the job scripts from this code dir). Run `bash setup.sh` once first (fills in the
# files shared unchanged with the rest of the release and regenerates the RL data).
_C="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSM_ROOT=${WSM_ROOT:-$(cd "$_C/../../../.." && pwd)}
source "$WSM_ROOT/uniform/env.sh"                 # WSM_ROOT, WSM_DATA, RL24_DATA, FT2_HOST, triton / caches, your venv
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} PYTHONUNBUFFERED=1
export DATA=${IF_DATA:-$RL24_DATA/ifeval}; mkdir -p $DATA
# IFEval checkers (lm_eval 0.4.13 + langdetect 1.0.9): own --target dir, installed once
export PYL=$DATA/pylib NLTK_DATA=$DATA/nltk_data
prep_pylib() { ( flock 9; _prep_pylib ) 9> $DATA/pylib.lock; }
_prep_pylib() {
  PYTHONPATH=$PYL python -c "import langdetect, nltk, immutabledict; import importlib.util as u; assert u.find_spec('lm_eval')" 2>/dev/null && return 0
  pip install -q --target $PYL --no-deps lm_eval==0.4.13 langdetect==1.0.9 && pip install -q --target $PYL nltk immutabledict absl-py six
  PYTHONPATH=$PYL python -c "import nltk; nltk.download('punkt_tab', download_dir='$NLTK_DATA', quiet=True); nltk.download('punkt', download_dir='$NLTK_DATA', quiet=True)"
}
build_tables() { ( flock 9; _build_tables ) 9> $DATA/tables.lock; }
_build_tables() {   # in heldout/: vocab tables for the Qwen3 tokenizer (cached under $DATA)
  if [ -e $DATA/tables.pt ]; then cp $DATA/tables.pt $DATA/vocab_strs.json . 2>/dev/null; return 0; fi
  PYTHONPATH=. python -c "
from transformers import AutoTokenizer
from common import vocab_strings
from tables import build
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B'); build(tok, vocab_strings(tok), device='cpu'); print('tables built')" && cp tables.pt $DATA/tables.pt.tmp && cp vocab_strs.json $DATA/vocab_strs.json.tmp && mv $DATA/vocab_strs.json.tmp $DATA/vocab_strs.json && mv $DATA/tables.pt.tmp $DATA/tables.pt
}
