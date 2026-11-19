"""Paths for the call-gate code. Runs against the repo's own micromachines/, bbh_generic/team{A,B,C} and the integration code
(bbh_generic/integrate/code: integ.py, nocall_prompts.py). Override with env vars:
  WSM_REPO   repo root (default: three levels up)
  FT_HOST    the v2 integration host checkpoint (full FT of Qwen3-0.6B-Base on integrate/data/train_union_v2.jsonl; not released)
  BBH_DIR / BBH_COT   official BBH task files / 3-shot CoT prompts (data/download_bbh.sh)
  CG_RESULTS          where probes and run outputs go (default bbh_generic/callgate/results)"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.environ.get('WSM_REPO', os.path.join(HERE, '..', '..', '..')))
G = os.path.join(REPO, 'bbh_generic')
CODE = os.path.join(G, 'integrate', 'code')                       # integ.py, nocall_prompts.py
INTEG_DATA = os.path.join(G, 'integrate', 'data')
CKPT = os.environ.get('FT_HOST', os.path.join(G, 'integrate', 'runs', 'ft_int_v2'))
BBH = os.environ.get('BBH_DIR', os.path.join(REPO, 'data', 'bbh', 'data'))
COT = os.environ.get('BBH_COT', os.path.join(REPO, 'data', 'bbh', 'cot'))
RES = os.environ.get('CG_RESULTS', os.path.join(HERE, '..', 'results'))
for p in (CODE, HERE):
    if p in sys.path: sys.path.remove(p)
    sys.path.insert(0, p)
import _paths  # noqa: F401,E402  (integration's path setup: repo + team code on sys.path)
