"""Paths for the integration step. The code runs against the repo's own micromachines/ and bbh_generic/team{A,B,C}
(the original run used byte-identical vendored copies of both)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.environ.get('WSM_REPO', os.path.join(HERE, '..', '..', '..')))
G = os.path.join(REPO, 'bbh_generic')
BBH = os.environ.get('BBH_DIR', os.path.join(REPO, 'data', 'bbh', 'data'))     # official BBH task files (data/download_bbh.sh)
COT = os.environ.get('BBH_COT', os.path.join(REPO, 'data', 'bbh', 'cot'))      # official 3-shot CoT prompts
VETO = os.environ.get('VETO', os.path.join(REPO, 'runs', 'veto_qwen3.pt'))      # calc veto (scripts/solve_veto.py)
os.environ.setdefault('WSM_REPO', REPO)
for p in (f'{G}/teamC', f'{G}/teamB', f'{G}/teamA', REPO, HERE):
    if p in sys.path: sys.path.remove(p)
    sys.path.insert(0, p)
