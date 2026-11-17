"""Append the all-27-task jobs (coordinator request) to jobs.txt; workers pick them up in file order."""
import os as _os
FT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))          # bbh_generic/ft100
REPO = _os.environ.get('WSM_REPO', _os.path.dirname(_os.path.dirname(FT)))     # repo root
R = _os.environ.get('BFT_R', _os.path.join(FT, 'work'))                        # work dir: runs/ out/ logs/ locks/ jobs.txt
G = _os.path.join(REPO, 'bbh_generic')                                         # team code (unchanged)
BBH = _os.environ.get('BBH_DIR', _os.path.join(REPO, 'data', 'bbh', 'data'))  # official BBH task files (data/download_bbh.sh)
COT = _os.environ.get('BBH_COT', _os.path.join(REPO, 'data', 'bbh', 'cot'))
PEEK = _os.environ.get('PEEK_OUT', _os.path.join(G, 'peek', 'out'))           # bbh_peek arm outputs (peek/scripts/run_job.sh)
TEAM_A_FT = _os.environ.get('TEAM_A_FT', _os.path.join(REPO, 'runs', 'teamA_ft_v3'))  # not released; teamA/ft.py on train3
VETO = _os.environ.get('VETO', _os.path.join(REPO, 'runs', 'veto_qwen3.pt'))  # scripts/solve_veto.py
S = f'{FT}/scripts/run27.py'
J = [('b27_baseC', '', f'python {S} {R}/out/b27_base_toolsC.json --tools C --max_new 400'),
     ('b27_A100', 'trainA_n100_s0', f'python {S} {R}/out/b27_A_n100_s0_toolsA.json --model {R}/runs/A_n100_s0 --tools A --max_new 600'),
     ('b27_B100', 'trainB_n100_s0', f'python {S} {R}/out/b27_B_n100_s0_toolsB.json --model {R}/runs/B_n100_s0 --tools B --max_new 400'),
     ('b27_base', '', f'python {S} {R}/out/b27_base_notools.json --tools none --max_new 400')]
with open(f'{R}/jobs.txt', 'a') as f:
    for n, d, c in J: f.write(f'{n}|{d}|{c}\n')
