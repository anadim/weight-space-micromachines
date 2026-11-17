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

L = [l for l in open(R + "/jobs.txt") if l.strip()]
ref = [l for l in L if l.startswith("evalA_ftv3")]; rest = [l for l in L if not l.startswith("evalA_ftv3")]
tasks = sorted(f[:-5] for f in _os.listdir(BBH))[::-1]
rev = f"b27_baseC_rev||python {FT}/scripts/run27.py {R}/out/b27_base_toolsC_rev.json --tools C --max_new 400 --tasks {','.join(tasks)}\n"
open(R + "/jobs.tmp", "w").write("".join(rest + [rev] + ref)); _os.replace(R + "/jobs.tmp", R + "/jobs.txt")
