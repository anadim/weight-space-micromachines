"""worker.py GPU TAG -- pull jobs from jobs.txt in order (deps must be done), claim with mkdir, run, mark done."""
import os, sys, time, subprocess
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
gpu, tag = sys.argv[1], sys.argv[2]
os.makedirs(f'{R}/logs', exist_ok=True); os.makedirs(f'{R}/out', exist_ok=True); os.makedirs(f'{R}/runs', exist_ok=True)
os.makedirs(f'{R}/locks', exist_ok=True); os.makedirs(f'{R}/done', exist_ok=True)
env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=REPO, TOKENIZERS_PARALLELISM='false')
while True:
    jobs = [l.rstrip('\n').split('|', 2) for l in open(f'{R}/jobs.txt') if l.strip()]
    pending = [j for j in jobs if not os.path.exists(f'{R}/locks/{j[0]}')]
    if not pending: break
    ready = [j for j in pending if all(os.path.exists(f'{R}/done/{d}') for d in j[1].split(',') if d)]
    got = None
    for name, dep, cmd in ready:
        try: os.mkdir(f'{R}/locks/{name}'); got = (name, cmd); break
        except FileExistsError: continue
    if got is None: time.sleep(30); continue
    name, cmd = got; t0 = time.time()
    print(f'[{tag}] start {name} {time.strftime("%H:%M:%S")}', flush=True)
    rc = subprocess.call(cmd, shell=True, env=env, stdout=open(f'{R}/logs/{name}.log', 'w'), stderr=subprocess.STDOUT)
    print(f'[{tag}] end {name} rc={rc} {time.time() - t0:.0f}s', flush=True)
    if rc == 0: open(f'{R}/done/{name}', 'w').close()
print(f'[{tag}] no jobs left', flush=True)
