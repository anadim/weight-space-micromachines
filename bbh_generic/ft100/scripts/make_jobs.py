"""Write jobs.txt: NAME|DEPS(comma)|COMMAND. Priority = file order."""
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
CA = f'{G}/teamA'; CB = f'{G}/teamB'
_os.makedirs(R, exist_ok=True)
EP_A, EP_B = 10, 15          # same for every n within a team; see NOTEBOOK (batch sizes differ between teams)
J = []
ck = [(100, 0), (32, 0), (300, 0), (100, 1)]
for n, s in ck:
    J.append((f'trainB_n{n}_s{s}', '', f'cd {FT}/sub/B_n{n}_s{s} && python {CB}/ft_B.py --out {R}/runs/B_n{n}_s{s} --epochs {EP_B} --n_main 100000 --n_aux 100000 --seed {s}'))
    J.append((f'trainA_n{n}_s{s}', '', f'cd {CA} && python ft.py --data {FT}/sub/A_n{n}_s{s}.jsonl --out {R}/runs/A_n{n}_s{s} --epochs {EP_A} --lr 1e-5 --tok_budget 16000 --seed {s}'))
BS = 'temporal_test,navigate_test,lies_test,meeting_oos,robot_oos,knights_oos,temporal_bbh,navigate_bbh,lies_bbh'
for n, s in ck:
    c = f'n{n}_s{s}'
    for arm, t in [('tools', 'all'), ('notools', 'none')]:
        J.append((f'evalB_{c}_{arm}', f'trainB_{c}', f'cd {CB} && python run_eval.py {R}/out/B_{c}_{arm}.json --model {R}/runs/B_{c} --sets {BS} --tools {t}'))
    for arm, t in [('tools', 'track_gen,order_gen'), ('notools', 'none')]:
        J.append((f'evalA_{c}_{arm}', f'trainA_{c}', f'cd {CA} && python run_eval.py {R}/evalA_ft.jsonl {R}/out/A_{c}_{arm}.json --model {R}/runs/A_{c} --tools {t} --bs 32'))
    if (n, s) == (32, 0):
        J.append(('evalA_base', '', f'cd {CA} && python run_eval.py {R}/evalA_base.jsonl {R}/out/A_base_testoos2.json --bs 32'))
V3 = TEAM_A_FT   # Team A ft_v3 host (reference row), read-only
for arm, t in [('tools', 'track_gen,order_gen'), ('notools', 'none')]:
    J.append((f'evalA_ftv3_{arm}', '', f'cd {CA} && python run_eval.py {R}/evalA_base.jsonl {R}/out/A_ftv3ref_{arm}_testoos2.json --model {V3} --tools {t} --bs 32'))
with open(f'{R}/jobs.txt', 'w') as f:
    for name, dep, cmd in J: f.write(f'{name}|{dep}|{cmd}\n')
print(len(J), 'jobs')
