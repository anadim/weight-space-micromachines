"""jobs.txt for the natural-language control. Same team scripts and hyperparameters as ../../ft100/scripts/make_jobs.py; only
the training rows differ (NL worked solutions instead of call strings, make_nl.py) and the fine-tune scripts are the
chunked-CE copies (memory only). Eval: tools none, larger max_new (also scored at the original 600/400 budget, score_nl.py).
Run the queue with ../../ft100/scripts/worker.py and the same BFT_R."""
import os
HERE = os.path.dirname(os.path.abspath(__file__)); G = os.path.abspath(os.path.join(HERE, '..', '..'))   # bbh_generic
R = os.environ.get('BFT_R', os.path.join(HERE, '..', 'work'))                     # work dir: nl/ runs/ out/ logs/ jobs.txt
BF = os.environ.get('FT100_R', os.path.join(G, 'ft100', 'work'))                  # ft100 work dir (evalA_ft.jsonl from make_eval_sets.py)
CA = f'{G}/teamA'; CB = f'{G}/teamB'
EP_A, EP_B = 10, 15
BS = 'temporal_test,navigate_test,lies_test,meeting_oos,robot_oos,knights_oos,temporal_bbh,navigate_bbh,lies_bbh'
os.makedirs(R, exist_ok=True); J = []
for n, s in [(100, 0), (100, 1), (300, 0), (32, 0)]:
    c = f'n{n}_s{s}'
    J.append((f'trainA_{c}', '', f'cd {CA} && python {HERE}/ft_A_chunked.py --data {R}/nl/A_{c}.jsonl --out {R}/runs/A_nl_{c} --epochs {EP_A} --lr 1e-5 --tok_budget 16000 --seed {s}'))
    J.append((f'evalA_{c}', f'trainA_{c}', f'cd {CA} && python run_eval.py {BF}/evalA_ft.jsonl {R}/out/A_nl_{c}.json --model {R}/runs/A_nl_{c} --tools none --bs 32 --max_new 1000'))
    J.append((f'trainB_{c}', '', f'cd {R}/nl/B_{c} && python {HERE}/ft_B_chunked.py --out {R}/runs/B_nl_{c} --epochs {EP_B} --n_main 100000 --n_aux 100000 --seed {s}'))
    J.append((f'evalB_{c}', f'trainB_{c}', f'cd {CB} && python run_eval.py {R}/out/B_nl_{c}.json --model {R}/runs/B_nl_{c} --sets {BS} --tools none --max_new 700'))
with open(f'{R}/jobs.txt', 'w') as f:
    for name, dep, cmd in J: f.write(f'{name}|{dep}|{cmd}\n')
print(len(J), 'jobs ->', f'{R}/jobs.txt')
