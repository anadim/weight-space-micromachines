"""Build the eval files (copies; no item inspection).
A: evalA_ft.jsonl = fresh test (seed 1, 600) + held-out oos2 (180, task renamed 'oos2_<task>') + real BBH (peek/bbh_A.jsonl made by peek/scripts/make_A_jsonl.py, 1500).
   evalA_base.jsonl = test + oos2 only (frozen base on real BBH is reused from bbh_peek A_base, same script).
B: data/{temporal,navigate,lies}_bbh.jsonl from the official BBH json (fields question/target), so Team B's run_eval.py
   reads real BBH the same way peek/scripts/peek_B.py does."""
import json
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
A = f'{G}/teamA/data'; B = f'{G}/teamB/data'   # writes teamB/data/*_bbh.jsonl (BBH copies; git-ignored, not shipped)
_os.makedirs(R, exist_ok=True)
test = [json.loads(l) for l in open(f'{A}/test.jsonl')]
oos2 = [json.loads(l) for l in open(f'{A}/oos2.jsonl')]
for r in oos2: r['task'] = 'oos2_' + r['task']; r['id'] = 'oos2_' + str(r['id'])
bbh = [json.loads(l) for l in open(_os.environ.get('BBH_A', f'{G}/peek/bbh_A.jsonl'))]
for name, rows in [('evalA_ft', test + oos2 + bbh), ('evalA_base', test + oos2)]:
    with open(f'{R}/{name}.jsonl', 'w') as f:
        for r in rows: f.write(json.dumps(r) + '\n')
    print(name, len(rows))
for t, bn in [('temporal', 'temporal_sequences'), ('navigate', 'navigate'), ('lies', 'web_of_lies')]:
    ex = json.load(open(f'{BBH}/{bn}.json'))['examples']
    with open(f'{B}/{t}_bbh.jsonl', 'w') as f:
        for i, e in enumerate(ex): f.write(json.dumps({'task': t, 'id': f'{bn}_{i}', 'question': e['input'], 'target': e['target']}) + '\n')
    print(t, len(ex))
