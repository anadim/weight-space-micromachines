"""[41] follow-up: dev-only lr selection at the final horizon (u300) between the lrs run to 300 u for each arm
(T: 1e-6 = T_s*, 4e-6 = fT4_s*; I*: 1e-6 = fI1_s*, 2e-6 = fI_s*). dev 800 (greedy), mean over the train seeds present.
Reads the dev records of tt/res/train_steps.jsonl.gz (= the dev lines of each run's train log).
python3 sel300.py [NSEEDS]   (train seeds 0..NSEEDS-1 scanned, default 20; 8 = the RL_RESULTS §12.6 selection table)"""
import gzip, json, os, statistics as S, sys
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
RES = os.environ.get('C36_RES', os.path.dirname(os.path.abspath(__file__)) + '/tt/res')
STEPS = {}   # run -> steps.jsonl records in order (tt/res/train_steps.jsonl.gz, built by mk_res.py)
for _r in map(json.loads, gzip.open(f'{RES}/train_steps.jsonl.gz', 'rt')):
    STEPS.setdefault(_r.pop('run'), []).append(_r)


def dev(nm, u):
    for r in STEPS.get(nm, []):
        if 'dev' in r and r['dev'].get('u') == u: return r['dev']
    return None


print('| arm | host lr | n | dev acc u100 | dev acc u300 (per seed) | dev scaled u300 |')
print('|---|---|---|---|---|---|')
for arm, lr, pat in (('T', '1e-6', 'T_s%d'), ('T', '2e-6', 'fT2_s%d'), ('T', '4e-6', 'fT4_s%d'), ('I*', '1e-6', 'fI1_s%d'), ('I*', '2e-6', 'fI_s%d'),
                     ('I0', '1e-6', 'fI01_s%d'), ('I0', '2e-6', 'fI0_s%d'), ('Itok', '1e-6', 'fItok1_s%d'), ('Itok', '2e-6', 'fItok_s%d')):
    d3 = [(k, dev(pat % k, 300)) for k in range(NS)]; d3 = [(k, d) for k, d in d3 if d]
    if not d3: print(f'| {arm} | {lr} | 0 | | | |'); continue
    d1 = [dev(pat % k, 100) for k, _ in d3]
    print(f"| {arm} | {lr} | {len(d3)} | {S.mean(d['dev_acc'] for d in d1):.2f} | **{S.mean(d['dev_acc'] for _, d in d3):.2f}** "
          f"({' / '.join(f'{d['dev_acc']:.1f}' for _, d in d3)}) | {S.mean(d['dev_scaled'] for _, d in d3):.2f} |")
