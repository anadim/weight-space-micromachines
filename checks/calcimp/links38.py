"""[38] addendum: arm I's trained wiring over training (link norms, in-wire admit pattern by operand size / operator).
Mean over the fI_s* runs present (per-update records of tt/res/train_steps.jsonl.gz); each cell averages the logged updates in (u-10, u].
Admit = the in-wire gate's admit rate on <calc> results at train rollouts (T = 1); dN = largest operand digit count.
python3 links38.py [PATTERN [NSEEDS]]   (default fI_s%d; e.g. fI1_s%d for host lr 1e-6, fItok_s%d for the writer-only control;
train seeds 0..NSEEDS-1 scanned, default 20; res/links38.txt = fI_s%d 8 and fItok_s%d 8)"""
import gzip, json, os, sys, statistics as S
RES = os.environ.get('C36_RES', os.path.dirname(os.path.abspath(__file__)) + '/tt/res')
STEPS = {}   # run -> steps.jsonl records in order (tt/res/train_steps.jsonl.gz, built by mk_res.py)
for _r in map(json.loads, gzip.open(f'{RES}/train_steps.jsonl.gz', 'rt')):
    STEPS.setdefault(_r.pop('run'), []).append(_r)
pat = sys.argv[1] if len(sys.argv) > 1 else 'fI_s%d'
U = [10, 50, 100, 150, 200, 250, 300]
runs = []
for k in range(int(sys.argv[2]) if len(sys.argv) > 2 else 20):
    if pat % k in STEPS:
        rs = [r for r in STEPS[pat % k] if 'dev' not in r and 'u' in r]
        if rs and rs[-1]['u'] >= 300: runs.append((k, rs))
if not runs: sys.exit('no finished runs for ' + pat)
print(f'{pat}: train seeds {"/".join(str(k) for k, _ in runs)}')


def cell(key, u, sub=None):
    vals = []
    for _, rs in runs:
        w = [r for r in rs if u - 10 < r['u'] <= u]
        x = [(r.get(sub) or {}).get(key) if sub else r.get(key) for r in w]
        x = [v for v in x if v is not None]
        if x: vals.append(S.mean(x))
    return S.mean(vals) if vals else None


def row(lab, key, sub=None, f='%.3f'):
    v = [cell(key, u, sub) for u in U]
    if all(x is None for x in v): return
    print(f'| {lab} | ' + ' | '.join('-' if x is None else f % x for x in v) + ' |')


print('| quantity | ' + ' | '.join(f'u{u}' for u in U) + ' |'); print('|---' * (len(U) + 1) + '|')
for k, lab in (('admit', 'admit (all calls)'), ('admit_d1_2', 'admit, operands ≤ 2 digits'), ('admit_d3_4', 'admit, 3–4 digits'),
               ('admit_d5p', 'admit, ≥ 5 digits'), ('admit_add', 'admit, +'), ('admit_sub', 'admit, −'), ('admit_mul', 'admit, ×'),
               ('admit_div', 'admit, ÷')):
    row(lab, k)
row('calls / row', 'tool_calls', f='%.2f')
row('reward', 'reward')
for k, lab in (('writer_dW', 'writer ‖ΔW‖/‖W0‖'), ('writer_gate', 'writer gate'), ('cw7_B', 'link L7 ‖B‖'), ('cw14_B', 'link L14 ‖B‖'),
               ('cw21_B', 'link L21 ‖B‖'), ('cwfinal_B', 'link final ‖B‖'), ('cw7_out', 'link L7 output norm'),
               ('cw14_out', 'link L14 output norm'), ('cw21_out', 'link L21 output norm'), ('cwfinal_out', 'link final output norm'),
               ('iw_b', 'in-wire bias b'), ('iw_v', 'in-wire ‖v‖')):
    row(lab, k, 'ci', '%.3g')
