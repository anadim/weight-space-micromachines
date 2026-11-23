"""(a) Per-attempt ("try k") accuracy of the A1 arm (RL with the whole serial stack + per-attempt credit), u0 and RL
seeds 3 / 4 / 6 (run_arm.sh A1 / A1_s2 / A1_s3), GSM8K test and GSM-Hard with N = 8 tries, plus the in-run dev curves.
Same rule as the §16 analysis: try k is right if attempt k wrote a valid register whose value equals gold (rel. 1e-6).
    python3 try8.py                    # shipped res/a1/try8.items.jsonl.gz + ../../gsm_stack/results/curves/A1*.steps.jsonl
    python3 try8.py FILE.jsonl [...]   # original per-item eval jsonl of gsm_stack/eval/fe.sh (rows with natt/rok/sym)
CPU only, no numpy, a few seconds."""
import sys, json, gzip, os, statistics as S

H = os.path.dirname(os.path.abspath(__file__))
PAD = 12
CURVES = os.path.join(os.environ.get('WSM_ROOT', os.path.join(H, '..', '..')), 'gsm_stack', 'results', 'curves')
RUNS = [('u0', 'u0'), ('A1', 'A1 seed 3'), ('A1_s2', 'A1 seed 4'), ('A1_s3', 'A1 seed 6')]


def right(v, gold):
    if v is None: return False
    try:
        x = float(v)
    except ValueError:
        return False
    g = float(gold)
    return abs(x - g) <= 1e-6 * max(1, abs(g))


def per_try(gold, natt, att):
    N = max(natt)
    return [round(100 * sum(right(a[i] if i < len(a) else None, g) for a, g in zip(att, gold)) / len(gold), 1) for i in range(N)]


if len(sys.argv) > 1:
    for fn in sys.argv[1:]:
        R = [json.loads(l) for l in open(fn)]
        att = [[''.join('0123456789-.'[c] for c in x['sym'][i] if c != PAD) if x['rok'][i] else None for i in range(x['natt'])] for x in R]
        print(os.path.basename(fn), 'n', len(R), 'acc', round(100 * sum(x['ok'] for x in R) / len(R), 2),
              'attempt_acc', per_try([x['gold'] for x in R], [x['natt'] for x in R], att))
    sys.exit()

T = {}
with gzip.open(os.path.join(H, 'res', 'a1', 'try8.items.jsonl.gz'), 'rt') as f:
    next(f); gold = {}
    for l in f:
        r = json.loads(l)
        if 'gold' in r: gold[r['set']] = r['gold']; continue
        g = gold[r['set']]
        T[r['set'], r['name']] = (round(100 * r['ok'].count('1') / len(g), 2), per_try(g, r['natt'], r['att']))
D = {}
for n, _ in RUNS[1:]:
    dv = [json.loads(l)['dev'] for l in open(os.path.join(CURVES, f'{n}.steps.jsonl')) if '"dev"' in l]
    D[n] = (dv[-1]['dev_vote8'], dv[-1]['dev_attempt_acc'], dv[-1]['u'])
    D['u0'] = (dv[0]['dev_vote8'], dv[0]['dev_attempt_acc'], dv[0]['u'])


def row(lab, vals, nd=1):
    s = vals[1:]
    return f'| {lab} | ' + ' | '.join(f'{v:.{nd}f}' for v in vals) + f' | **{S.mean(s):.{nd}f}** ({S.stdev(s):.1f}) |'


print('| | u0 | A1 seed 3 | A1 seed 4 | A1 seed 6 | 3-seed mean (sd) |')
print('|---|---|---|---|---|---|')
print(row('GSM8K test, vote N=8', [T['test', n][0] for n, _ in RUNS], 2))
print(row('test, try 1', [T['test', n][1][0] for n, _ in RUNS]))
print(row('test, try 8', [T['test', n][1][7] for n, _ in RUNS]))
print(row('dev (600), vote N=8 (u300)', [D[n][0] for n, _ in RUNS], 2))
print(row('dev, try 8 (u300)', [D[n][1][7] for n, _ in RUNS], 2))
print(row('GSM-Hard, vote N=8', [T['hard', n][0] for n, _ in RUNS], 2))
print(row('GSM-Hard, try 8', [T['hard', n][1][7] for n, _ in RUNS]))
print('\nper-try curves (try 1..8):')
for st in ('test', 'hard'):
    for n, lab in RUNS: print(f'- {st} {lab}: ' + ' '.join(f'{v:.1f}' for v in T[st, n][1]))
for n, lab in RUNS: print(f'- dev {lab} (u{D[n][2]}): ' + ' '.join(f'{v:.2f}' for v in D[n][1]))
