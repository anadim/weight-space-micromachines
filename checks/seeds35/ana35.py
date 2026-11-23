"""seeds35 analysis: GSM8K 0.6B rows 2 (P64), 2b (P64 + untrained rung 2 at eval) and 4 (CW + rung 2), RL seeds 0-5.
Same loader and paired item bootstrap as the uniform-rows analysis (per-item mean over runs, 2000 resamples, rng seed 0,
95% CI = sorted[50], sorted[1949]). CPU only, no numpy:
    cd checks/seeds35 && python3 ana35.py --stamp 2026-11-23T10:40Z > AUTO_final.md
Inputs (all shipped): res/fe/<set>.items.jsonl.gz (per-item ok/ntok of the RL seeds 3-5 eval runs, reduced from the fe.sh
rows <set>_<name>.jsonl); the RL seeds 0-2 runs (the §17 runs) from the release's uniform/final_eval/items bundles via
uniform/items.py (WSM_ROOT overrides the release root; FE_ROWS / FE_ITEMS as in items.py); res/dev.json (in-run dev records of the 12 RL arms), res/fe_audit.txt ("FE audit" lines of the
packed eval logs). Rows use whichever of their runs exist; the run count is shown."""
import json, os, random, gzip, sys, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get('S35_RES', os.path.join(HERE, 'res'))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.environ.get('WSM_ROOT', os.path.join(HERE, '..', '..')), 'uniform'))
import items as U                    # noqa: E402
N_ITEMS = {'test': 1319, 'hard': 1319, 'gsmplus': 2100}
_RUNS = {}


def runs(st):
    if st not in _RUNS:
        with gzip.open(f'{RES}/fe/{st}.items.jsonl.gz', 'rt') as f:
            next(f)
            _RUNS[st] = {r['name']: r for r in map(json.loads, f)}
    return _RUNS[st]


def run(st, n):                      # res/fe first, then the uniform bundles (None if neither has it)
    r = runs(st).get(n)
    if r is None:
        try: rows = U.load(st, n)
        except FileNotFoundError: return None
        r = runs(st)[n] = {'name': n, 'ok': ''.join(str(int(bool(x['ok']))) for x in rows), 'ntok': [x['ntok'] for x in rows]}
    return r


def load(st, names):
    cols, toks = [], []
    for n in names:
        r = run(st, n)
        cols.append([float(c == '1') for c in r['ok']]); toks.append(sum(r['ntok']) / len(r['ntok']))
    return [sum(c[i] for c in cols) / len(cols) for i in range(len(cols[0]))], [100 * sum(c) / len(c) for c in cols], sum(toks) / len(toks)


def ok(st, n):
    r = run(st, n)
    return r is not None and len(r['ok']) == N_ITEMS[st]


def boot(a, b, rng):
    d = [x - y for x, y in zip(a, b)]; n = len(d)
    bs = sorted(100 * sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(2000))
    return f'{100*sum(d)/n:+.2f} [{bs[50]:+.2f}, {bs[1949]:+.2f}]'


def name(row, s, st, es):
    p = 'gp_' if st == 'gsmplus' else ''
    x = '' if es == 7 else f'_s{es}'
    return {'2': f'{p}P64_s{s}{x}', '2b': f'{p}r2_P64_s{s}{x}', '4': f'{p}r2_CW_s{s}{x}'}[row]


def block(st, seeds, eseeds, title):
    out = [f'**{title}**', '', '| row | runs | mean | per run | tok/q |', '|---|---|---|---|---|']
    V = {}
    for row in ('2', '2b', '4'):
        names = [name(row, s, st, es) for es in eseeds for s in seeds]
        have = [n for n in names if ok(st, n)]
        if not have:
            out.append(f'| {row} | 0/{len(names)} | – | | |'); continue
        v, per, tq = load(st, have); V[row] = v
        out.append(f'| {row} | {len(have)}/{len(names)} | {100*sum(v)/len(v):.2f} | {" / ".join("%.2f" % p for p in per)} | {tq:.0f} |')
    out += ['', '| effect | Δ [95% CI] |', '|---|---|']
    rng = random.Random(0)   # one rng per table, effects in ana_q32.py 't' order, then 4 − 2
    for a, b in (('2b', '2'), ('4', '2b'), ('4', '2')):
        if a in V and b in V:
            out.append(f'| {a} − {b} | {boot(V[a], V[b], rng)} |')
    return out + ['']


def per_seed():
    out = ['| set | eval seed | row | s0 | s1 | s2 | s3 | s4 | s5 | mean s0-2 | mean s3-5 | mean s0-5 |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for st in ('test', 'hard', 'gsmplus'):
        for es in (7, 11, 13):
            for row in ('2', '2b', '4'):
                vals = []
                for s in range(6):
                    n = name(row, s, st, es)
                    vals.append(load(st, [n])[1][0] if ok(st, n) else None)
                f = lambda xs: ('%.2f' % (sum(xs) / len(xs))) if xs and all(x is not None for x in xs) else '–'
                out.append(f'| {st} | {es} | {row} | ' + ' | '.join('%.2f' % v if v is not None else '–' for v in vals)
                           + f' | {f(vals[:3])} | {f(vals[3:])} | {f(vals)} |')
    return out + ['']


def dev():
    out = ['| arm | status | last dev u | dev (P64: dev_acc; CW: dev_vote4) | dev2 | CW dev_vote1 | tok |', '|---|---|---|---|---|---|---|']
    D = json.load(open(f'{RES}/dev.json'))
    for a in [f'P64_s{s}' for s in range(6)] + [f'CW_s{s}' for s in range(6)]:
        st = 'DONE' if a in D and D[a]['done'] else 'running/queued'
        last = D[a]['dev'][-1] if a in D and D[a]['dev'] else None
        if last:
            if 'dev_acc' in last:
                out.append(f'| {a} | {st} | {last["u"]} | {last["dev_acc"]:.2f} | {last.get("dev2_acc", float("nan")):.2f} | | {last["dev_tok"]:.0f} |')
            else:
                out.append(f'| {a} | {st} | {last["u"]} | {last["dev_vote4"]:.2f} | {last.get("dev2_vote4", float("nan")):.2f} | {last["dev_vote1"]:.2f} | {last["dev_tok_per_attempt"]:.0f}/att |')
        else:
            out.append(f'| {a} | {st} | – | | | | |')
    return out + ['']


def audit():
    out = ['- ' + l.strip() for l in open(f'{RES}/fe_audit.txt') if l.startswith('FE audit')]
    return (out or ['- (none yet)']) + ['']


stamp = sys.argv[sys.argv.index('--stamp') + 1] if '--stamp' in sys.argv else f'{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%MZ}'
L = [f'# seeds35 auto-analysis ({stamp})', '',
     'Rows: 2 = P64 (1 sample), 2b = P64 + untrained rung 2 at eval (row-4 settings, stk_r2q), 4 = CW + rung 2. '
     'Row 4 seed-7 s0-2 = j23509 (stk_r2, §12.1); everything new on stk_r2q. Paired item bootstrap as ana_q32.py.', '',
     '## In-run dev (u300)', ''] + dev()
L += ['## Eval seed 7 (§12.1 table protocol)', '']
for st in ('test', 'hard', 'gsmplus'):
    L += block(st, range(3), (7,), f'{st}, seed 7, RL seeds 0-2')
    L += block(st, range(3, 6), (7,), f'{st}, seed 7, RL seeds 3-5 (new)')
    L += block(st, range(6), (7,), f'{st}, seed 7, RL seeds 0-5')
L += ['## Averaged over eval seeds 7/11/13', '']
for st in ('test', 'hard', 'gsmplus'):
    L += block(st, range(3), (7, 11, 13), f'{st}, eval seeds 7/11/13, RL seeds 0-2')
    L += block(st, range(6), (7, 11, 13), f'{st}, eval seeds 7/11/13, RL seeds 0-5')
L += ['## Per-run accuracy', ''] + per_seed() + ['## Writer audit (new runs)', ''] + audit()
print('\n'.join(L))
