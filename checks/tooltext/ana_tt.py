"""[35] analysis: arm 1 (plain RL), arm 2 (RL + calculator implant, 1 attempt), arm X (RL + text <calc> tool); u300, seed 0.
Reads res/items.jsonl.gz (per-item correctness + per-run sums and summary.json of every eval dir, built from the fe_tool.py
outputs by ../calcimp/mk_res.py; override with TT_RES=DIR) and res/gsmplus_ptype.json. Accuracy = mean over eval seeds 7/11/13; 95% CI = item
bootstrap (2000) of the per-item seed-mean; differences paired by item. Writes rows.json with --write.  python3 ana_tt.py [--write]"""
import gzip, json, os, random, sys
H = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get('TT_RES', H + '/res')
EV = {r['ev']: r for r in map(json.loads, gzip.open(f'{RES}/items.jsonl.gz', 'rt'))}
SEEDS = [7, 11, 13]
SETS = [('test', 'GSM8K test (1319)'), ('hard', 'GSM-Hard (1319)'), ('gsmplus', 'GSM-Plus testmini (2100)')]
ROWS = [('0', 'Base, plain 8-shot prompt (no RL)', ['ev_base1']), ('02', 'Base + calculator implant (no RL)', ['ev_base2']),
        ('0X', 'Base, <calc> 8-shot prompt + text tool (no RL)', ['ev_baseX']),
        ('1', 'plain RL (arm 1), train seed 0', ['ev_a1s0']), ('2', 'RL + calculator implant, 1 attempt (arm 2), train seed 0', ['ev_a2s0']),
        ('X', 'RL + text calculator tool (arm X), train seed 0', ['ev_Xs0']),
        ('1·k', 'arm 1, train seeds pooled', ['ev_a1s%d']), ('2·k', 'arm 2, train seeds pooled', ['ev_a2s%d']),
        ('X·k', 'arm X, train seeds pooled', ['ev_Xs%d'])]
# pooled rows: the train seeds whose evals exist for all three arms
KS = [k for k in range(32) if all(p % k in EV for p in ('ev_a1s%d', 'ev_a2s%d', 'ev_Xs%d'))]
ROWS = [(r, l + f' ({"/".join(map(str, KS))})', [n[0] % k for k in KS]) if '%d' in n[0] else (r, l, n) for r, l, n in ROWS
        if '%d' not in n[0] or len(KS) >= 2]
# [37](a): arm X u300 re-evaluated with the cap on model tokens only (cap_extra 128; e37a_Xs*, run from the ../calcimp/tt code), train seeds 0-4,
# next to the same 5 seeds under the old total-token cap and arm 2
K5 = [0, 1, 2, 3, 4]
ROWS += [('X·5', 'arm X, train seeds 0-4, cap on all tokens (as [35])', [f'ev_Xs{k}' for k in K5]),
         ('Xm·5', '[37](a) arm X, train seeds 0-4, cap on model tokens only (+128 tool tokens)', [f'e37a_Xs{k}' for k in K5]),
         ('2·5', 'arm 2, train seeds 0-4', [f'ev_a2s{k}' for k in K5])]
PAIRS = [('X', '2'), ('X', '1'), ('2', '1'), ('0X', '0'), ('02', '0'), ('0X', '02'), ('X·k', '2·k'), ('X·k', '1·k'), ('2·k', '1·k'), ('Xm·5', 'X·5'), ('Xm·5', '2·5')]


def load(name, st):
    out = {}
    for s in SEEDS:
        r = EV.get(name, {}).get('sets', {}).get(f'{st}_s{s}')
        if r is None: return None
        out[s] = r
    return out


def boot(vals, B=2000, seed=0):
    rng = random.Random(seed); n = len(vals)
    ms = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return 100 * ms[int(0.025 * B)], 100 * ms[int(0.975 * B) - 1]


def hboot(A, B, nb=2000, seed=2):
    """two-level bootstrap of mean(A) - mean(B): resample train seeds (independently per arm), then items (shared)."""
    rng = random.Random(seed); n = len(A[0]); ms = []
    for _ in range(nb):
        sa = [A[rng.randrange(len(A))] for _ in A]; sb = [B[rng.randrange(len(B))] for _ in B]
        ma = [sum(c) / len(c) for c in zip(*sa)]; mb = [sum(c) / len(c) for c in zip(*sb)]
        idx = [rng.randrange(n) for _ in range(n)]
        ms.append(100 * sum(ma[i] - mb[i] for i in idx) / n)
    ms.sort(); return ms[int(0.025 * nb)], ms[int(0.975 * nb) - 1]


res = dict(host='Qwen3-0.6B-Base', metric='accuracy (%)', sets={},
           notes='[35] text-tool baseline. gsm_grpo recipe (8-shot prefix, Dr.GRPO, 300 u, 64 prompts x 8, lr 1e-6, T 1, '
                 'max_new 320, seed 0), all three arms trained with the same code, one B200 per job. Eval: T = 1, one attempt, eval seeds '
                 '7/11/13, max_new 320, batched engine (B_max 512). Tool tokens count toward max_new and toward tok/q; '
                 'tok/q (model) excludes them. CI: item bootstrap of the per-item seed mean; pooled rows also give a two-level (train seed x item) bootstrap CI for differences.')
lines = []
for st, label in SETS:
    R = {}; RS = {}; rows = []
    lines.append(f'\n### {label}\n| row | config | acc (eval seeds 7/11/13; ·k rows: per train seed) | 95% CI | tok/q (model) | calls/q | s/q batched | s/q single-stream |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for rid, lab, names in ROWS:
        ds = [load(nm, st) for nm in names]
        if not ds or any(d is None for d in ds):
            lines.append(f'| {rid} | {lab} | MISSING | | | | | |'); continue
        runs = [d[s] for d in ds for s in SEEDS]           # (train seed, eval seed) runs
        n = runs[0]['n']
        per = [sum(int(r['ok'][j]) for r in runs) / len(runs) for j in range(n)]
        R[rid] = per
        RS[rid] = [[sum(int(d[sd]['ok'][j]) for sd in SEEDS) / len(SEEDS) for j in range(n)] for d in ds]
        accs = [100 * r['ok'].count('1') / n for r in runs]
        mean = sum(accs) / len(accs); ci = boot(per)
        avg = lambda k: sum(r['sum'][k] / n for r in runs) / len(runs)
        summs = [EV[nm]['summary'] for nm in names]
        spq = sum(sm[f'{st}_s{s}']['s_per_q_batched'] for sm in summs for s in SEEDS) / len(runs)
        lat = sum(sm.get('latency_B1', {}).get('s_per_q', 0) for sm in summs) / len(summs) if st == 'test' else None
        row = dict(row=rid, label=lab, configs=[f'{nm}/{st}_s{s}' for nm in names for s in SEEDS], mean=round(mean, 2),
                   seeds=[round(a, 2) for a in accs], ci95=[round(ci[0], 2), round(ci[1], 2)], tokq=round(avg('ntok'), 1),
                   tokq_model=round(avg('ntok_model'), 1), calls_q=round(avg('calls'), 2), s_per_q_batched=round(spq, 5),
                   fmt=round(avg('parsed') * 100, 2))
        if lat is not None: row['s_per_q_single_stream'] = round(lat, 4)
        rows.append(row)
        if len(names) > 1:                                  # per-train-seed means (eval seeds averaged)
            tm = [sum(accs[i * len(SEEDS):(i + 1) * len(SEEDS)]) / len(SEEDS) for i in range(len(names))]
            row['train_seed_means'] = [round(a, 2) for a in tm]; accs = tm
        lines.append(f"| {rid} | {lab} | **{mean:.2f}** ({' / '.join(f'{a:.2f}' for a in accs)}) | [{ci[0]:.2f}, {ci[1]:.2f}] | "
                     f"{row['tokq']:.0f} ({row['tokq_model']:.0f}) | {row['calls_q']:.2f} | {1000 * spq:.1f} ms | "
                     f"{'%.0f ms' % (1000 * lat) if lat else ''} |")
    diffs = []
    for a, b in PAIRS:
        if a in R and b in R:
            dv = [x - y for x, y in zip(R[a], R[b])]
            m = 100 * sum(dv) / len(dv); ci = boot(dv, seed=1)
            dd = dict(pair=f'{a} - {b}', delta=round(m, 2), ci95=[round(ci[0], 2), round(ci[1], 2)])
            txt = f'  {a} − {b}: {m:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]'
            if len(RS[a]) > 1:
                hc = hboot(RS[a], RS[b]); dd['ci95_seeds_items'] = [round(hc[0], 2), round(hc[1], 2)]
                txt += f' item CI; train-seed x item CI [{hc[0]:+.2f}, {hc[1]:+.2f}]'
            diffs.append(dd); lines.append(txt)
    res['sets'][st] = dict(label=label, rows=rows, diffs=diffs)
# GSM-Plus by perturbation type (fe_tool keeps every testmini item: 2100 rows in file order)
_pt = json.load(open(f'{H}/res/gsmplus_ptype.json')); PT = [_pt['types'][int(c)] for c in _pt['idx']]
def armacc(names):
    """per-item accuracy averaged over eval seeds and the given train-seed runs"""
    per = [load(n, 'gsmplus') for n in names]; n = per[0][7]['n']
    return [sum(int(r[s]['ok'][i]) for r in per for s in SEEDS) / (len(per) * len(SEEDS)) for i in range(n)]
ptab = {}
for tag, ks in (('seed 0', [0]), (f'train seeds {"/".join(map(str, KS))}', KS)):
    A = {a: armacc([f'{p}s{k}' for k in ks]) for a, p in (('1', 'ev_a1'), ('2', 'ev_a2'), ('X', 'ev_X'))}
    assert len(A['X']) == len(PT)
    lines.append(f'\nGSM-Plus by perturbation type ({tag}): n | arm 1 | arm 2 | arm X | X - 2 [item CI]')
    for t in sorted(set(PT)) + ['ALL']:
        idx = [i for i in range(len(PT)) if t in ('ALL', PT[i])]
        m = {a: 100 * sum(A[a][i] for i in idx) / len(idx) for a in A}
        ci = boot([A['X'][i] - A['2'][i] for i in idx])
        lines.append(f'  {t:28s} {len(idx):5d} | {m["1"]:6.2f} | {m["2"]:6.2f} | {m["X"]:6.2f} | {m["X"]-m["2"]:+6.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]')
        ptab.setdefault(tag, []).append(dict(ptype=t, n=len(idx), **{f'arm{a}': round(m[a], 2) for a in A}, x_minus_2=round(m['X'] - m['2'], 2), ci95=[round(ci[0], 2), round(ci[1], 2)]))
res['gsmplus_by_ptype'] = ptab
print('\n'.join(lines))
if '--write' in sys.argv:
    json.dump(res, open(f'{H}/rows.json', 'w'), indent=1); print('wrote rows.json')
