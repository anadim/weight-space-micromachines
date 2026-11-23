"""[38]/[41] C36 analysis: arm T (text <calc> tool, model-token cap + 128 tool tokens) vs arm I (exact calculator implant on
the same <calc> call path) and its controls I0 (frozen hand writer, no in-wire/adapters) and Itok (writer only), u300,
8 train seeds each at each arm's [41]-selected config. Reads tt/res/items.jsonl.gz (per-item correctness + per-run sums and
summary.json of every eval dir, built from the fe_tool.py outputs by mk_res.py; override with C36_RES=DIR).
Accuracy = mean over eval seeds 7/11/13 (and train seeds); CI = item bootstrap of the per-item mean; differences paired by
item, plus a two-level (train seed x item) bootstrap. Writes calcimp/rows38.json with --write (rows38_all.json with --all).
python3 ana38.py [--all] [--write] [--sens]   (--sens adds Ix/Ie0x = I/Ie0 without the collapsed seeds; written to rows38*_sens.json)"""
import gzip, json, os, random, sys
RES = os.environ.get('C36_RES', os.path.dirname(os.path.abspath(__file__)) + '/tt/res')
EV = {r['ev']: r for r in map(json.loads, gzip.open(f'{RES}/items.jsonl.gz', 'rt'))}
SEEDS = [7, 11, 13]
SETS = [('test', 'GSM8K test (1319)'), ('hard', 'GSM-Hard (1319)'), ('gsmplus', 'GSM-Plus testmini (2100)')]
CFG = json.load(open(os.path.dirname(os.path.abspath(__file__)) + '/c36_cfg.json'))   # eval-name patterns + chosen configs
ALL = "--all" in sys.argv   # each row on all of its own finished train seeds (0-39 scanned)
K = [k for k in range(8) if all(CFG[a] % k in EV for a in ('T', 'I', 'I0', 'Itok'))]
ROWS = [('0X', 'Base, <calc> prompt + text tool (no RL; cap 320 model + 128 tool)', ['eb_baseX']),
        ('0I0', 'Base, <calc> prompt + implant I0 (no RL)', ['eb_baseI0'])]
for a, lab in (('T', 'arm T (text tool)'), ('I', 'arm I (implant: writer + adapters 7/14/21/final + in-wire)'),
               ('I0', 'control I0 (frozen hand writer)'), ('Itok', 'control Itok (writer only trainable)'),
               ('T1', 'arm T at the [35] host lr'), ('I1', 'arm I at the [35] host lr'),
               ('Ie0', 'arm I, in-wire floor off at eval'), ('I01', 'control I0 at the [35] host lr'),
               ('Itok1', 'control Itok at the [35] host lr'), ('T2', 'arm T at I\'s host lr')):
    own = [k for k in range(40 if ALL else 8) if CFG[a] % k in EV]
    ks = own if ALL or len(K) < 2 else K
    if ks: ROWS.append((a, f'{lab}, {CFG["desc_" + a]}, train seeds {"/".join(map(str, ks))}', [CFG[a] % k for k in ks]))
COLLAPSED_I = [14, 17]   # A100 fI seeds whose calculator use collapsed during RL (RL_RESULTS §12.6 "Arm-I instability")
if '--sens' in sys.argv:   # sensitivity only: arm I / Ie0 without the collapsed seeds (headline rows keep them)
    for a, src in (('Ix', 'I'), ('Ie0x', 'Ie0')):
        r = [r for r in ROWS if r[0] == src]
        if r:
            names = [n for n in r[0][2] if not any(n == CFG[src] % k for k in COLLAPSED_I)]
            ROWS.append((a, f'{src} excluding collapsed seeds {"/".join(map(str, COLLAPSED_I))} (sensitivity), {len(names)} train seeds', names))
ONLY = [x for x in os.environ.get('ANA38_ONLY', '').split(',') if x]   # quick subset run (no --write): e.g. ANA38_ONLY=I,Ix,T2
if ONLY:
    ROWS = [r for r in ROWS if r[0] in ONLY]
PAIRS = [('Ix', 'T2'), ('Ie0x', 'T2'), ('Ix', 'I0'), ('I', 'T'), ('I1', 'T1'), ('T', 'T1'), ('I', 'T1'), ('I0', 'T1'), ('Itok', 'T1'), ('I', 'I1'), ('I0', 'T'), ('Itok', 'T'), ('I', 'I0'), ('I', 'Itok'), ('Itok', 'I0'), ('Ie0', 'I'), ('Ie0', 'I0'), ('Ie0', 'Itok'), ('Ie0', 'T1'), ('I01', 'T1'), ('Itok1', 'T1'), ('I1', 'I01'), ('I0', 'I01'), ('T2', 'T1'), ('I', 'T2'), ('I0', 'T2'), ('Itok', 'T2'), ('Ie0', 'T2'), ('0I0', '0X')]
if ONLY:
    PAIRS = [p for p in PAIRS if p[0] in ONLY and p[1] in ONLY]


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


res = dict(host='Qwen3-0.6B-Base', metric='accuracy (%)', sets={}, config=CFG,
           notes='[38] equal-footing text tool vs exact implant on the <calc> call path. gsm_grpo recipe (<calc> 8-shot prefix, '
                 'Dr.GRPO, 300 u, 64 prompts x 8, T 1, max_new 320 model tokens + 128 tool/implant-written tokens), each arm at '
                 'its [41] dev-selected config (u100, dev 800). Eval: T = 1, one attempt, eval seeds 7/11/13, batched engine. '
                 'tok/q (model) excludes tool / implant-written tokens. s/q batched includes the exact core (CPU, int64); '
                 '"excl. core" subtracts the logged core seconds.')
lines = []
for st, label in SETS:
    R = {}; RS = {}; rows = []
    lines.append(f'\n### {label}\n| row | config | acc (per train seed) | 95% CI | tok/q (model) | calls/q | written/q | admit | '
                 f'ms/q batched (excl. core) | ms/q single-stream |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|')
    for rid, lab, names in ROWS:
        ds = [load(nm, st) for nm in names]
        if not ds or any(d is None for d in ds):
            lines.append(f'| {rid} | {lab} | MISSING | | | | | | | |'); continue
        runs = [d[s] for d in ds for s in SEEDS]
        n = runs[0]['n']
        per = [sum(int(r['ok'][j]) for r in runs) / len(runs) for j in range(n)]
        R[rid] = per
        RS[rid] = [[sum(int(d[sd]['ok'][j]) for sd in SEEDS) / len(SEEDS) for j in range(n)] for d in ds]
        accs = [100 * r['ok'].count('1') / n for r in runs]
        mean = sum(accs) / len(accs); ci = boot(per)
        avg = lambda k: sum(r['sum'].get(k, 0) / n for r in runs) / len(runs)
        summs = [EV[nm]['summary'] for nm in names]
        cells = [sm[f'{st}_s{s}'] for sm in summs for s in SEEDS]
        spq = sum(c['s_per_q_batched'] for c in cells) / len(cells)
        spq_x = sum((c['seconds'] - c.get('core_s', 0.0)) / c['n'] for c in cells) / len(cells)
        adm = [c['admit'] for c in cells if 'admit' in c]
        lat = sum(sm.get('latency_B1', {}).get('s_per_q', 0) for sm in summs) / len(summs) if st == 'test' else None
        row = dict(row=rid, label=lab, configs=[f'{nm}/{st}_s{s}' for nm in names for s in SEEDS], mean=round(mean, 2),
                   seeds=[round(a, 2) for a in accs], ci95=[round(ci[0], 2), round(ci[1], 2)], tokq=round(avg('ntok'), 1),
                   tokq_model=round(avg('ntok_model'), 1), calls_q=round(avg('calls'), 2), written_q=round(avg('nwritten'), 2),
                   admit=round(sum(adm) / len(adm), 4) if adm else None, s_per_q_batched=round(spq, 5),
                   s_per_q_batched_excl_core=round(spq_x, 5), fmt=round(avg('parsed') * 100, 2))
        if lat is not None: row['s_per_q_single_stream'] = round(lat, 4)
        rows.append(row)
        if len(names) > 1:
            tm = [sum(accs[i * len(SEEDS):(i + 1) * len(SEEDS)]) / len(SEEDS) for i in range(len(names))]
            row['train_seed_means'] = [round(a, 2) for a in tm]; accs = tm
        lines.append(f"| {rid} | {lab} | **{mean:.2f}** ({' / '.join(f'{a:.2f}' for a in accs)}) | [{ci[0]:.2f}, {ci[1]:.2f}] | "
                     f"{row['tokq_model']:.0f} | {row['calls_q']:.2f} | {row['written_q']:.1f} | "
                     f"{'%.3f' % row['admit'] if adm else '-'} | {1000 * spq:.1f} ({1000 * spq_x:.1f}) | "
                     f"{'%.0f ms' % (1000 * lat) if lat else ''} |")
    diffs = []
    for a, b in PAIRS:
        if a in R and b in R:
            dv = [x - y for x, y in zip(R[a], R[b])]
            m = 100 * sum(dv) / len(dv); ci = boot(dv, seed=1)
            dd = dict(pair=f'{a} - {b}', delta=round(m, 2), ci95=[round(ci[0], 2), round(ci[1], 2)])
            txt = f'  {a} − {b}: {m:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] item CI'
            if len(RS[a]) > 1 and len(RS[b]) > 1:
                hc = hboot(RS[a], RS[b]); dd['ci95_seeds_items'] = [round(hc[0], 2), round(hc[1], 2)]
                txt += f'; train-seed x item CI [{hc[0]:+.2f}, {hc[1]:+.2f}]'
            diffs.append(dd); lines.append(txt)
    res['sets'][st] = dict(label=label, rows=rows, diffs=diffs)
print('\n'.join(lines))
if '--write' in sys.argv and not ONLY:
    fn = ('rows38_all' if ALL else 'rows38') + ('_sens' if '--sens' in sys.argv else '') + '.json'
    json.dump(res, open(os.path.dirname(os.path.abspath(__file__)) + '/' + fn, 'w'), indent=1); print('wrote', fn)
