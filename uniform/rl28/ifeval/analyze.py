"""IFEval 4-row table of task [28] (Qwen3-1.7B-Base; copy of the rl24 [26] analysis) next to the 0.6B rows of
../../rl24/ifeval/rows.json. (pure python, reads res/ only; AO=out for fresh runs).
Rows: 1 plain host (res/u0/eval_u0.plain), 2 host+RL plain (res/P_s*/eval_u300.plain), 3 host+implant (res/u0/eval_u0.imp),
4 host+implant+RL (res/I_s*/eval_u300.imp); secondary: P_s* evaluated with the implant attached (eval_u300.imp).
Per row: IFEval-541 prompt-level strict (and loose), instruction-level strict, tok/q (mean generated tokens, stop excluded),
95% paired item-level bootstrap CIs (10000 resamples of the 541 prompts; RL rows average the per-item indicator over seeds,
so the CI is over prompts with seeds pooled), plus per-seed values and the paired delta vs row 1 and row 4 vs row 3.
python analyze.py [--out RESULTS_TABLE.md]
"""
import json, os, random, sys, glob, gzip

HERE = os.path.dirname(os.path.abspath(__file__)); O = os.environ.get('AO', os.path.join(HERE, 'res'))
B = 10000
SUF = os.environ.get('SUF', '')                     # RL variant suffix, e.g. SUF=_lr2e-6 -> P_s0_lr2e-6


def load(p):
    if not os.path.exists(p):
        if not os.path.exists(p + '.gz'): return None
        d = json.load(gzip.open(p + '.gz', 'rt')); return {r['key']: r for r in d['rows']}
    d = json.load(open(p)); return {r['key']: r for r in d['rows']}


def item(rows, keys):
    ps = [float(all(rows[k]['strict'])) for k in keys]; pl = [float(all(rows[k]['loose'])) for k in keys]
    ist = [sum(rows[k]['strict']) for k in keys]; ni = [len(rows[k]['strict']) for k in keys]
    nt = [float(rows[k]['ntok'] or 0) for k in keys]
    return ps, pl, ist, ni, nt


def avg(vs): return [sum(x) / len(x) for x in zip(*vs)]


def boot(f, n, seed=0):
    rng = random.Random(seed); out = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]; out.append(f(idx))
    out.sort(); return out[int(.025 * B)], out[int(.975 * B) - 1]


ROWS = [('1 plain host', [('u0', 'eval_u0.plain')]),
        ('2 host + RL (no implant)', [(f'P_s{s}{SUF}', 'eval_u300.plain') for s in range(3)]),
        ('3 host + 14-rule implant', [('u0', 'eval_u0.imp')]),
        ('4 host + implant + RL (implant attached)', [(f'I_s{s}{SUF}', 'eval_u300.imp') for s in range(3)]),
        ("2b RL'd plain host + implant attached at eval", [(f'P_s{s}{SUF}', 'eval_u300.imp') for s in range(3)]),
        ("4p implant-RL'd host, implant detached at eval", [(f'I_s{s}{SUF}', 'eval_u300.plain') for s in range(3)])]


RID = {'1 plain host': ('1', 'host'), '2 host + RL (no implant)': ('2', 'host + RL'), '3 host + 14-rule implant': ('3', 'host + implants'),
       '4 host + implant + RL (implant attached)': ('4', 'host + implants + RL'), "2b RL'd plain host + implant attached at eval": ('2b', 'host + RL, implants attached at eval only'),
       "4p implant-RL'd host, implant detached at eval": ('4p', 'host + implants + RL, implants detached at eval')}


def cfg(d, f): return d + ('_imp' if f.endswith('.imp') else '_plain') if d == 'u0' else d


CHOSEN = open(os.path.join(O, 'setup', 'CHOSEN')).read().strip() if os.path.exists(os.path.join(O, 'setup', 'CHOSEN')) else '?'
_r06 = json.load(open(os.path.join(HERE, '..', '..', 'rl24', 'ifeval', 'rows.json')))['sets']
R06 = {r['row']: r for k in ('ifeval541', 'ifeval541_2b') for r in _r06[k]['rows']}


def row06_4p():
    # 0.6B row 4p from the rl24 [26] I-arm u300 evals without the implant (read only; not in rl24 rows.json)
    O6 = os.path.join(HERE, '..', '..', 'rl24', 'ifeval', 'res')
    runs = [load(os.path.join(O6, f'I_s{s}', 'eval_u300.plain.score.json')) for s in range(3)]
    runs = [r for r in runs if r]
    if not runs: return None
    ks = sorted(runs[0]); n = len(ks); per = [item(r, ks) for r in runs]; ps = avg([p[0] for p in per]); nt = avg([p[4] for p in per])
    lo, hi = boot(lambda idx: sum(ps[i] for i in idx) / len(idx), n)
    return dict(row='4p', label='host + implants + RL, implants detached at eval', configs=[f'I_s{s}' for s in range(len(runs))],
                mean=round(100 * sum(ps) / n, 2), seeds=[round(100 * sum(p[0]) / n, 2) for p in per], ci95=[round(100 * lo, 2), round(100 * hi, 2)],
                tokq=round(sum(nt) / n, 1))


R06['4p'] = row06_4p()


def main():
    res = {}; keys = None; lines = []; J = {'ps': [], 'is': []}
    for name, srcs in ROWS:
        runs = [(d, load(os.path.join(O, d, f + '.score.json'))) for d, f in srcs]
        runs = [(d, r) for d, r in runs if r]
        if not runs: lines.append(f'| {name} | (missing) | | | | |'); continue
        keys = keys or sorted(runs[0][1]); ks = keys
        per = [item(r, ks) for _, r in runs]
        ps = avg([p[0] for p in per]); pl = avg([p[1] for p in per]); ist = avg([p[2] for p in per]); ni = per[0][3]
        nt = avg([p[4] for p in per]); n = len(ks)
        m = lambda v, idx=None: sum(v[i] for i in (idx or range(n))) / len(idx or range(n))
        inst = lambda idx: sum(ist[i] for i in idx) / sum(ni[i] for i in idx)
        res[name] = dict(ps=ps, ist=ist, ni=ni)
        lo, hi = boot(lambda idx: m(ps, idx), n)
        ilo, ihi = boot(inst, n, 1)
        seeds = ' / '.join(f'{100 * sum(p[0]) / n:.2f}' for p in per)
        rid, lab = RID[name]; cf = [cfg(d, f) for (d, f) in srcs if load(os.path.join(O, d, f + '.score.json'))]
        J['ps'].append(dict(row=rid, label=lab, configs=cf, mean=round(100 * m(ps), 2), seeds=[round(100 * sum(p[0]) / n, 2) for p in per],
                            ci95=[round(100 * lo, 2), round(100 * hi, 2)], tokq=round(m(nt), 1)))
        J['is'].append(dict(row=rid, label=lab, configs=cf, mean=round(100 * inst(range(n)), 2),
                            seeds=[round(100 * sum(p[2]) / sum(p[3]), 2) for p in per], ci95=[round(100 * ilo, 2), round(100 * ihi, 2)],
                            tokq=round(m(nt), 1)))
        lines.append(f'| {name} | {100 * m(ps):.2f} [{100 * lo:.2f}, {100 * hi:.2f}] | {100 * m(pl):.2f} | '
                     f'{100 * inst(range(n)):.2f} [{100 * ilo:.2f}, {100 * ihi:.2f}] | {m(nt):.0f} | {seeds} ({len(runs)}) |')
    hdr = ['| row | prompt strict % [95% CI] | prompt loose % | inst strict % [95% CI] | tok/q | per seed prompt strict (n) |',
           '|---|---|---|---|---|---|']
    out = ['## IFEval 541 (%s prompts)' % (len(keys) if keys else '?'), *hdr, *lines, '', '### paired deltas (prompt strict, pp, 95% item bootstrap CI)']
    pairs = [('2 host + RL (no implant)', '1 plain host'), ('3 host + 14-rule implant', '1 plain host'),
             ('4 host + implant + RL (implant attached)', '1 plain host'),
             ('4 host + implant + RL (implant attached)', '3 host + 14-rule implant'),
             ('4 host + implant + RL (implant attached)', '2 host + RL (no implant)'),
             ("2b RL'd plain host + implant attached at eval", '2 host + RL (no implant)'),
             ("2b RL'd plain host + implant attached at eval", '4 host + implant + RL (implant attached)'),
             ("4p implant-RL'd host, implant detached at eval", '2 host + RL (no implant)'),
             ('4 host + implant + RL (implant attached)', "4p implant-RL'd host, implant detached at eval")]
    for a, b in pairs:
        if a in res and b in res:
            x, y = res[a]['ps'], res[b]['ps']; n = len(x)
            d = lambda idx: sum(x[i] - y[i] for i in idx) / len(idx)
            lo, hi = boot(d, n, 2)
            out.append(f'- {a}  minus  {b}: {100 * d(range(n)):+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]')
    out += ['', '## 0.6B next to 1.7B (IFEval 541 prompt strict %, mean over seeds [95% item-bootstrap CI]; per seed)', '',
            '| row | Qwen3-0.6B-Base | Qwen3-1.7B-Base |', '|---|---|---|']
    r17 = {r['row']: r for r in J['ps']}
    for rid in ['1', '2', '3', '4', '2b', '4p']:
        c = []
        for r in (R06.get(rid), r17.get(rid)):
            if not r: c.append('(missing)'); continue
            c.append(f"{r['mean']:.2f}" + (f" [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]" if len(r['seeds']) > 1 else '') +
                     (f" ({' / '.join(f'{x:.2f}' for x in r['seeds'])})" if len(r['seeds']) > 1 else '') + f", tok/q {r['tokq']:.0f}")
        out.append(f"| {rid} | {c[0]} | {c[1]} |")
    txt = '\n'.join(out); print(txt)
    if '--out' in sys.argv:
        open(sys.argv[sys.argv.index('--out') + 1], 'w').write(txt + '\n')
        main4 = lambda L: [r for r in L if r['row'] not in ('2b', '4p')]
        rows = {'metric': 'prompt-strict acc (%)',
                'sets': {'ifeval541': {'label': 'IFEval 541 prompt-strict', 'rows': main4(J['ps'])},
                         'ifeval541_inst': {'label': 'IFEval 541 instruction-level strict', 'rows': main4(J['is'])},
                         'ifeval541_2b': {'label': "IFEval 541 prompt-strict, secondary: row-2 hosts with the implant attached at eval only",
                                          'rows': [r for r in J['ps'] if r['row'] == '2b']},
                         'ifeval541_4p': {'label': "IFEval 541 prompt-strict, secondary: row-4 hosts (trained with the implant) evaluated without it",
                                          'rows': [r for r in J['ps'] if r['row'] == '4p']}},
                'notes': ('host Qwen3-1.7B-Base; format = Qwen3 chat template with thinking off (empty think block), same for all rows; greedy, '
                          'max_new 2048; seeded lm_eval IFEval scorer; implant = the same full14 compiled 14-rule circuit, host readers/writers refit to 1.7B '
                          '(force_fix re-solved on Qwen3-1.7B; readout export %s chosen on dev); rows 2/4: GRPO (Dr.GRPO) 300 u, 64x8, lr 1e-6, T 1, kl 0, train max_new 1024, '
                          'fresh gen_data prompts (seed 200, filtered vs the 541), reward = lm_eval prompt-strict; row 4 trained with the implant attached. '
                          'ci95 = item bootstrap (10000) of the seed-averaged per-item score. tokq = mean generated tokens per prompt (stop excluded). '
                          'Rows 2/4 are trained under the same format.' % CHOSEN + (f' SUF={SUF}' if SUF else '')),
                'host_0.6B': R06}
        json.dump(rows, open(os.path.join(HERE, f'rows{SUF}.json'), 'w'), indent=1)


main()
