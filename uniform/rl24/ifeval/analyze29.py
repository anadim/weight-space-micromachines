"""task [29]: IFEval-541 rows 1 (plain) and 3 (+14-rule implant) on ft_int_v2 (BBH call-FT host) vs Qwen3-0.6B-Base.
Inputs: res/u0/eval_u0.{plain,imp}.score.json[.gz] (Base) and res/ftint/eval_ftint.{plain,imp}.score.json (+ summaries).
Paired item-level bootstrap (10000) of the ft_int_v2 - Base delta per row.  python analyze29.py [--out FILE]
"""
import json, os, random, sys, gzip

O = os.environ.get('AO', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res')); B = 10000


def load(p):
    if not os.path.exists(p):
        if not os.path.exists(p + '.gz'): return None
        return {r['key']: r for r in json.load(gzip.open(p + '.gz', 'rt'))['rows']}
    return {r['key']: r for r in json.load(open(p))['rows']}


def item(rows, keys):
    ps = [float(all(rows[k]['strict'])) for k in keys]; pl = [float(all(rows[k]['loose'])) for k in keys]
    return ps, pl, [sum(rows[k]['strict']) for k in keys], [len(rows[k]['strict']) for k in keys], [float(rows[k]['ntok'] or 0) for k in keys]


def boot(f, n, seed=0):
    rng = random.Random(seed); out = sorted(f([rng.randrange(n) for _ in range(n)]) for _ in range(B))
    return out[int(.025 * B)], out[int(.975 * B) - 1]

H = {'Base': ('u0', 'eval_u0'), 'ft_int_v2': ('ftint', 'eval_ftint')}


def metrics(rows, keys):
    ps, pl, ist, ni, nt = item(rows, keys)
    il = [sum(rows[k]['loose']) for k in keys]
    n = len(keys)
    return dict(ps=100 * sum(ps) / n, pl=100 * sum(pl) / n, is_=100 * sum(ist) / sum(ni), il=100 * sum(il) / sum(ni),
                tq=sum(nt) / n, fired=sum(rows[k]['circ']['fired'] for k in keys), vec=ps)


L = ['| host | row | prompt-strict | prompt-loose | inst-strict | inst-loose | tok/q | fired | idle identical |', '|' + '---|' * 9]
res = {}
for h, (d, p) in H.items():
    sm = os.path.join(O, d, p + '.summary.json')
    idle = json.load(open(sm)).get('idle', {}) if os.path.exists(sm) else {}
    for m, r in (('plain', '1 plain'), ('imp', '3 + implant')):
        rows = load(os.path.join(O, d, f'{p}.{m}.score.json'))
        if rows is None: L.append(f'| {h} | {r} | (missing) |||||||'); continue
        keys = sorted(rows); x = metrics(rows, keys); res[h, m] = (x, rows)
        idl = f"{idle.get('identical_in_full_runs')}/{idle.get('n_idle')}" if m == 'imp' and idle else ''
        L.append(f"| {h} | {r} | {x['ps']:.2f} | {x['pl']:.2f} | {x['is_']:.2f} | {x['il']:.2f} | {x['tq']:.0f} | "
                 f"{x['fired'] if m == 'imp' else ''} | {idl} |")
L += ['', '| row | delta prompt-strict (ft_int_v2 - Base) | 95% paired CI | > 2 points? |', '|---|---|---|---|']
for m, r in (('plain', '1 plain'), ('imp', '3 + implant')):
    if ('Base', m) not in res or ('ft_int_v2', m) not in res: continue
    (a, _), (b, _) = res['Base', m], res['ft_int_v2', m]
    d = [y - x for x, y in zip(a['vec'], b['vec'])]; n = len(d)
    lo, hi = boot(lambda idx: 100 * sum(d[i] for i in idx) / n, n)
    D = 100 * sum(d) / n
    L.append(f"| {r} | {D:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {'yes' if abs(D) > 2 else 'no'} |")
s = '\n'.join(L); print(s)
if '--out' in sys.argv: open(sys.argv[sys.argv.index('--out') + 1], 'w').write(s + '\n')
