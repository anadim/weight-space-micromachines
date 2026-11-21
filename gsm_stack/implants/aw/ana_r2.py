"""ladder-r2 analysis: per-seed acc / tok/q / natt / rep/q of aw-format runs, 4-seed means, and paired fixes/breaks.
python3 implants/aw/ana_r2.py"""
import glob, gzip, json, os
A = os.path.expanduser('gsm_stack/implants/aw/dev_tree/out')
FE = os.path.expanduser('gsm_stack/implants/wpr/fe/out/fe')
CO = os.path.expanduser('gsm_stack/implants/aw/confirm_old/out')
SEEDS = [7, 11, 13, 17]


def load(path):
    op = gzip.open if path.endswith('.gz') else open
    return [json.loads(l) for l in op(path, 'rt')]


def aw(d, s, seed, base=A):
    f = glob.glob(f'{base}/{d}/{s}_*_seed{seed}.jsonl.gz')
    return load(f[0]) if f else None


def fe(s, seed):
    p = f'{FE}/{s}_wpr_s{seed}.jsonl'
    return load(p) if os.path.exists(p) else None


def stat(r):
    n = len(r); g = lambda k: sum(x.get(k, 0) for x in r) / n
    return 100 * g('ok'), g('ntok'), g('natt'), g('nrep')


def fb(a, b):
    """fixes / breaks of b vs a (paired by item order)"""
    return sum(1 for x, y in zip(a, b) if y['ok'] and not x['ok']), sum(1 for x, y in zip(a, b) if x['ok'] and not y['ok'])


RUNS = [('r2 old (1 proc)', lambda s, k: aw('r2_' + s, s, k)), ('r2 fixed (1 proc)', lambda s, k: aw('r2f_' + s, s, k)),
        ('r2 fixed s11 fresh', lambda s, k: aw('r2f1_' + s, s, k)), ('r1 fixed (1 proc)', lambda s, k: aw('r1f_' + s, s, k)),
        ('r1 fe (proc/seed, old)', fe), ('r1 aw atl (proc/seed, old)', lambda s, k: aw(f'r1atl_{s}_s{k}', s, k) or (aw('r1atl', s, k) if k == 7 and s == 'dev' else None)),
        ('r0 base', lambda s, k: aw('base', s, k)),
        ('old r2 s11 fresh', lambda s, k: aw('c_r2s11', s, k, CO)), ('old r1 s7,11 1proc', lambda s, k: aw('c_w711', s, k, CO))]
for s in ('dev', 'dev2'):
    print(f'== {s}')
    R = {}
    for nm, fn in RUNS:
        rs = {k: fn(s, k) for k in SEEDS}; rs = {k: v for k, v in rs.items() if v}
        if not rs: continue
        R[nm] = rs
        st = {k: stat(v) for k, v in rs.items()}
        per = ' '.join(f's{k} {a:.2f} (tok {t:.0f} natt {na:.2f} rep {rp:.2f})' for k, (a, t, na, rp) in st.items())
        m = [sum(x[i] for x in st.values()) / len(st) for i in range(4)]
        print(f'{nm:28s} mean({len(st)}) {m[0]:.2f} tok {m[1]:.0f} natt {m[2]:.2f} rep {m[3]:.2f} | {per}')
    for a, b in [('r1 fixed (1 proc)', 'r2 fixed (1 proc)'), ('r0 base', 'r1 fixed (1 proc)'), ('r1 fe (proc/seed, old)', 'r1 fixed (1 proc)'),
                 ('r2 old (1 proc)', 'r2 fixed (1 proc)'), ('r1 aw atl (proc/seed, old)', 'r1 fixed (1 proc)')]:
        if a in R and b in R:
            ks = [k for k in SEEDS if k in R[a] and k in R[b]]
            x = [fb(R[a][k], R[b][k]) for k in ks]
            print(f'  {b} vs {a}: ' + ' '.join(f's{k} +{f}/-{bb}' for k, (f, bb) in zip(ks, x)) + f' | total +{sum(f for f, _ in x)}/-{sum(b_ for _, b_ in x)}')
