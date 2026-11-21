"""M5 step 3: compare implant runs vs base rows (same seeds). usage: python3 ana2.py BASEDIR RUNDIR1 [RUNDIR2 ...] [--seeds=7,11]
each DIR is a path prefix; rows are {DIR}_{set}/{set}_N8_*seed{seed}.jsonl.gz (e.g. gsm_stack/implants/aw/dev_tree/out/r1f)"""
import sys, glob, collections
sys.path.insert(0, '.')
import m5lib as L

args = [a for a in sys.argv[1:] if not a.startswith('--')]
seeds = [int(x) for x in (next((a.split('=')[1] for a in sys.argv if a.startswith('--seeds=')), '7,11')).split(',')]
BASE, RUNS = args[0], args[1:]


def load(run, st, s):
    c = sorted(glob.glob(f'{run}_{st}/{st}_N8_*seed{s}.jsonl.gz'))
    if len(c) != 1: return None
    rows = L.load_rows(c[0])
    return [(r, L.attempts_of_row(r, 'gen')[0]) for r in rows]


def stats(D):
    n = len(D); v = sum(r['ok'] for r, _ in D) / n
    o = sum(any(a['right'] for a in at) for _, at in D) / n
    pa = sum(sum(a['right'] for a in at) / max(1, len(at)) for _, at in D) / n
    byk = collections.defaultdict(list)
    for _, at in D:
        for a in at: byk[a['k']].append(a['right'])
    dl = sum(len({L.TK.decode(a['fl']).strip() for a in at}) for _, at in D) / n
    da = sum(len({a['val'] for a in at if a['ok']}) for _, at in D) / n
    tok = sum(len(r['tokens']) for r, _ in D) / n
    return dict(vote=100 * v, orc=100 * o, pa=100 * pa, dl=dl, da=da, tok=tok, k=[100 * sum(byk[k]) / len(byk[k]) for k in range(8)])


print('| run | set | vote | oracle@8 | per-att | acc k=0..7 | distinct L1/8 | distinct ans | tok/q | fixes | breaks | net | n seeds |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for st in ['dev', 'dev2']:
    for run in [BASE] + RUNS:
        S = []; fx = br = 0; ns = 0
        for s in seeds:
            D = load(run, st, s); B = load(BASE, st, s)
            if D is None or B is None: continue
            ns += 1; S.append(stats(D))
            for (r, _), (b, _) in zip(D, B):
                assert r.get('q') == b.get('q') and r['gold'] == b['gold']
                fx += r['ok'] and not b['ok']; br += b['ok'] and not r['ok']
        if not S: continue
        m = lambda k: sum(x[k] for x in S) / len(S)
        kk = ' '.join(f'{sum(x["k"][i] for x in S) / len(S):.0f}' for i in range(8))
        print(f'| {run.rsplit("/",1)[-1]} | {st} | {m("vote"):.2f} | {m("orc"):.2f} | {m("pa"):.2f} | {kk} | {m("dl"):.2f} | {m("da"):.2f} | {m("tok"):.0f} | {fx} | {br} | {fx - br:+d} | {ns} |')
