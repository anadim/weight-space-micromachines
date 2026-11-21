"""within-item AUROC of side-net scores and lg blends (weights chosen on trainC OOF) - local, from job/out/sc_TAG.npz"""
import sys, gzip, json, numpy as np
sc = dict(np.load(sys.argv[1])); cfgs = sorted({k.split('|')[0] for k in sc})
T = {}
for S in ('trainC', 'dev', 'dev2'):
    rows = [json.loads(l) for l in gzip.open(f'tables/attempts_{S}.jsonl.gz', 'rt')]
    T[S] = dict(idx=np.array([r['idx'] for r in rows]), ok=np.array([r['ok'] for r in rows], bool),
                lg=np.array([-30 if r['lg'] is None else r['lg'] for r in rows], float),
                cm=np.array([r['hash_off'] is not None and r['lg'] is not None and r['lg'] > -29.5 for r in rows]))
def within(S, s):
    t = T[S]; num = den = 0
    m0 = t['cm']; idx = t['idx'][m0]; y = t['ok'][m0]; s = s[m0]
    o = np.argsort(idx, kind='stable'); idx, y, s = idx[o], y[o], s[o]
    cut = np.flatnonzero(np.diff(idx)) + 1
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(idx)]):
        yy = y[a:b]; ss = s[a:b]
        if yy.all() or (~yy).all(): continue
        d = ss[yy][:, None] - ss[~yy][None]; num += (d > 0).sum() + .5 * (d == 0).sum(); den += d.size
    return num / den
z = lambda S, v: (v - v[T[S]['cm']].mean()) / v[T[S]['cm']].std()
print('lg', {S: round(within(S, T[S]['lg']), 4) for S in T})
for c in cfgs:
    best = max(np.linspace(0, 1, 11), key=lambda w: within('trainC', w * z('trainC', sc[f'{c}|trainC']) + (1 - w) * z('trainC', np.maximum(T['trainC']['lg'], -12))))
    r = {S: within(S, sc[f'{c}|{S}']) for S in T}
    rb = {S: within(S, best * z(S, sc[f'{c}|{S}']) + (1 - best) * z(S, np.maximum(T[S]['lg'], -12))) for S in T}
    r5 = {S: within(S, .5 * z(S, sc[f'{c}|{S}']) + .5 * z(S, np.maximum(T[S]['lg'], -12))) for S in T}
    print(f'{c:22s} sn ' + ' '.join(f'{S} {r[S]:.4f}' for S in T) + f' | blend w={best:.1f} ' + ' '.join(f'{S} {rb[S]:.4f}' for S in T)
          + ' | w=.5 ' + ' '.join(f'{S} {r5[S]:.4f}' for S in T))
