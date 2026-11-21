"""lane seqver: offline within-item AUROC of seq-verifier scores vs b = max(sn,-12)+max(lg,-12), blends chosen on trainC OOF,
and export of per-attempt deltas for repick_sq.py.  python3.11 off/ana.py job/out/sc_TAG.npz [more.npz ...] > off/ana_TAG.log
inc cfgs (mode inc): score file holds delta (s - b) -> full = b + delta; also v-scaled (v chosen on trainC OOF).
alone cfgs: blend b + v * zt(s) (zt = z-score with trainC OOF stats), v on a grid chosen on trainC OOF.
Writes off/sc_inc_TAG.npz with keys 'NAME:inc:sel|S' = delta to add to b (the trainC-selected variant of each cfg)."""
import sys, gzip, json, os, numpy as np
H = os.path.dirname(os.path.abspath(__file__))
T = {}
SN = dict(np.load(f'{H}/../job/tables/sc_g4.npz'))
for S in ('trainC', 'dev', 'dev2'):
    rows = [json.loads(l) for l in gzip.open(f'{H}/../job/tables/attempts_{S}.jsonl.gz', 'rt')]
    lg = np.array([-30 if r['lg'] is None else r['lg'] for r in rows], float)
    T[S] = dict(idx=np.array([r['idx'] for r in rows]), ok=np.array([r['ok'] for r in rows], bool), lg=lg,
                cm=np.array([r['hash_off'] is not None and r['lg'] is not None and r['lg'] > -29.5 for r in rows]),
                b=np.maximum(SN[f'c4:HM:lin:pair|{S}'], -12) + np.maximum(lg, -12))
    t = T[S]; m0 = t['cm']; o = np.flatnonzero(m0)[np.argsort(t['idx'][m0], kind='stable')]
    ii = t['idx'][o]; cut = np.flatnonzero(np.diff(ii)) + 1
    t['groups'] = [(o[a], t['ok'][o[a]]) for a in np.split(np.arange(len(o)), cut)]
    t['groups'] = [(g, y) for g, y in t['groups'] if y.any() and (~y).any()]


def within(S, s):
    num = den = 0
    for g, y in T[S]['groups']:
        ss = s[g]; d = ss[y][:, None] - ss[~y][None]; num += (d > 0).sum() + .5 * (d == 0).sum(); den += d.size
    return num / den


def wcorr(S, a, b):
    """mean within-item correlation (items with >= 3 committed attempts), pooled after centering per item"""
    t = T[S]; m0 = t['cm']; ac = a.astype(float).copy(); bc = b.astype(float).copy(); keep = np.zeros(len(a), bool)
    for it in np.unique(t['idx'][m0]):
        j = np.flatnonzero(m0 & (t['idx'] == it))
        if len(j) < 3: continue
        ac[j] -= ac[j].mean(); bc[j] -= bc[j].mean(); keep[j] = True
    return float(np.corrcoef(ac[keep], bc[keep])[0, 1])


VG = [0, .05, .1, .2, .3, .5, .75, 1, 1.5, 2, 3, 5]
base = {S: within(S, T[S]['b']) for S in T}
print('b = max(sn,-12)+max(lg,-12) within: ' + ' '.join(f'{S} {base[S]:.4f}' for S in T), flush=True)
print('lg within: ' + ' '.join(f'{S} {within(S, T[S]["lg"]):.4f}' for S in T), flush=True)
print(f'{"cfg":28s} | alone/delta trainC dev dev2 | full(b+delta or b+v*z) trainC dev dev2 | Δ vs b trainC dev dev2 | v | wcorr(s,b) dev', flush=True)
OUT = {}
for fn in sys.argv[1:]:
    tag = os.path.basename(fn)[3:-4]
    sc = dict(np.load(fn)); cfgs = sorted({k.split('|')[0] for k in sc}, key=lambda c: list(sc).index(f'{c}|trainC'))
    for c in cfgs:
        s = {S: sc[f'{c}|{S}'].astype(float) for S in T}
        inc = ':inc' in c
        mu, sd = s['trainC'][T['trainC']['cm']].mean(), s['trainC'][T['trainC']['cm']].std() + 1e-9
        z = {S: (s[S] - mu) / sd for S in T}
        if inc:
            cand = {v: {S: v * s[S] for S in T} for v in [0, .25, .5, .75, 1, 1.5, 2]}
        else:
            sdb = T['trainC']['b'][T['trainC']['cm']].std()
            cand = {v: {S: v * sdb * z[S] for S in T} for v in VG}
        vb = max(cand, key=lambda v: within('trainC', T['trainC']['b'] + cand[v]['trainC']))
        alone = {S: within(S, s[S]) for S in T}
        full = {S: within(S, T[S]['b'] + cand[vb][S]) for S in T}
        f1 = {S: within(S, T[S]['b'] + cand[1][S]) for S in T} if inc else None
        wc = wcorr('dev', s['dev'], T['dev']['b'])
        print(f'{tag+":"+c:28s} | ' + ' '.join(f'{alone[S]:.4f}' for S in T) + ' | ' + ' '.join(f'{full[S]:.4f}' for S in T)
              + ' | ' + ' '.join(f'{full[S] - base[S]:+.4f}' for S in T) + f' | v={vb:g} | {wc:.2f}'
              + ((' | v=1: ' + ' '.join(f'{f1[S] - base[S]:+.4f}' for S in T)) if inc else ''), flush=True)
        nm = f'{tag}.{c.replace(":", "_")}:inc:sel'
        for S in T: OUT[f'{nm}|{S}'] = cand[vb][S].astype(np.float32)
        if inc and vb != 1:
            for S in T: OUT[f'{tag}.{c.replace(":", "_")}:inc:v1|{S}'] = cand[1][S].astype(np.float32)
tags = '_'.join(os.path.basename(f)[3:-4] for f in sys.argv[1:])
np.savez_compressed(f'{H}/sc_inc_{tags}.npz', **OUT)
print('wrote', f'{H}/sc_inc_{tags}.npz', len(OUT) // 3, 'deltas')
