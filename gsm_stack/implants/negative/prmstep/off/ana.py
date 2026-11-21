"""lane prmstep offline analysis: attempt-level PRM score from per-line logits (job/out/sc_prm.npz), within-item AUROC alone / vs the
rung-4 blend b = max(sn,-12)+max(lg,-12), within-item corr, increment of b + v*prm (v on trainC OOF only), and an sc file for
attnver/repick_av.py (key 'prm_AGG:inc:v|S' = delta = v*prm).   python3.11 off/ana.py [SC] [OUTSC]"""
import sys, os, gzip, json
import numpy as np
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SC = dict(np.load(sys.argv[1] if len(sys.argv) > 1 else f'{H}/job/out/sc_prm.npz'))
OUTSC = sys.argv[2] if len(sys.argv) > 2 else f'{H}/off/sc_inc.npz'
G4 = dict(np.load(f'{H}/job/tables/sc_g4.npz'))
SETS = ('trainC', 'dev', 'dev2'); A = {}
for S in SETS:
    rows = [json.loads(l) for l in gzip.open(f'{H}/job/tables/attempts_{S}.jsonl.gz', 'rt')]
    idx = np.array([r['idx'] for r in rows]); ok = np.array([bool(r['ok']) for r in rows])
    lg = np.array([np.nan if r.get('lg') is None else float(r['lg']) for r in rows]); lg = np.nan_to_num(lg, nan=-30.0)
    hh = np.array([r.get('hash_off') is not None for r in rows])
    sn = G4[f'c4:HM:lin:pair|{S}']
    A[S] = dict(idx=idx, ok=ok, cm=hh & (lg > -29.5), b=np.maximum(sn, -12) + np.maximum(lg, -12), n=len(rows))
    del rows


def pairs(S):
    d = A[S]; R, W = [], []
    for it in np.unique(d['idx'][d['cm']]):
        ii = np.flatnonzero(d['cm'] & (d['idx'] == it)); yy = d['ok'][ii]
        if yy.all() or (~yy).all(): continue
        r, w = ii[yy], ii[~yy]; R.append(np.repeat(r, len(w))); W.append(np.tile(w, len(r)))
    return np.concatenate(R), np.concatenate(W)


PA = {S: pairs(S) for S in SETS}
def wauc(S, s):
    r, w = PA[S]; d = s[r] - s[w]; return float(((d > 0) + 0.5 * (d == 0)).mean())


def wcorr(S, x, y):
    d = A[S]; m = d['cm']; xs, ys = [], []
    for it in np.unique(d['idx'][m]):
        ii = np.flatnonzero(m & (d['idx'] == it))
        if len(ii) < 2: continue
        xs.append(x[ii] - x[ii].mean()); ys.append(y[ii] - y[ii].mean())
    return float(np.corrcoef(np.concatenate(xs), np.concatenate(ys))[0, 1])


def agg(S, key, how):
    l = SC[f'{key}|{S}']; att = SC[f'att|{S}']; lab = SC[f'lab|{S}']; e = lab >= 0
    n = A[S]['n']; out = np.full(n, np.nan)
    if how == 'mean':
        s = np.bincount(att[e], l[e], minlength=n); c = np.bincount(att[e], minlength=n); out[c > 0] = s[c > 0] / c[c > 0]
    elif how == 'min':
        o = np.full(n, np.inf); np.minimum.at(o, att[e], l[e]); out[np.isfinite(o)] = o[np.isfinite(o)]
    elif how == 'last':
        o = np.full(n, np.nan); o[att[e]] = l[e]; out = o                     # last eligible line (att sorted, lines in order)
    return out


keys = sorted({k.split('|')[0] for k in SC if '@' in k})
TR = json.load(open(f'{H}/job/out/train_pl.json')) if os.path.exists(f'{H}/job/out/train_pl.json') else {}
print('baseline b within-AUROC', {S: round(wauc(S, A[S]['b']), 4) for S in SETS})
best = max(keys, key=lambda k: TR.get(k, {}).get('trainC', 0)) if TR else keys[0]
print('line-AUROC (trainC OOF / dev / dev2) per config:')
for k in keys: print(' ', k, {s: round(v, 4) for s, v in TR.get(k, {}).items()}, '<- best trainC' if k == best else '')
OUT = {}; rows_out = []
for k in [best]:
    for how in ('min', 'mean', 'last'):
        P = {}
        for S in SETS:
            p = agg(S, k, how); fill = np.nanmedian(agg('trainC', k, how)); P[S] = np.where(np.isnan(p), fill, p)
        sd_tc = P['trainC'][A['trainC']['cm']].std()
        r = dict(cfg=k, agg=how, nan_frac={S: round(float(np.isnan(agg(S, k, how)).mean()), 3) for S in SETS},
                 auc_alone={S: round(wauc(S, P[S]), 4) for S in SETS},
                 corr_b={S: round(wcorr(S, P[S], A[S]['b']), 3) for S in SETS})
        vs = [0.0] + [x * 1.0 for x in (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)]
        tc = [wauc('trainC', A['trainC']['b'] + v * P['trainC']) for v in vs]
        vb = vs[int(np.argmax(tc))]
        r['v_trainC'] = vb; r['sd_prm_tc'] = round(float(sd_tc), 3)
        r['inc_grid_trainC'] = [round(x - tc[0], 4) for x in tc]
        r['auc_inc'] = {S: round(wauc(S, A[S]['b'] + vb * P[S]), 4) for S in SETS}
        r['delta_vs_b'] = {S: round(r['auc_inc'][S] - wauc(S, A[S]['b']), 4) for S in SETS}
        for v in sorted({vb, 0.3, 1.0}):
            if v == 0: continue
            for S in SETS: OUT[f'prm_{how}:inc:{v:g}|{S}'] = v * P[S]
        print(json.dumps(r), flush=True); rows_out.append(r)
np.savez(OUTSC, **OUT)
json.dump(rows_out, open(f'{H}/off/ana.json', 'w'), indent=1)
