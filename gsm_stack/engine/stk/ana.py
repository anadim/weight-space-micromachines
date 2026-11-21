"""Analysis of the stacked runs (pod CPU). usage: python ana.py OUTDIR SETS [--refit 1] [--probe FILE] [--hard_sets hard]
Writes OUTDIR/summary.json and per-question selection jsonl (OUTDIR/<set>_S3.jsonl, _S3a, _S4)."""
import sys, os, json, argparse, numpy as np, torch
ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('sets'); ap.add_argument('--refit', type=int, default=1)
ap.add_argument('--probe', default=''); ap.add_argument('--hard_sets', default='hard'); ap.add_argument('--use', default='auto')
ap.add_argument('--B', type=int, default=2000)
a = ap.parse_args(); O = a.out
HARD = set(a.hard_sets.split(','))


def L(fn): return [json.loads(l) for l in open(fn)] if os.path.exists(fn) else None


def keyv(p):
    if p is None: return None
    try: return round(float(p), 6)
    except (ValueError, TypeError): return None


def is_ok(v, gold):
    try: return v is not None and abs(float(v) - float(gold)) <= 1e-6 * max(1.0, abs(float(gold)))
    except ValueError: return False


def vote(keys, w, flags, sanity):
    idx = [i for i, k in enumerate(keys) if k is not None]
    if sanity:
        ok = [i for i in idx if flags[i] == 0]
        if ok: idx = ok
    if not idx: return None, idx
    sc, first = {}, {}
    for i in idx: sc[keys[i]] = sc.get(keys[i], 0.0) + w[i]; first.setdefault(keys[i], i)
    return max(sc, key=lambda k: (sc[k], -first[k])), idx


def probe_p(D, w, b):
    H = D['hcap'].float().numpy()
    p = 1 / (1 + np.exp(-(H @ w + b)))
    p[D['hpos'].numpy() < 0] = 0.0
    return p


def auroc_within(D, p):
    from sklearn.metrics import roc_auc_score
    K = D['K']; ok = np.array([r['ok'] for r in D['rows']]); n = len(ok) // K; au = []
    for j in range(n):
        y = ok[j * K:(j + 1) * K]
        if 0 < y.sum() < K: au.append(roc_auc_score(y, p[j * K:(j + 1) * K]))
    return round(float(np.mean(au)), 3), len(au)


# ---------------------------------------------------------------- probes
probes = {}
if a.probe:
    d = torch.load(a.probe); probes['orig'] = (d['w'].numpy().astype(np.float64), d['b'])
if a.refit and os.path.exists(f'{O}/trainC_samp.pt'):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Dt = torch.load(f'{O}/trainC_samp.pt')
    X = Dt['hcap'].float().numpy(); y = np.array([r['ok'] for r in Dt['rows']])
    has = (Dt['hpos'].numpy() >= 0) & np.array([keyv(r['pred']) is not None for r in Dt['rows']])
    SC = StandardScaler().fit(X[has]); PM = LogisticRegression(C=0.001, max_iter=3000).fit(SC.transform(X[has]), y[has])
    w_eff = PM.coef_[0] / SC.scale_; b_eff = PM.intercept_[0] - (PM.coef_[0] * SC.mean_ / SC.scale_).sum()
    torch.save(dict(layer=18, w=torch.tensor(w_eff, dtype=torch.float32), b=float(b_eff), C=0.001, n_train=int(has.sum()),
                    train=f'split C GSM8K, T=1, K={Dt["K"]}, calc v2 on ({O})'), f'{O}/verifier_probe_L18_refit.pt')
    probes['refit'] = (w_eff.astype(np.float64), float(b_eff))
    print('refit on', int(has.sum()), 'rows', flush=True)

summ = json.load(open(f'{O}/summary.json')) if os.path.exists(f'{O}/summary.json') else {}
# probe choice: within-question AUROC on dev samples (never test)
if 'probe_auroc' not in summ: summ['probe_auroc'] = {}
for nm in ('dev', 'dev2'):
    if os.path.exists(f'{O}/{nm}_samp.pt'):
        D = torch.load(f'{O}/{nm}_samp.pt')
        for pk, (w, b) in probes.items(): summ['probe_auroc'][f'{nm}/{pk}'] = auroc_within(D, probe_p(D, w, b))
if a.use == 'auto':
    au = summ['probe_auroc']
    use = 'refit' if ('refit' in probes and (au.get('dev/refit', (0,))[0] > au.get('dev/orig', (0,))[0] or 'orig' not in probes)) else 'orig'
else: use = a.use
summ['probe_used'] = use
W, Bb = probes[use] if probes else (None, None)
print('probe auroc', summ['probe_auroc'], 'use', use, flush=True)


def boot(d, B=a.B):
    rng = np.random.RandomState(0); d = np.asarray(d, float); n = len(d)
    m = [d[rng.randint(0, n, n)].mean() for _ in range(B)]
    return [round(100 * float(np.percentile(m, 2.5)), 2), round(100 * float(np.percentile(m, 97.5)), 2)]


def select(D, p, K, rule, adaptive=False):
    """returns per-question (answer, ok, tokens, samples drawn)."""
    rows = D['rows']; n = len(rows) // D['K']; out = []
    for j in range(n):
        sl = list(range(j * D['K'], (j + 1) * D['K']))
        stages = [4, 8, 16] if adaptive else [K]
        for st in stages:
            ii = sl[:st]; kk = [keyv(rows[i]['pred']) for i in ii]; ff = [rows[i]['flag'] for i in ii]
            ww = p[ii] if rule.startswith('probe') else np.ones(st)
            v, keep = vote(kk, ww, ff, rule.endswith('sanity'))
            tot = sum(ww[i] for i in keep) + 1e-9
            share = sum(ww[i] for i in keep if kk[i] == v) / tot if v is not None else 0
            if not adaptive or share >= 0.75 - 1e-6 or st == stages[-1]: break
        out.append(dict(ans=v, ok=int(is_ok(v, D['gold'][j])), ntok=sum(rows[i]['ntok'] for i in ii), drawn=len(ii)))
    return out


for nm in a.sets.split(','):
    hard = nm in HARD
    R = {}
    S0 = L(f'{O}/{nm}_S0.jsonl')
    if S0 is None: continue
    n = len(S0); base = np.array([x['ok'] for x in S0])
    res = {}

    def put(cfg, oks, toks, fire=None, extra=None):
        oks = np.asarray(oks, float)
        r = dict(acc=round(100 * oks.mean(), 2), tok_q=round(float(np.mean(toks)), 1), n=n,
                 fixes=int(((oks == 1) & (base == 0)).sum()), breaks=int(((oks == 0) & (base == 1)).sum()),
                 delta=round(100 * (oks.mean() - base.mean()), 2), ci_delta=boot(oks - base))
        if fire is not None: r['fire_rate'] = round(100 * float(np.mean(fire)), 2)
        if extra: r.update(extra)
        res[cfg] = r
        R[cfg] = oks
    put('S0', base, [x['ntok'] for x in S0])
    S1 = L(f'{O}/{nm}_S1.jsonl')
    if S1:
        sil = [k for k in range(n) if S1[k]['c2fires'] == 0]
        put('S1', [x['ok'] for x in S1], [x['ntok'] for x in S1], [x['c2fires'] > 0 for x in S1],
            dict(silent_identical=f"{sum(S1[k]['tokens'] == S0[k]['tokens'] for k in sil)}/{len(sil)}"))
    S2 = L(f'{O}/{nm}_S2.jsonl')
    if S2:
        put('S2', [x['ok'] for x in S2], [x['ntok'] for x in S2], [x['fire'] > 0 for x in S2],
            dict(c2_fire_rate=res['S1']['fire_rate'], vs_S1=dict(fixes=sum(1 for x, y in zip(S2, S1) if x['ok'] and not y['ok']),
                 breaks=sum(1 for x, y in zip(S2, S1) if not x['ok'] and y['ok'])),
                 path_counts={pth: sum(1 for x in S2 if x['path'] == pth) for pth in sorted({x['path'] for x in S2})}))
    if os.path.exists(f'{O}/{nm}_samp.pt'):
        D = torch.load(f'{O}/{nm}_samp.pt')
        p = probe_p(D, W, Bb) if W is not None else np.ones(len(D['rows']))
        rule = 'majority' if hard else 'probe_vote_sanity'
        g1 = [keyv(x['pred']) for x in S1] if S1 else None
        for cfg, K, ad, rl in (('S3', 8, False, rule), ('S3a', 16, True, rule), ('maj8', 8, False, 'majority'),
                               ('majsan8', 8, False, 'majority_sanity'), ('pv8', 8, False, 'probe_vote'), ('pvsan8', 8, False, 'probe_vote_sanity'),
                               ('pvsan16', 16, False, 'probe_vote_sanity'), ('maj16', 16, False, 'majority'), ('sample0', 1, False, 'majority')):
            sel = select(D, p, K, rl, ad)
            ch = [s['ans'] != g1[j] for j, s in enumerate(sel)] if g1 else None
            put(cfg, [s['ok'] for s in sel], [s['ntok'] for s in sel], ch,
                dict(rule=rl, K=('adaptive 4/8/16 thr .75' if ad else K), mean_samples=round(float(np.mean([s['drawn'] for s in sel])), 2)))
            if cfg in ('S3', 'S3a'):
                with open(f'{O}/{nm}_{cfg}.jsonl', 'w') as f:
                    for j, s in enumerate(sel): f.write(json.dumps(dict(j=j, gold=D['gold'][j], **s)) + '\n')
        res['samples'] = dict(mean_ok=round(100 * float(np.mean([r['ok'] for r in D['rows']])), 2),
                              flagged=round(100 * float(np.mean([r['flag'] != 0 for r in D['rows']])), 2),
                              c2_fire=round(100 * float(np.mean([r['c2fires'] > 0 for r in D['rows']])), 2),
                              oracle8=round(100 * float(np.mean([any(D['rows'][j * D['K'] + i]['ok'] for i in range(8)) for j in range(n)])), 2))
        if os.path.exists(f'{O}/{nm}_rep.pt') and not hard:
            Dr = torch.load(f'{O}/{nm}_rep.pt')
            pr = probe_p(Dr, W, Bb)
            sel = select(Dr, pr, 8, 'probe_vote_sanity')
            put('S4', [s['ok'] for s in sel], [s['ntok'] for s in sel],
                [any(Dr['rows'][j * Dr['K'] + i]['fire'] > 0 for i in range(8)) for j in range(n)], dict(rule='repair flagged, then probe_vote_sanity', K=8))
            with open(f'{O}/{nm}_S4.jsonl', 'w') as f:
                for j, s in enumerate(sel): f.write(json.dumps(dict(j=j, gold=Dr['gold'][j], **s)) + '\n')
    summ[nm] = res
    print(nm, json.dumps({k: (v['acc'], v['tok_q'], v['fixes'], v['breaks'], v.get('fire_rate')) for k, v in res.items() if 'acc' in v}), flush=True)
json.dump(summ, open(f'{O}/summary.json', 'w'), indent=1)
