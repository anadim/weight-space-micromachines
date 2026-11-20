"""Answer selection over K samples: majority vote, sanity-filtered vote, learned verifier probe (fit on split C only),
log-prob vote, adaptive sampling. Runs on the pod CPU on the .pt dumps of sel_hook.py.

usage: python analyze.py OUTDIR [--T 1.0] [--sets dev,dev2]
"""
import sys, os, json, argparse, random, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--T', default='1.0'); ap.add_argument('--trainT', default='1.0,0.7')
ap.add_argument('--sets', default='dev,dev2'); ap.add_argument('--R', type=int, default=30)
ap.add_argument('--B', type=int, default=4000); ap.add_argument('--res', default='')
ap.add_argument('--Ks', default='1,2,4,8,16'); ap.add_argument('--merge', type=int, default=0); ap.add_argument('--fixprobe', default='')
a = ap.parse_args()
O = a.out


def load(nm, mode, G, s=0):
    fn = f'{O}/{nm}_{mode}_G{G}_s{s}.pt'
    return torch.load(fn) if os.path.exists(fn) else None


def keyv(p):
    if p is None: return None
    try: return round(float(p), 6)
    except ValueError: return None


def feats(D, li):
    H = D['hcap'][:, li].float().numpy()
    has = (D['hpos'].numpy() >= 0) & np.array([keyv(p) is not None for p in D['pred']])
    return H, has


# ------------------------------------------------------------------ probe on split C (disjoint from dev / dev2 / test)
TR = [load('trainC', f'T{t}', 8) for t in a.trainT.split(',')]
TR = [d for d in TR if d is not None]
print('train files', [(d['mode'], len(d['q'])) for d in TR], flush=True)
layers = TR[0]['layers']


def vote(keys, w, flags=None, sanity=False):
    """weighted exact vote over answer values; None abstains; sanity: drop flag != 0 unless nothing is left.
    ties -> the answer whose first occurrence is earliest."""
    idx = [i for i, k in enumerate(keys) if k is not None]
    if sanity:
        ok = [i for i in idx if flags[i] == 0]
        if ok: idx = ok
    if not idx: return None
    sc, first = {}, {}
    for i in idx:
        sc[keys[i]] = sc.get(keys[i], 0.0) + w[i]; first.setdefault(keys[i], i)
    return max(sc, key=lambda k: (sc[k], -first[k]))


def grp(D):
    G = D['G']; n = len(D['q'])
    keys = [keyv(p) for p in D['pred']]
    return [(keys[j * G:(j + 1) * G], D['flag'][j * G:(j + 1) * G], D['ntok'][j * G:(j + 1) * G]) for j in range(n)]


def cv_probe(li, Cc, folds=5):
    """problem-grouped CV on split C: out-of-fold p(correct) and the probe-vote accuracy it gives (K=8)."""
    Xs, ys, gs, Ds = [], [], [], []
    for d in TR:
        H, has = feats(d, li); Xs.append(H); ys.append(np.array(d['ok'])); gs.append(np.arange(len(d['ok'])) // d['G'])
    X = np.concatenate(Xs); y = np.concatenate(ys)
    g = np.concatenate(gs)                       # same question in both temps -> same group id
    has = np.concatenate([feats(d, li)[1] for d in TR])
    rng = np.random.RandomState(0); perm = rng.permutation(g.max() + 1); fold = perm[g] % folds
    oof = np.zeros(len(y))
    for f in range(folds):
        tr = (fold != f) & has; te = fold == f
        sc = StandardScaler().fit(X[tr]); m = LogisticRegression(C=Cc, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    oof[~has] = 0.0
    # selection accuracy from oof weights
    acc, o = [], 0
    for d in TR:
        n = len(d['ok']); w = oof[o:o + n]; o += n; G = d['G']
        keys = [keyv(p) for p in d['pred']]
        for j in range(n // G):
            sl = slice(j * G, (j + 1) * G)
            v = vote(keys[sl], list(w[sl]))
            gold = keyv(d['gold'][j]); acc.append(float(v is not None and gold is not None and abs(v - gold) <= 1e-6 * max(1, abs(gold))))
    from sklearn.metrics import roc_auc_score
    return float(np.mean(acc)), float(roc_auc_score(y[has], oof[has]))


if a.fixprobe:
    li, Cc = [int(x) for x in a.fixprobe.split(':')[:1]] + [float(a.fixprobe.split(':')[1])]
    li = layers.index(li); cvres = {}
else:
    cvres = {}
    for li, L in enumerate(layers):
        for Cc in (0.001, 0.003, 0.01, 0.03):
            cvres[f'L{L}_C{Cc}'] = cv_probe(li, Cc); print('CV', L, Cc, cvres[f'L{L}_C{Cc}'], flush=True)
    best = max(cvres, key=lambda k: cvres[k][0])
    li = layers.index(int(best.split('_')[0][1:])); Cc = float(best.split('_C')[1])
print('probe: layer', layers[li], 'C', Cc, flush=True)
X = np.concatenate([feats(d, li)[0][feats(d, li)[1]] for d in TR])
y = np.concatenate([np.array(d['ok'])[feats(d, li)[1]] for d in TR])
SC = StandardScaler().fit(X); PM = LogisticRegression(C=Cc, max_iter=3000).fit(SC.transform(X), y)
# fold the standardisation into one affine read: p = sigmoid(w . h + b)  (this is the in-engine probe)
w_eff = PM.coef_[0] / SC.scale_; b_eff = PM.intercept_[0] - (PM.coef_[0] * SC.mean_ / SC.scale_).sum()
torch.save(dict(layer=layers[li], w=torch.tensor(w_eff, dtype=torch.float32), b=float(b_eff), C=Cc, n_train=len(y),
                train='GSM8K train split C (md5 part), T in ' + a.trainT + ', G=8'), f'{O}/verifier_probe_L{layers[li]}.pt')


def pcorr(D):
    H, has = feats(D, li)
    p = 1 / (1 + np.exp(-(H @ w_eff + b_eff)))
    p[~has] = 0.0
    return p


def is_ok(v, gold):
    g = keyv(gold)
    return float(v is not None and g is not None and abs(v - g) <= 1e-6 * max(1, abs(g)))


RULES = ['sample0', 'majority', 'maj_sanity', 'probe_vote', 'probe_vote_sanity', 'probe_select', 'logprob_vote']


def rule_pick(rule, keys, flags, p, lpm):
    K = len(keys)
    if rule == 'sample0': return keys[0]
    if rule == 'majority': return vote(keys, [1.0] * K)
    if rule == 'maj_sanity': return vote(keys, [1.0] * K, flags, True)
    if rule == 'probe_vote': return vote(keys, list(p))
    if rule == 'probe_vote_sanity': return vote(keys, list(p), flags, True)
    if rule == 'probe_select':
        i = int(np.argmax(p)); return keys[i]
    if rule == 'logprob_vote': return vote(keys, list(np.exp(np.clip(lpm, -50, 0))))


def evalset(nm, T):
    Dg = load(nm, 'greedy', 1); Ds = load(nm, f'T{T}', 16)
    D1 = load(nm, f'T{T}', 16, 1) if a.merge else None
    if Ds is not None and D1 is not None:          # two independent 16-sample seeds -> G = 32 per question
        n_ = len(Ds['q']); idx0 = [j * 16 + i for j in range(n_) for i in range(16)]
        M = dict(Ds); M['G'] = 32
        order = [(0, j * 16 + i) if i < 16 else (1, j * 16 + i - 16) for j in range(n_) for i in range(32)]
        for k in ('tokens', 'pred', 'ok', 'flag', 'ntok', 'lp_mean'):
            M[k] = [(Ds if s == 0 else D1)[k][r] for s, r in order]
        M['hcap'] = torch.stack([(Ds if s == 0 else D1)['hcap'][r] for s, r in order])
        M['hpos'] = torch.stack([(Ds if s == 0 else D1)['hpos'][r] for s, r in order])
        Ds = M
    if Ds is None: return None
    n = len(Ds['q']); G = Ds['G']
    p = pcorr(Ds); keys = [keyv(x) for x in Ds['pred']]; lpm = np.array(Ds['lp_mean'])
    res = {'n': n, 'T': T}
    if Dg is not None:
        res['greedy'] = 100 * float(np.mean(Dg['ok'])); res['greedy_tok'] = float(np.mean(Dg['ntok']))
        gk = [keyv(x) for x in Dg['pred']]; gp = pcorr(Dg)
    res['tok_per_sample'] = float(np.mean(Ds['ntok']))
    rng = random.Random(0)
    Ks = [int(x) for x in a.Ks.split(',')]
    perq = {}                       # (rule, K) -> per-question expected accuracy (mean over R subsets)
    toks = {}
    for K in Ks:
        subs = [list(range(K))] + [rng.sample(range(G), K) for _ in range(a.R - 1)] if K < G else [list(range(G))]
        for rule in RULES:
            accq = np.zeros(n)
            for j in range(n):
                sl = list(range(j * G, (j + 1) * G))
                tot = 0.0
                for s in subs:
                    ii = [sl[i] for i in s]
                    v = rule_pick(rule, [keys[i] for i in ii], [Ds['flag'][i] for i in ii], p[ii], lpm[ii])
                    tot += is_ok(v, Ds['gold'][j])
                accq[j] = tot / len(subs)
            perq[(rule, K)] = accq
        toks[K] = float(np.mean([sum(Ds['ntok'][j * G + i] for i in s) / 1 for j in range(n) for s in subs[:1]]))
        # greedy + (K-1) samples (row 0 greedy): a deterministic first attempt plus K-1 samples
        if Dg is not None and K >= 2:
            for rule in ['majority', 'maj_sanity', 'probe_vote', 'probe_vote_sanity']:
                accq = np.zeros(n)
                subs2 = [list(range(K - 1))] + [rng.sample(range(G), K - 1) for _ in range(a.R - 1)]
                for j in range(n):
                    tot = 0.0
                    for s in subs2:
                        ii = [j * G + i for i in s]
                        kk = [gk[j]] + [keys[i] for i in ii]; ff = [Dg['flag'][j]] + [Ds['flag'][i] for i in ii]
                        pp = np.concatenate([[gp[j]], p[ii]])
                        v = rule_pick(rule, kk, ff, pp, None)
                        tot += is_ok(v, Ds['gold'][j])
                    accq[j] = tot / len(subs2)
                perq[('g+' + rule, K)] = accq
    res['acc'] = {f'{r}@{K}': round(100 * float(v.mean()), 2) for (r, K), v in perq.items()}
    res['tok'] = {f'K{K}': round(toks[K], 1) for K in Ks}
    res['oracle@16'] = round(100 * float(np.mean([max(Ds['ok'][j * G:(j + 1) * G]) for j in range(n)])), 2)
    # paired bootstrap over questions
    bs = np.random.RandomState(0)
    def boot(x, y):
        d = x - y; m = d.mean(); idx = bs.randint(0, n, (a.B, n)); b = d[idx].mean(1)
        return [round(100 * m, 2), round(100 * np.percentile(b, 2.5), 2), round(100 * np.percentile(b, 97.5), 2)]
    cmp = {}
    for K in Ks:
        if K == 1: continue
        cmp[f'probe_vote-majority@{K}'] = boot(perq[('probe_vote', K)], perq[('majority', K)])
        cmp[f'maj_sanity-majority@{K}'] = boot(perq[('maj_sanity', K)], perq[('majority', K)])
        cmp[f'probe_vote_sanity-maj_sanity@{K}'] = boot(perq[('probe_vote_sanity', K)], perq[('maj_sanity', K)])
        cmp[f'probe_select-majority@{K}'] = boot(perq[('probe_select', K)], perq[('majority', K)])
        cmp[f'logprob_vote-majority@{K}'] = boot(perq[('logprob_vote', K)], perq[('majority', K)])
        if Dg is not None:
            g = np.array(Dg['ok'], float)
            cmp[f'majority-greedy@{K}'] = boot(perq[('majority', K)], g)
            cmp[f'probe_vote_sanity-greedy@{K}'] = boot(perq[('probe_vote_sanity', K)], g)
            cmp[f'g+probe_vote_sanity-greedy@{K}'] = boot(perq[('g+probe_vote_sanity', K)], g)
    res['cmp'] = cmp
    # within-question AUROC of the probe vs logprob (questions with mixed outcomes)
    from sklearn.metrics import roc_auc_score
    au_p, au_l = [], []
    for j in range(n):
        o = np.array(Ds['ok'][j * G:(j + 1) * G])
        if 0 < o.sum() < G:
            au_p.append(roc_auc_score(o, p[j * G:(j + 1) * G])); au_l.append(roc_auc_score(o, lpm[j * G:(j + 1) * G]))
    res['within_auroc'] = dict(probe=round(float(np.mean(au_p)), 3), logprob=round(float(np.mean(au_l)), 3), nq=len(au_p))
    # ---------------- adaptive: k0 samples, stop if the (probe-weighted / plain) vote is decisive, else grow to 8, 16
    ad = {}
    for rule in ['maj_sanity', 'probe_vote_sanity']:
        for k0 in (2, 4):
            for thr in (1.0, 0.75, 0.6, 0.5):
                accs, tk = [], []
                for j in range(n):
                    sl = list(range(j * G, (j + 1) * G)); stages = [k0, 8, 16] if k0 < 8 else [8, 16]
                    for st in stages:
                        ii = sl[:st]; kk = [keys[i] for i in ii]; ff = [Ds['flag'][i] for i in ii]
                        ww = p[ii] if rule.startswith('probe') else np.ones(st)
                        keep = [i for i, k in enumerate(kk) if k is not None and ff[i] == 0]
                        if not keep: keep = [i for i, k in enumerate(kk) if k is not None]
                        tot = sum(ww[i] for i in keep) + 1e-9
                        v = rule_pick(rule, kk, ff, p[ii], None)
                        share = sum(ww[i] for i in keep if kk[i] == v) / tot if v is not None else 0
                        if share >= thr - 1e-6 or st == 16: break
                    accs.append(is_ok(v, Ds['gold'][j])); tk.append(sum(Ds['ntok'][i] for i in ii))
                ad[f'{rule}_k0{k0}_thr{thr}'] = dict(acc=round(100 * float(np.mean(accs)), 2), tok=round(float(np.mean(tk)), 1),
                                                    mean_samples=round(float(np.mean(tk)) / res['tok_per_sample'], 2))
    # greedy-first adaptive: greedy row; then k0 samples; stop if all sane answers agree with greedy; else grow 8, 16
    if Dg is not None:
        for rule in ['maj_sanity', 'probe_vote_sanity']:
            for k0 in (0, 1, 2, 3):
                for thr in (1.0, 0.75, 0.5):
                    accs, tk = [], []
                    for j in range(n):
                        sl = list(range(j * G, (j + 1) * G))
                        for st in ([k0, 7, 15] if k0 > 0 else [0, 7, 15]):
                            ii = sl[:st]
                            kk = [gk[j]] + [keys[i] for i in ii]; ff = [Dg['flag'][j]] + [Ds['flag'][i] for i in ii]
                            pp = np.concatenate([[gp[j]], p[ii]])
                            ww = pp if rule.startswith('probe') else np.ones(len(kk))
                            v = rule_pick(rule, kk, ff, pp, None)
                            keep = [i for i, k in enumerate(kk) if k is not None and ff[i] == 0] or [i for i, k in enumerate(kk) if k is not None]
                            tot = sum(ww[i] for i in keep) + 1e-9
                            share = sum(ww[i] for i in keep if kk[i] == v) / tot if v is not None else 0
                            if k0 == 0 and st == 0:            # greedy alone: continue only if flagged (sanity gate)
                                if ff[0] == 0: break
                                continue
                            if share >= thr - 1e-6 or st == 15: break
                        accs.append(is_ok(v, Ds['gold'][j])); tk.append(Dg['ntok'][j] + sum(Ds['ntok'][i] for i in ii))
                    ad[f'G_{rule}_k0{k0}_thr{thr}'] = dict(acc=round(100 * float(np.mean(accs)), 2), tok=round(float(np.mean(tk)), 1),
                                                          tok_x_greedy=round(float(np.mean(tk)) / res['greedy_tok'], 2))
    res['adaptive'] = ad
    # fixes / breaks vs greedy for the recommended rule at K=8 (first 8 samples)
    if Dg is not None:
        fx, br, ex = 0, 0, []
        for j in range(n):
            ii = list(range(j * G, j * G + 8))
            v = rule_pick('probe_vote_sanity', [keys[i] for i in ii], [Ds['flag'][i] for i in ii], p[ii], None)
            o = is_ok(v, Ds['gold'][j]); g0 = Dg['ok'][j]
            if o and not g0: fx += 1
            if g0 and not o: br += 1
            if len(ex) < 6 and o != g0: ex.append(dict(j=j, gold=Ds['gold'][j], greedy=Dg['pred'][j], picked=v, fix=bool(o),
                                                    answers=[keys[i] for i in ii], p=[round(float(p[i]), 2) for i in ii]))
        res['pvs8_vs_greedy'] = dict(fixes=fx, breaks=br, examples=ex)
    # in-engine tensor reduction (probe_engine.group_select) == this python vote, on the first 8 samples
    try:
        sys.path.insert(0, os.environ.get('SEL_DIR', os.path.dirname(os.path.abspath(__file__))))
        from probe_engine import group_select
        K = 8; ii = np.array([j * G + i for j in range(n) for i in range(K)])
        vals = torch.tensor([keys[i] if keys[i] is not None else 0.0 for i in ii], dtype=torch.float64)
        valid = torch.tensor([keys[i] is not None and Ds['hpos'][i] >= 0 for i in ii])
        fl = torch.tensor([Ds['flag'][i] for i in ii])
        H = Ds['hcap'][ii, li].float()
        pk = group_select(vals, valid, fl, H, torch.tensor(w_eff), b_eff, G=K, sanity=True)
        agree = 0
        for j in range(n):
            sl = [j * G + i for i in range(K)]
            v = rule_pick('probe_vote_sanity', [keys[i] for i in sl], [Ds['flag'][i] for i in sl], p[sl], None)
            vt = None if int(pk[j]) < 0 else keys[sl[int(pk[j])]]
            agree += int(is_ok(v, Ds['gold'][j]) == is_ok(vt, Ds['gold'][j]))
        res['tensor_select_agree'] = f'{agree}/{n}'
    except Exception as e:
        res['tensor_select_agree'] = repr(e)
    return res


allres = dict(probe=dict(layer=layers[li], C=Cc, n_train=int(len(y)), cv=cvres))
for nm in a.sets.split(','):
    for T in a.T.split(','):
        r = evalset(nm, T)
        if r is None: print('missing', nm, T); continue
        allres[f'{nm}_T{T}'] = r
        print(nm, 'T', T, json.dumps({k: v for k, v in r.items() if k in ('n', 'greedy', 'greedy_tok', 'tok_per_sample', 'oracle@16', 'within_auroc', 'tensor_select_agree')}), flush=True)
        for K in a.Ks.split(','):
            print('  K', K, {rl: r['acc'].get(f'{rl}@{K}') for rl in RULES + ['g+majority', 'g+probe_vote_sanity']}, 'tok', r['tok'][f'K{K}'], flush=True)
        for k, v in r['cmp'].items(): print('  ', k, v)
        for k, v in r['adaptive'].items(): print('   adaptive', k, v)
        print('  ', r.get('pvs8_vs_greedy', {}).get('fixes'), 'fixes', r.get('pvs8_vs_greedy', {}).get('breaks'), 'breaks (pvs@8 vs greedy)')
json.dump(allres, open(a.res or f'{O}/sel_results_T{a.T}.json', 'w'), indent=1, default=float)
print('ANADONE')
