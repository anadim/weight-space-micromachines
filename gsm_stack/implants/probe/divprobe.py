"""Step-level divergence probes (node, CPU ok): is the first divergent step D detectable from the hidden state
(a: at the end of the D line, 'error detector'), and is it anticipated (b: at the line end just before the D line)?

    bash run.sh python3 divprobe.py --train dev --test dev2      # -> out/divprobe_{train}.json

Labels: tables/divlab_{set}.jsonl.gz (gid, dl = first token of the D line, dt = token of the divergent value; atlas diverge).
(a) positives = D-line end rows of wrong attempts (mixed items); negatives = line ends (step >= 1) of right attempts of
    mixed items + wrong attempts' line ends before the D line. Features: h_t, dh = h_t - h_{t-1}, [h_t, dh]. Metrics:
    pooled AUROC and matched within-item AUROC (D-line end vs right attempts' line end at the same step index).
(b) positives = wrong attempts' line end just before the D line (step t0), negatives = right attempts' line ends at the
    same step t0 of the same item; states demeaned per (item, t0) group for training; scored raw, matched within-item.
"""
import os, json, gzip, argparse, time, collections
import numpy as np, torch
import fit

ap = argparse.ArgumentParser()
ap.add_argument('--train', default='dev'); ap.add_argument('--test', default='dev2'); ap.add_argument('--layers', default='18,22')
a = ap.parse_args()
SETS = [a.train, a.test]; LS = [int(x) for x in a.layers.split(',')]
T0 = time.time(); RES = {}
def log(*x): print(f'[{time.time() - T0:6.0f}s]', *x, flush=True)
torch.set_num_threads(max(1, os.cpu_count() or 1))


def folds_of(groups, k=5, seed=0):
    ug = np.unique(groups); rng = np.random.default_rng(seed); rng.shuffle(ug)
    m = {g: i % k for i, g in enumerate(ug)}; return np.array([m[g] for g in groups])


def prep(S):
    A = dict(np.load(f'{fit.ROOT}/{S}/attempts.npz'))
    L_ = [json.loads(l) for l in gzip.open(f'{fit.ROOT}/{S}/attempts_light.jsonl.gz', 'rt')]
    pl = np.array([int(r.get('phrase_len') or 0) for r in L_]); gid = [r['gid'] for r in L_]
    pm = np.load(f'{fit.ROOT}/{S}/pos_meta.npz'); k = pm['kind'] == 1; att = pm['att'][k]; off = pm['off'][k]
    ho = A['hash_off'][att]; mask = (off >= pl[att]) & ((ho < 0) | (off < ho))
    X, m = fit.load(S, LS, 'nl', rows=mask); X = X.reshape(len(X), -1)
    att = m['att'].astype(int); off = m['off'].astype(int)
    per = collections.defaultdict(list)
    for r in np.lexsort((off, att)): per[int(att[r])].append(int(r))
    step = np.zeros(len(att), int); prev = np.full(len(att), -1)
    for i, rr in per.items():
        for t, r in enumerate(rr): step[r] = t; prev[r] = rr[t - 1] if t else -1
    ok = A['ok'].astype(int); idx = A['idx']
    pr = collections.defaultdict(list)
    for i, o in zip(idx, ok): pr[int(i)].append(o)
    mixed = {i for i, v in pr.items() if 0 < np.mean(v) < 1}
    gi = {g: i for i, g in enumerate(gid)}
    lab = [json.loads(l) for l in gzip.open(f'tables/divlab_{S}.jsonl.gz', 'rt')]
    D = {}
    for d in lab:
        i = gi.get(d['gid'])
        if i is None or i not in per: continue
        offs = off[per[i]]; e1 = np.flatnonzero(offs >= d['dt']); e0 = np.flatnonzero(offs < d['dl'])
        if not len(e1): continue
        D[i] = (int(e0[-1]) if len(e0) else -1, int(e1[0]), d['cat'])
    eq = {}
    for l in gzip.open(f'tables/valtok_{S}.jsonl.gz', 'rt'):
        v = json.loads(l)
        if v['gid'] in gi: eq[gi[v['gid']]] = v['eq']
    log(S, 'rows', len(att), 'wrong with D', len(D), 'mixed items', len(mixed))
    return dict(X=X, att=att, off=off, per=per, step=step, prev=prev, ok=ok, idx=idx, mixed=mixed, D=D, eq=eq)


DS = {S: prep(S) for S in SETS}


def task_a(S, eq_only=False):
    d = DS[S]; rows, y = [], []
    for i, rr in d['per'].items():
        if d['idx'][i] not in d['mixed']: continue
        if eq_only:                                       # keep only line ends whose line holds a computed value '... = v'
            eqs = np.array(d['eq'].get(i, []), int); offs = d['off'][rr]; lo = np.r_[-1, offs[:-1]]
            iseq = [bool(((eqs > l) & (eqs <= h)).any()) for l, h in zip(lo, offs)]
        else:
            iseq = [True] * len(rr)
        if d['ok'][i]:
            neg = [r for t, r in enumerate(rr) if t >= 1 and iseq[t]]; rows += neg; y += [0] * len(neg)
        elif i in d['D']:
            t0, t1, _ = d['D'][i]
            if t1 < 1 or not iseq[t1]: continue
            neg = [rr[t] for t in range(1, t0 + 1) if iseq[t]]; rows += neg + [rr[t1]]; y += [0] * len(neg) + [1]
    rows = np.array(rows); y = np.array(y, np.float32)
    return rows, y


def feats(S, rows, kind):
    d = DS[S]; h = d['X'][rows].astype(np.float32)
    if kind == 'h': return h
    dh = h - d['X'][d['prev'][rows]].astype(np.float32)
    return dh if kind == 'dh' else np.concatenate([h, dh], 1)


def matched_w(S, score_of_row, which):
    """within-item AUROC: wrong attempt's row at step t (t1 for 'a', t0 for 'b') vs right attempts' rows at step t."""
    d = DS[S]; num = den = 0.0
    right = collections.defaultdict(list)
    for i in d['per']:
        if d['ok'][i] and d['idx'][i] in d['mixed']: right[d['idx'][i]].append(i)
    for i, (t0, t1, _) in d['D'].items():
        t = t1 if which == 'a' else t0
        if t < (1 if which == 'a' else 0): continue
        sw = score_of_row(d['per'][i][t])
        rv = np.array([score_of_row(d['per'][j][t]) for j in right[d['idx'][i]] if len(d['per'][j]) > t], float)
        rv = rv[~np.isnan(rv)]
        if np.isnan(sw) or not len(rv): continue
        num += (sw > rv).sum() + 0.5 * (sw == rv).sum(); den += len(rv)          # P(wrong flagged above right)
    return num / den if den else float('nan'), int(den)


# ---------------- (a) error detector at the end of the D line
RES['a'] = {}
for eq_only, kind in ((False, 'h'), (False, 'dh'), (False, 'hdh'), (True, 'h'), (True, 'hdh')):
    rtr, ytr = task_a(a.train, eq_only); rte, yte = task_a(a.test, eq_only); g = DS[a.train]['idx'][DS[a.train]['att'][rtr]]
    w = np.where(ytr == 1, 0.5 / ytr.mean(), 0.5 / (1 - ytr.mean()))
    log('task a', 'eq_only' if eq_only else 'all lines', 'rows', len(rtr), 'pos', int(ytr.sum()), '| test', len(rte), int(yte.sum()))
    Xtr = feats(a.train, rtr, kind); Xte = feats(a.test, rte, kind); best = None
    for l2 in (1e-2, 1e-1, 1.0):
        f = folds_of(g); oof = np.zeros(len(ytr), np.float32)
        for k in range(5):
            p, _, _ = fit.fit_logreg(Xtr[f != k], ytr[f != k], l2=l2, w=w[f != k]); oof[f == k] = p(Xtr[f == k])
        au = fit.auroc(ytr, oof)
        if best is None or au > best[0]: best = (au, l2, oof)
    au, l2, oof = best
    p, W, b = fit.fit_logreg(Xtr, ytr, l2=l2, w=w); ste = p(Xte)
    s_tr = dict(zip(rtr.tolist(), oof)); s_te = dict(zip(rte.tolist(), ste))
    mw_tr = matched_w(a.train, lambda r: s_tr.get(r, np.nan), 'a'); mw_te = matched_w(a.test, lambda r: s_te.get(r, np.nan), 'a')
    kname = kind + ('_eq' if eq_only else ''); RES['a'][kname] = dict(n_train=len(ytr), pos_train=int(ytr.sum()), n_test=len(yte), pos_test=int(yte.sum()), l2=l2, oof_auc=au, test_auc=fit.auroc(yte, ste), oof_matched_w=mw_tr[0], test_matched_w=mw_te[0],
                          pairs_test=mw_te[1])
    log('a', kname, json.dumps(RES['a'][kname]))
    # per category (test)
    cat_of = {DS[a.test]['per'][i][t1]: c for i, (t0, t1, c) in DS[a.test]['D'].items() if t1 >= 1}
    pc = {}
    for c in sorted(set(cat_of.values())):
        keep = np.array([(yte[j] == 0) or (cat_of.get(int(r)) == c) for j, r in enumerate(rte)])
        if (yte[keep] == 1).sum() >= 10: pc[c] = round(fit.auroc(yte[keep], ste[keep]), 3)
    RES['a'][kname]['test_by_cat'] = pc
    log('a', kname, 'by cat', pc)
    # operating points: per-line FPR fixed on train OOF negatives; test TPR on D lines, attempt-level flag rates
    dt = DS[a.test]; ops = {}
    neg_tr = oof[ytr == 0]
    for fpr in (0.01, 0.02, 0.05, 0.1):
        th = float(np.quantile(neg_tr, 1 - fpr)); fl = ste > th
        att_te = dt['att'][rte]; right = dt['ok'][att_te] == 1
        r_any = collections.defaultdict(bool); w_hit = collections.defaultdict(bool); w_early = collections.defaultdict(bool)
        for j, r in enumerate(rte):
            i = int(att_te[j])
            if right[j]: r_any[i] |= bool(fl[j])
            elif yte[j] == 1: w_hit[i] |= bool(fl[j])
            else: w_early[i] |= bool(fl[j])
        ops[fpr] = dict(thr=th, line_fpr_test=float(fl[yte == 0].mean()), tpr_D=float(fl[yte == 1].mean()),
                        right_att_flagged=float(np.mean(list(r_any.values()))),
                        wrong_flag_at_D=float(np.mean(list(w_hit.values()))) if w_hit else None,
                        wrong_flag_before_D=float(np.mean([w_early.get(i, False) for i in w_hit])) if w_hit else None)
    RES['a'][kname]['ops'] = ops
    log('a', kname, 'ops', json.dumps(ops))

# ---------------- (b) anticipation: line end before the D line vs right attempts at the same step (within-item)
def task_b(S):
    d = DS[S]; rows, y, key = [], [], []
    right = collections.defaultdict(list)
    for i in d['per']:
        if d['ok'][i] and d['idx'][i] in d['mixed']: right[d['idx'][i]].append(i)
    for i, (t0, t1, _) in d['D'].items():
        if t0 < 0: continue
        rv = [d['per'][j][t0] for j in right[d['idx'][i]] if len(d['per'][j]) > t0]
        if not rv: continue
        kk = int(d['idx'][i]) * 64 + min(t0, 63)
        rows += [d['per'][i][t0]] + rv; y += [1] + [0] * len(rv); key += [kk] * (1 + len(rv))
    rows = np.array(rows); y = np.array(y, np.float32); key = np.array(key)
    _, first = np.unique(np.c_[key, rows], axis=0, return_index=True)            # right rows can repeat across wrong attempts
    first = np.sort(first); return rows[first], y[first], key[first]


RES['b'] = {}
rb, yb, kb = task_b(a.train); rbt, ybt, kbt = task_b(a.test)
log('task b rows', len(rb), 'pos', int(yb.sum()), '| test', len(rbt), int(ybt.sum()))
for kind in ('h', 'dh'):
    if kind == 'dh':
        ok_tr = DS[a.train]['prev'][rb] >= 0; ok_te = DS[a.test]['prev'][rbt] >= 0
    else:
        ok_tr = np.ones(len(rb), bool); ok_te = np.ones(len(rbt), bool)
    R, Y, K = rb[ok_tr], yb[ok_tr], kb[ok_tr]; Rt, Yt = rbt[ok_te], ybt[ok_te]
    X = feats(a.train, R, kind); uk, inv = np.unique(K, return_inverse=True)
    mu = np.zeros((len(uk), X.shape[1]), np.float32); np.add.at(mu, inv, X); mu /= np.bincount(inv)[:, None]
    Xd = X - mu[inv]
    w = np.where(Y == 1, 0.5 / Y.mean(), 0.5 / (1 - Y.mean()))
    gb = K // 64; best = None
    for l2 in (1e-1, 1.0, 10.0):
        f = folds_of(gb); oof = np.zeros(len(Y), np.float32)
        for k in range(5):
            p, _, _ = fit.fit_logreg(Xd[f != k], Y[f != k], l2=l2, w=w[f != k]); oof[f == k] = p(X[f == k])
        s_tr = dict(zip(R.tolist(), oof)); au = matched_w(a.train, lambda r: s_tr.get(r, np.nan), 'b')[0]
        if best is None or au > best[0]: best = (au, l2)
    au, l2 = best
    p, W, b = fit.fit_logreg(Xd, Y, l2=l2, w=w); ste = p(feats(a.test, Rt, kind)); s_te = dict(zip(Rt.tolist(), ste))
    mw = matched_w(a.test, lambda r: s_te.get(r, np.nan), 'b')
    RES['b'][kind] = dict(l2=l2, oof_matched_w=au, test_matched_w=mw[0], pairs_test=mw[1])
    log('b', kind, json.dumps(RES['b'][kind]))

os.makedirs('out', exist_ok=True)
json.dump(RES, open(f'out/divprobe_{a.train}_{a.test}.json', 'w'), indent=1)
log('done')
