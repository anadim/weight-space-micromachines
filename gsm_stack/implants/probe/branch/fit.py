"""Branch probe fits (local, CPU torch).  python3.11 probe/branch/fit.py [--train trainC] [--test dev,dev2]

Pair = fork with two candidate tokens a, b (identical visible context, host hidden state h). Label y = a leads right.
Models (all antisymmetric in (a,b) except 'trust'):
  base     : sign(lp_a - lp_b) (host teacher-forced logprob)
  had_L    : logistic on [dlp, hn_L * du] (hn = h / rms(h), du = g * (u_a - u_b), g = final-norm weight; for L=28 the
             feature sum is exactly the host logit difference -> diagonal reweighting of the host's own readout)
  lr_r     : s = beta*dlp + sum_L hn_L^T A_L B^T du, rank r (Adam, L2, early stop on grouped holdout)
  trust_L  : logistic on [z(h_L), |dlp|] predicting 'the higher-prob token is the right one' (token-agnostic)
Train sets: 'pairs' = atlas branch pairs (wrong attempt vs closest right, hard labels), 'forks' = all trie forks with
child values differing (soft labels v = n_right/n, weight |v_a - v_b|).
"""
import os, sys, json, gzip, argparse, collections
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f'{HERE}/data'
torch.set_num_threads(16)
LAYS = [14, 18, 22, 27, 28]


def rd(fn):
    with gzip.open(fn, 'rt') as f: return [json.loads(l) for l in f]


UN = None
def unemb():
    global UN
    if UN is None:
        z = np.load(f'{OUT}/unemb.npz')
        UN = dict(map={int(t): i for i, t in enumerate(z['ids'])}, U=z['U'].astype(np.float32), g=z['norm_w'].astype(np.float32))
    return UN


def load_set(S):
    """-> dict with H (n,5,1024) fp32 normalized per row, fork meta, pairs list (branch) and fork pairs (soft)"""
    H = np.load(f'{OUT}/{S}_fork_h.npy').astype(np.float32)
    M = dict(np.load(f'{OUT}/{S}_fork_meta.npz'))
    row = {f: i for i, f in enumerate(M['fid'])}
    forks = {f['fid']: f for f in rd(f'{HERE}/tables/forks_{S}.jsonl.gz')}
    pairs = [p for p in rd(f'{HERE}/tables/pairs_{S}.jsonl.gz') if p['clean'] and p['fid'] in row]
    def lp(r, t):
        c = M['cand'][r]; k = np.where(c == t)[0]
        return float(M['cand_lp'][r, k[0]]) if len(k) else float('nan')
    P = []
    for p in pairs:
        r = row[p['fid']]
        P.append(dict(r=r, a=p['tok_r'], b=p['tok_w'], y=1.0, w=1.0, lpa=lp(r, p['tok_r']), lpb=lp(r, p['tok_w']), cat=p['cat'],
                      idx=p['idx'], vs=p['b_vs_div'], typ=p['b_type'], slpa=p['lp_r'], slpb=p['lp_w'], n_wrong=p['n_wrong'],
                      top2=set(M['top_id'][r, :2].tolist()) == {p['tok_r'], p['tok_w']}, fid=p['fid'], wtxt=p['wtxt'], rtxt=p['rtxt']))
    FP = []
    for f, r in row.items():
        ch = forks[f]['children']
        for i in range(len(ch)):
            for j in range(i + 1, len(ch)):
                va, vb = ch[i][2] / ch[i][1], ch[j][2] / ch[j][1]
                if va == vb: continue
                FP.append(dict(r=r, a=ch[i][0], b=ch[j][0], y=float(va > vb), w=abs(va - vb), lpa=lp(r, ch[i][0]), lpb=lp(r, ch[j][0]),
                               idx=forks[f]['idx'], na=ch[i][1], nb=ch[j][1], va=va, vb=vb, cat='fork',
                               top2=set(M['top_id'][r, :2].tolist()) == {ch[i][0], ch[j][0]}, fid=f))
    rms = np.sqrt((H ** 2).mean(-1, keepdims=True)) + 1e-6
    return dict(S=S, H=H / rms, M=M, pairs=P, fpairs=FP)


def feats(d, P):
    U = unemb()
    ia = np.array([U['map'][p['a']] for p in P]); ib = np.array([U['map'][p['b']] for p in P])
    du = (U['U'][ia] - U['U'][ib]) * U['g']
    hn = d['H'][np.array([p['r'] for p in P])]
    dlp = np.array([p['lpa'] - p['lpb'] for p in P], np.float32)
    y = np.array([p['y'] for p in P], np.float32); w = np.array([p['w'] for p in P], np.float32)
    g = np.array([p['idx'] for p in P])
    return dict(hn=torch.tensor(hn), du=torch.tensor(du, dtype=torch.float32), dlp=torch.tensor(dlp), y=torch.tensor(y), w=torch.tensor(w), g=g)


def check_host(F):
    """sum(hn_28 * du) == host logit diff (sanity on the unembedding / norm convention)"""
    s = (F['hn'][:, 4] * F['du']).sum(-1)
    return float(np.corrcoef(s.numpy(), F['dlp'].numpy())[0, 1]), float((s - F['dlp']).abs().median())


class Model(torch.nn.Module):
    def __init__(s, kind, layers, r=16, D=1024):
        super().__init__(); s.kind, s.layers = kind, layers
        s.beta = torch.nn.Parameter(torch.ones(1))
        if kind == 'had': s.w = torch.nn.Parameter(torch.zeros(len(layers), D))
        if kind == 'lr':
            s.A = torch.nn.Parameter(torch.randn(len(layers), D, r) * 0.01); s.B = torch.nn.Parameter(torch.randn(D, r) * 0.01)
        if kind == 'trust':
            s.w = torch.nn.Parameter(torch.zeros(len(layers), D)); s.c = torch.nn.Parameter(torch.zeros(2))

    def forward(s, F):
        h = F['hn'][:, s.layers]
        if s.kind == 'had': return s.beta * F['dlp'] + torch.einsum('nld,ld,nd->n', h, s.w, F['du'])
        if s.kind == 'lr': return s.beta * F['dlp'] + torch.einsum('nld,ldr,nr->n', h, s.A, F['du'] @ s.B)
        if s.kind == 'trust':
            sg = torch.sign(F['dlp']); sg[sg == 0] = 1
            t = s.c[0] + s.c[1] * F['dlp'].abs() + torch.einsum('nld,ld->n', h, s.w)      # logit P(higher-prob token right)
            return sg * (F['dlp'].abs() * s.beta + t)   # > 0 <=> choose a ; trained with its own loss below
        if s.kind == 'base': return s.beta * F['dlp']

    def reg(s):
        if s.kind in ('had', 'trust'): return (s.w ** 2).sum()
        if s.kind == 'lr': return (s.A ** 2).sum() + (s.B ** 2).sum()
        return torch.zeros(())


def sub(F, m):
    return {k: (v[m] if isinstance(v, torch.Tensor) else v[m]) for k, v in F.items()}


def loss_of(model, F):
    s = model(F)
    if model.kind == 'trust':
        sg = torch.sign(F['dlp']); sg[sg == 0] = 1
        yy = ((sg > 0).float() == F['y']).float()                                          # higher-prob token is right
        h = F['hn'][:, model.layers]
        t = model.c[0] + model.c[1] * F['dlp'].abs() + torch.einsum('nld,ld->n', h, model.w)
        return (torch.nn.functional.binary_cross_entropy_with_logits(t, yy, reduction='none') * F['w']).sum() / F['w'].sum()
    return (torch.nn.functional.binary_cross_entropy_with_logits(s, F['y'], reduction='none') * F['w']).sum() / F['w'].sum()


def trust_score(model, F):
    """antisymmetric decision score for 'trust': + => choose a"""
    with torch.no_grad():
        sg = torch.sign(F['dlp']); sg[sg == 0] = 1
        h = F['hn'][:, model.layers]
        t = model.c[0] + model.c[1] * F['dlp'].abs() + torch.einsum('nld,ld->n', h, model.w)
        return sg * t          # t>0: keep higher-prob; t<0: flip


def train(kind, layers, F, l2, r=16, epochs=300, lr=3e-3, val=None, seed=0):
    torch.manual_seed(seed)
    model = Model(kind, layers, r)
    if kind in ('had', 'trust', 'base'):
        opt = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=200, history_size=20, line_search_fn='strong_wolfe')
        def closure():
            opt.zero_grad(); L = loss_of(model, F) + l2 * model.reg(); L.backward(); return L
        opt.step(closure)
        return model
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = (1e9, None)
    for ep in range(epochs):
        opt.zero_grad(); L = loss_of(model, F) + l2 * model.reg(); L.backward(); opt.step()
        if val is not None and ep % 10 == 9:
            with torch.no_grad(): vl = float(loss_of(model, val))
            if vl < best[0]: best = (vl, {k: v.clone() for k, v in model.state_dict().items()}, ep)
    if val is not None and best[1] is not None: model.load_state_dict(best[1]); model.best_ep = best[2]
    return model


def score(model, F):
    if model.kind == 'trust': return trust_score(model, F).numpy()
    with torch.no_grad(): return model(F).numpy()


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    o = np.argsort(s, kind='mergesort'); rk = np.empty(len(s)); rk[o] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, rk); rk = (sums / cnt)[inv]
    n1 = y.sum(); n0 = len(y) - n1
    return float((rk[y == 1].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))


def metrics(s, F, P=None):
    """pairwise accuracy (choose the y=1 token), weighted; abstain curve (top-q most confident)"""
    y = F['y'].numpy(); w = F['w'].numpy()
    corr = ((s > 0) == (y > 0.5)) * 1.0; corr[s == 0] = 0.5
    out = dict(n=int(len(y)), acc=float((corr * w).sum() / w.sum()))
    # AUROC: decision score oriented toward the 'right' token (pairs are stored with a = right for branch pairs)
    sr = np.where(y > 0.5, s, -s)
    out['auc_sym'] = auroc(np.r_[np.ones(len(sr)), np.zeros(len(sr))], np.r_[sr, -sr])
    o = np.argsort(-np.abs(s))
    for q in (0.1, 0.25, 0.5):
        k = max(1, int(q * len(s))); out[f'acc@{int(q * 100)}%'] = float(corr[o[:k]].mean())
    return out


def evaluate(models, d, which='pairs'):
    P = d[which]; F = feats(d, P)
    res = {}
    for name, mdl in models.items():
        s = score(mdl, F) if mdl is not None else F['dlp'].numpy()
        r = metrics(s, F); res[name] = r
        if which == 'pairs':
            nw = np.array([p['n_wrong'] for p in P], float); y = F['y'].numpy()
            c = ((s > 0) == (y > 0.5)) * 1.0; c[s == 0] = 0.5
            r['acc_per_wrong_att'] = float((c * nw).sum() / nw.sum())
            by = collections.defaultdict(list)
            for i, p in enumerate(P): by['cat:' + p['cat']].append(i); by['vs:' + str(p['vs'])].append(i)
            ad = np.abs(F['dlp'].numpy())
            for i, x in enumerate(ad):
                by['|dlp|' + ('<0.5' if x < 0.5 else '0.5-1.5' if x < 1.5 else '1.5-3' if x < 3 else '>=3')].append(i)
                by['top2' if P[i]['top2'] else 'not_top2'].append(i)
            for k, ix in by.items():
                ix = np.array(ix); y = F['y'].numpy()[ix]; ss = s[ix]
                c = ((ss > 0) == (y > 0.5)) * 1.0; c[ss == 0] = 0.5
                r[k] = (int(len(ix)), round(float(c.mean()), 3))
    return res, F


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--train', default='trainC'); ap.add_argument('--test', default='dev,dev2')
    ap.add_argument('--tag', default='trainC'); ap.add_argument('--trainon', default='pairs,forks')
    a = ap.parse_args()
    tr_sets = a.train.split(','); te_sets = a.test.split(',')
    D = {S: load_set(S) for S in set(tr_sets + te_sets)}
    for S, d in D.items():
        F = feats(d, d['pairs']); c, md = check_host(F)
        print(S, 'pairs', len(d['pairs']), 'fork pairs', len(d['fpairs']), '| host check corr', round(c, 4), 'med|diff|', round(md, 3),
              '| tf vs sampled lp mae', round(float(np.nanmean([abs(p['lpa'] - p['slpa']) + abs(p['lpb'] - p['slpb']) for p in d['pairs']]) / 2), 3),
              '| top2 share', round(np.mean([p['top2'] for p in d['pairs']]), 3), flush=True)
    results = {}
    for trainon in a.trainon.split(','):
        TP = []
        for S in tr_sets:
            d = D[S]; PP = d['pairs'] if trainon == 'pairs' else d['fpairs']
            TP.append((d, PP))
        # concat training features
        Fs = [feats(d, PP) for d, PP in TP]
        Ftr = {k: (torch.cat([F[k] for F in Fs]) if isinstance(Fs[0][k], torch.Tensor) else np.concatenate([F[k] for F in Fs])) for k in Fs[0]}
        # pairs are stored oriented (a = right); randomize orientation so the model can't learn the side
        rng = np.random.default_rng(0); flip = torch.tensor(rng.random(len(Ftr['y'])) < 0.5)
        Ftr['du'][flip] *= -1; Ftr['dlp'][flip] *= -1; Ftr['y'][flip] = 1 - Ftr['y'][flip]
        g = Ftr['g']; ug = np.unique(g); rng.shuffle(ug); hold = np.isin(g, ug[:len(ug) // 5])
        Fa, Fv = sub(Ftr, torch.tensor(~hold)), sub(Ftr, torch.tensor(hold))
        cfgs = [('base', [4], 0)]
        for L in (1, 2, 3, 4):
            for l2 in (1e-3, 1e-2, 1e-1): cfgs.append(('had', [L], l2))
        for l2 in (1e-2, 1e-1): cfgs.append(('had', [0, 1, 2, 3, 4], l2))
        for L in (1, 2, 4):
            for l2 in (1e-2, 1e-1): cfgs.append(('trust', [L], l2))
        for r in (8, 32):
            for l2 in (1e-3, 1e-2): cfgs.append(('lr', [1, 2, 4], l2, r))
        sel = {}
        for c in cfgs:
            kind, layers, l2 = c[:3]; r = c[3] if len(c) > 3 else 16
            m = train(kind, layers, Fa, l2, r=r, val=Fv)
            vs = score(m, Fv); vm = metrics(vs, Fv)
            name = f"{kind}_{'+'.join(str(LAYS[i]) for i in layers)}" + (f'_r{r}' if kind == 'lr' else '')
            print(f'[{trainon}] {name} l2={l2} holdout acc {vm["acc"]:.3f} acc@25% {vm["acc@25%"]:.3f}', flush=True)
            if name not in sel or vm['acc'] > sel[name][0]: sel[name] = (vm['acc'], c)
        models = {'host_argmax': None}
        for name, (_, c) in sel.items():
            kind, layers, l2 = c[:3]; r = c[3] if len(c) > 3 else 16
            if kind == 'lr': models[name] = train(kind, layers, Fa, l2, r=r, val=Fv)
            else: models[name] = train(kind, layers, Ftr, l2, r=r)
            models[name].cfg = c
        for S in te_sets:
            for which in ('pairs', 'fpairs'):
                res, _ = evaluate(models, D[S], which)
                results[f'{trainon}|{S}|{which}'] = res
                print(f'== train {trainon} -> {S} {which}')
                for nm, r_ in res.items():
                    print(f'  {nm:18s} n {r_["n"]} acc {r_["acc"]:.3f} accW {r_.get("acc_per_wrong_att", float("nan")):.3f} '
                          f'auc {r_["auc_sym"]:.3f} acc@10% {r_["acc@10%"]:.3f} '
                          f'@25% {r_["acc@25%"]:.3f} @50% {r_["acc@50%"]:.3f}', flush=True)
        torch.save({k: (v.state_dict(), v.cfg) for k, v in models.items() if v is not None}, f'{HERE}/out_models_{a.tag}_{trainon}.pt')
    json.dump(results, open(f'{HERE}/fit_{a.tag}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
