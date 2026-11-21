"""selcheck step-1 probes: linear probes on the check-format hidden states (hs_{set}.pt from fz.py), fit on trainC only.
For each format key and layer set: standardise with trainC stats, L2-logistic (full-batch LBFGS, torch), l2 chosen by
item-grouped 5-fold OOF within-item AUROC on trainC; full fit applied to dev/dev2.
Out: out/fzp_{set}.json = {key: [score per candidate (list of K for sampled formats)]}, out/fzp_summary.json."""
import os, sys, json, torch, random
HERE = os.path.dirname(os.path.abspath(__file__))
HD = sys.argv[1] if len(sys.argv) > 1 else '$WSM_DATA/implants/selcheck'
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
LS = {'L14': [14], 'L18': [18], 'L22': [22], 'c4': [14, 18, 22, 28]}
L2 = [1e-3, 1e-2, 1e-1, 1.0]


def within_auc(s, ok, idx):
    by = {}
    for v, o, i in zip(s, ok, idx): by.setdefault(i, ([], []))[0 if o else 1].append(v)
    w = n = 0.0
    for r, wr in by.values():
        for a in r:
            for b in wr: w += (a > b) + 0.5 * (a == b); n += 1
    return w / max(n, 1)


def fit(X, y, l2, it=200):
    w = torch.zeros(X.shape[1], device=dev, requires_grad=True); b = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=it, line_search_fn='strong_wolfe')
    def clo():
        opt.zero_grad()
        lo = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y) + l2 * (w * w).sum() / len(y) * 100
        lo.backward(); return lo
    opt.step(clo)
    return w.detach(), b.detach()


def load_hs(S):
    if os.path.exists(f'{HD}/hs_{S}.pt'): return torch.load(f'{HD}/hs_{S}.pt')
    P = [torch.load(f'{HD}/hs_{S}_p{i}.pt') for i in range(2)]          # fz.py --part i/2 --tag _p{i}, concatenated p0, p1
    cat = lambda a, b: torch.cat([a, b], dim=a.dim() - 2)              # candidate axis: (nL, n, D) or (K, nL, n, D)
    return dict(hs={k: cat(P[0]['hs'][k], P[1]['hs'][k]) for k in P[0]['hs']}, LAY=P[0]['LAY'],
                idx=P[0]['idx'] + P[1]['idx'], ok=P[0]['ok'] + P[1]['ok'])


D = {S: load_hs(S) for S in ('trainC', 'dev', 'dev2')}
LAY = D['trainC']['LAY']
summ = {}; outs = {S: {} for S in ('trainC', 'dev', 'dev2')}
keys = list(D['trainC']['hs'])
for key in keys:
    for ln, ls in LS.items():
        li = [LAY.index(L) for L in ls]
        def feats(S):
            H = D[S]['hs'][key]
            if H.dim() == 3: H = H[None]                      # (K, nL, n, D)
            K, _, n, Dm = H.shape
            X = H[:, li].permute(0, 2, 1, 3).reshape(K, n, len(li) * Dm).float()
            return X                                          # (K, n, F)
        Xt = feats('trainC'); K = Xt.shape[0]
        yt = torch.tensor(D['trainC']['ok'], dtype=torch.float32); it = D['trainC']['idx']
        mu = Xt.reshape(-1, Xt.shape[-1]).mean(0); sd = Xt.reshape(-1, Xt.shape[-1]).std(0) + 1e-3
        Z = ((Xt - mu) / sd).to(dev)
        items = sorted(set(it)); random.Random(0).shuffle(items); fold = {i: j % 5 for j, i in enumerate(items)}
        fo = torch.tensor([fold[i] for i in it])
        best = None
        for l2 in L2:
            oof = torch.zeros(K, len(it))
            for f in range(5):
                tr = (fo != f); te = (fo == f)
                w, b = fit(Z[:, tr].reshape(-1, Z.shape[-1]), yt[tr].repeat(K).to(dev), l2)
                oof[:, te] = (Z[:, te] @ w + b).cpu()
            a = within_auc(oof.mean(0).tolist(), D['trainC']['ok'], it)
            a1 = sum(within_auc(oof[k].tolist(), D['trainC']['ok'], it) for k in range(K)) / K
            if best is None or a > best[0]: best = (a, a1, l2, oof)
        a, a1, l2, oof = best
        w, b = fit(Z.reshape(-1, Z.shape[-1]), yt.repeat(K).to(dev), l2)
        nm = f'p_{key}_{ln}'
        outs['trainC'][nm] = oof.t().tolist() if K > 1 else oof[0].tolist()
        rec = dict(l2=l2, trainC_oof_within=a, trainC_oof_within_1sample=a1)
        for S in ('dev', 'dev2'):
            Xs = feats(S); s = (((Xs - mu) / sd).to(dev) @ w + b).cpu()
            outs[S][nm] = s.t().tolist() if K > 1 else s[0].tolist()
            rec[S + '_within'] = within_auc(s.mean(0).tolist(), D[S]['ok'], D[S]['idx'])
            rec[S + '_within_1sample'] = sum(within_auc(s[k].tolist(), D[S]['ok'], D[S]['idx']) for k in range(K)) / K
        summ[nm] = rec
        print(nm, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)
os.makedirs(f'{HERE}/out', exist_ok=True)
for S in outs: json.dump(outs[S], open(f'{HERE}/out/fzp_{S}.json', 'w'))
json.dump(summ, open(f'{HERE}/out/fzp_summary.json', 'w'), indent=1)
print('FIT DONE')
