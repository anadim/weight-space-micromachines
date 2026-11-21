"""selcheck step-1 probe UPPER BOUND (no trainC needed): per set (dev, dev2) item-grouped 5-fold OOF logistic probes on the
check-format hidden states (same probe/l2 grid as fit.py; l2 picked by OOF within-item AUROC on the set itself = optimistic).
Out: out/fzo_{set}.json = {o_{fmt}_{layerset}: per-candidate score (list of K for sampled formats)}."""
import os, sys, json, torch, random
HERE = os.path.dirname(os.path.abspath(__file__))
HD = sys.argv[1] if len(sys.argv) > 1 else '$WSM_DATA/implants/selcheck'
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(16)
LS = {'L18': [18], 'c4': [14, 18, 22, 28]}
L2 = [1e-2, 1e-1, 1.0]
exec(open(f'{HERE}/fit.py').read().split('def load_hs')[0].split('L2 = ')[1].split('\n', 1)[1])   # within_auc, fit
for S in ('dev', 'dev2'):
    D = torch.load(f'{HD}/hs_{S}.pt'); LAY = D['LAY']; it = D['idx']; ok = D['ok']
    yt = torch.tensor(ok, dtype=torch.float32); out = {}
    items = sorted(set(it)); random.Random(0).shuffle(items); fold = {i: j % 5 for j, i in enumerate(items)}
    fo = torch.tensor([fold[i] for i in it])
    for key, H in D['hs'].items():
        if H.dim() == 3: H = H[None]
        for ln, ls in LS.items():
            li = [LAY.index(L) for L in ls]; K, _, n, Dm = H.shape
            X = H[:, li].permute(0, 2, 1, 3).reshape(K, n, len(li) * Dm).float()
            mu = X.reshape(-1, X.shape[-1]).mean(0); sd = X.reshape(-1, X.shape[-1]).std(0) + 1e-3
            Z = ((X - mu) / sd).to(dev); best = None
            for l2 in L2:
                oof = torch.zeros(K, n)
                for f in range(5):
                    tr = (fo != f); te = (fo == f)
                    w, b = fit(Z[:, tr].reshape(-1, Z.shape[-1]), yt[tr].repeat(K).to(dev), l2, it=100)
                    oof[:, te] = (Z[:, te] @ w + b).cpu()
                a = within_auc(oof.mean(0).tolist(), ok, it)
                if best is None or a > best[0]: best = (a, l2, oof)
            a, l2, oof = best; nm = f'o_{key}_{ln}'
            out[nm] = oof.t().tolist() if K > 1 else oof[0].tolist()
            a1 = sum(within_auc(oof[k].tolist(), ok, it) for k in range(K)) / K
            print(S, nm, 'l2', l2, 'oof within-item AUROC Kmean %.3f 1-sample %.3f' % (a, a1), flush=True)
    os.makedirs(f'{HERE}/out', exist_ok=True); json.dump(out, open(f'{HERE}/out/fzo_{S}.json', 'w'))
print('XFIT DONE')
