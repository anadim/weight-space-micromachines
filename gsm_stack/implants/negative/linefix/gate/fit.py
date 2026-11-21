"""Loader + question-grouped logistic probes over the hidden states written by extract.py (runs on the node).

    import fit
    X, meta = fit.load('dev', layers=[14, 18, 22], kind='hash')     # X: (N, 3, 1024) float16 numpy
    y = meta['ok']; g = meta['idx']                                  # labels, question groups
    oof, auc = fit.grouped_cv_probe(X[:, 1], y, g)                   # 5-fold question-grouped L2 logistic probe

kind: 'hash' (token completing '####'), 'nl' (each newline token of the attempt's visible segment; meta['j'] =
newline ordinal, meta['off'] = token offset in seg_tokens), 'last' (last token of the segment), 'all' (all three),
'qend' (last question token, one row per item; meta has idx, and item-level fields if items_{set}.jsonl.gz exists).
meta: dict of numpy arrays, one entry per row: position fields (att, kind, j, off, pos, tok) and the attempt's fields
(idx, seed, k, ok, lg, bad, nrep, hash_off, seg_len, gid; plus 'cls', 'n_right' from the items table if present).
att indexes attempts_light.jsonl.gz / attempts.npz (same order as the atlas attempts table, after --limit_items).

CLI demo:  python3 fit.py --set dev --kind hash --layers 14,18,22,27 [--target ok]
"""
import os, json, gzip, argparse
import numpy as np
import torch

ROOT = os.environ.get('HS_ROOT', '$WSM_DATA/implants/hs')
KIND = {'nl': 1, 'hash': 2, 'last': 3}


def info(set_, root=ROOT):
    return json.load(open(f'{root}/{set_}/info.json'))


def _items(d):
    p = [f for f in os.listdir(d) if f.startswith('items_') and f.endswith('.jsonl.gz')]
    if not p: return None
    with gzip.open(f'{d}/{p[0]}', 'rt') as f: rows = [json.loads(l) for l in f]
    return {r['idx']: r for r in rows}


def load(set_, layers=(18,), kind='hash', root=ROOT, rows=None):
    """-> X (N, len(layers), D) float16, meta (dict of (N,) arrays). rows: optional bool mask / index array applied to
    the position rows of `kind` before reading (to load a subset)."""
    d = f'{root}/{set_}'; inf = info(set_, root)
    li = [inf['layers'].index(L) for L in layers]
    A = dict(np.load(f'{d}/attempts.npz'))
    it = _items(d)
    if kind == 'qend':
        qm = dict(np.load(f'{d}/qend_meta.npz'))
        arr = np.load(f'{d}/qend.npy', mmap_mode='r')
        sel = np.arange(len(qm['idx'])) if rows is None else np.arange(len(qm['idx']))[rows]
        X = np.stack([np.asarray(arr[l][sel]) for l in li], 1)
        meta = {k: v[sel] for k, v in qm.items()}
        idx = meta['idx']
        okm = {i: A['ok'][A['idx'] == i].mean() for i in np.unique(idx)}
        meta['p_right'] = np.array([okm[i] for i in idx], np.float32)     # fraction of this item's attempts that are right
    else:
        pm = dict(np.load(f'{d}/pos_meta.npz'))
        m = np.isin(pm['kind'], [1, 2, 3]) if kind == 'all' else pm['kind'] == KIND[kind]
        sel = np.nonzero(m)[0]
        if rows is not None: sel = sel[rows]
        meta = {k: v[sel] for k, v in pm.items()}
        X = np.empty((len(sel), len(li), inf['D']), np.float16)
        for s, sh in enumerate(inf['shards']):
            w = np.nonzero(meta['shard'] == s)[0]
            if not len(w): continue
            arr = np.load(f"{d}/{sh['file']}", mmap_mode='r')
            r = meta['row'][w]; o = np.argsort(r)
            for c, l in enumerate(li):
                X[w[o], c] = arr[l][r[o]]
        for k, v in A.items(): meta[k] = v[meta['att']]
    if it is not None:
        meta['cls'] = np.array([it.get(int(i), {}).get('class', '') for i in meta['idx']])
        meta['n_right'] = np.array([it.get(int(i), {}).get('n_right', -1) for i in meta['idx']])
    return X, meta


def auroc(y, s):
    y = np.asarray(y).astype(bool); s = np.asarray(s, float)
    if y.all() or (~y).all(): return float('nan')
    r = np.empty(len(s)); o = np.argsort(s, kind='mergesort'); r[o] = np.arange(1, len(s) + 1)
    for v in np.unique(s[o]):                                        # average ranks of ties
        t = s == v
        if t.sum() > 1: r[t] = r[t].mean()
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_logreg(X, y, l2=1e-2, w=None, iters=200, dev=None):
    """L2 logistic regression (torch LBFGS, standardized inputs). l2 = penalty on mean loss (lambda * |w|^2 / 2).
    Returns predict(Xnew) -> logits (numpy) and the params in input units (w, b)."""
    dev = dev or ('cuda' if torch.cuda.is_available() else 'cpu')
    Xt = torch.as_tensor(np.asarray(X, np.float32), device=dev); yt = torch.as_tensor(np.asarray(y, np.float32), device=dev)
    mu = Xt.mean(0); sd = Xt.std(0) + 1e-4; Z = (Xt - mu) / sd
    sw = torch.ones_like(yt) if w is None else torch.as_tensor(np.asarray(w, np.float32), device=dev)
    sw = sw / sw.mean()
    W = torch.zeros(Z.shape[1], device=dev, requires_grad=True); b = torch.zeros((), device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, history_size=20, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(Z @ W + b, yt, reduction='none') * sw).mean() + 0.5 * l2 * (W * W).sum()
        loss.backward(); return loss
    opt.step(closure)
    Wi = (W / sd).detach(); bi = (b - (mu / sd) @ W).detach()
    def predict(Xn):
        return (torch.as_tensor(np.asarray(Xn, np.float32), device=dev) @ Wi + bi).cpu().numpy()
    return predict, Wi.cpu(), float(bi)


def grouped_cv_probe(X, y, groups, folds=5, l2=1e-2, seed=0, **kw):
    """Question-grouped K-fold: all rows of a question are in one fold. X (N, D) or (N, nL, D) (flattened).
    -> out-of-fold logits (N,), AUROC."""
    X = np.asarray(X).reshape(len(X), -1)
    ug = np.unique(groups); rng = np.random.default_rng(seed); rng.shuffle(ug)
    fold_of = {g: i % folds for i, g in enumerate(ug)}
    f = np.array([fold_of[g] for g in groups]); oof = np.zeros(len(y), np.float32)
    for k in range(folds):
        tr, te = f != k, f == k
        pred, _, _ = fit_logreg(X[tr], np.asarray(y)[tr], l2=l2, **kw)
        oof[te] = pred(X[te])
    return oof, auroc(y, oof)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='dev'); ap.add_argument('--kind', default='hash')
    ap.add_argument('--layers', default='14,18,22,27'); ap.add_argument('--target', default='ok')
    ap.add_argument('--l2', type=float, default=1e-2); ap.add_argument('--root', default=ROOT)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    Ls = [int(x) for x in a.layers.split(',')]
    import time; t0 = time.time()
    X, meta = load(a.set, Ls, a.kind, root=a.root)
    print(f'loaded {a.set} {a.kind} X{X.shape} in {time.time() - t0:.1f}s', flush=True)
    y = meta[a.target].astype(np.float32) if a.kind != 'qend' else (meta['p_right'] >= 0.5).astype(np.float32)
    res = dict(set=a.set, kind=a.kind, n=int(len(y)), pos_rate=float(y.mean()), auc={})
    if a.kind != 'qend' and 'lg' in meta:
        ok = np.isfinite(meta['lg']) & (meta['lg'] > -29.5)
        res['auc_register_lg'] = auroc(y[ok], meta['lg'][ok])
    for c, L in enumerate(Ls):
        _, auc = grouped_cv_probe(X[:, c], y, meta['idx'], l2=a.l2)
        res['auc'][L] = round(auc, 4); print(f'layer {L}: grouped-CV AUROC {auc:.4f}', flush=True)
    if len(Ls) > 1:
        _, auc = grouped_cv_probe(X, y, meta['idx'], l2=a.l2); res['auc']['concat'] = round(auc, 4)
        print(f'concat {Ls}: AUROC {auc:.4f}', flush=True)
    print(json.dumps(res))
    if a.out: json.dump(res, open(a.out, 'w'), indent=1)
