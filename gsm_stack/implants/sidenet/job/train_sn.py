"""Side-net training (lane sidenet). Runs on the node (GPU) over the extract_sn.py features.
Train on trainC only (item-grouped 5-fold CV, fold = idx % 5); dev/dev2 scored by the mean of the 5 fold models (held out).
Features: concat over a layer set of H ('####' state) and/or M (line-end mean); standardized with trainC stats.
Models: lin (logistic) / mlp (d -> hid -> 1, GELU, dropout). Losses: bce | pair (bce/2 + within-item pairwise logistic) |
list (bce/2 + listwise softmax over answer groups of each stream, group score = logsumexp of member scores).
Metrics: pooled AUROC and within-item AUROC (right vs wrong attempts of the same item, all seeds) on committed attempts
(has '####' and register lg set), vs the register lg. Snapshots at several epochs; the epoch is chosen on trainC OOF only.
Out: out/train_{tag}.json (metrics), out/sc_{tag}.npz (per-config OOF trainC + dev/dev2 scores, float32)."""
import os, sys, json, time, argparse, itertools
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='$WSM_DATA/implants/sidenet')
ap.add_argument('--grid', default='c4:HM:mlp:bce,c4:HM:mlp:pair,c4:HM:mlp:list,c4:HM:lin:bce,c4:HM:lin:pair')
ap.add_argument('--epochs', default='2,4,8,16')
ap.add_argument('--hid', type=int, default=256)
ap.add_argument('--drop', type=float, default=0.3)
ap.add_argument('--idrop', type=float, default=0.1)
ap.add_argument('--wd', type=float, default=1e-2)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--bitems', type=int, default=64)
ap.add_argument('--tag', default='g1')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--save_w', type=int, default=0, help='1: lin models -> out/w_{tag}_{cfg}.pt (fold-mean folded vector)')
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
LSETS = {'c4': [14, 18, 22, 27], 'c8': [12, 14, 16, 18, 20, 22, 24, 27], 'all': list(range(10, 29)), 'l18': [18],
         'c6': [14, 16, 18, 20, 22, 27], 'mid': [16, 17, 18, 19, 20, 21, 22]}
EP = [int(x) for x in a.epochs.split(',')]
SETS = ('trainC', 'dev', 'dev2')
META = {S: dict(np.load(f'{a.root}/{S}/meta.npz')) for S in SETS}
AUX = {S: dict(np.load(f'{HERE}/tables/aux_{S}.npz')) for S in SETS}          # grp (stream-local answer group, -1 none), sid
for S in SETS:
    assert len(AUX[S]['grp']) == len(META[S]['idx']) and (AUX[S]['idx'] == META[S]['idx']).all() and (AUX[S]['k'] == META[S]['k']).all()
LAYS = list(META['trainC']['layers'])
_cache = {}


def feats(S, lset, kinds):
    key = (S, lset, kinds)
    if key in _cache: return _cache[key]
    parts = []
    for kd in kinds:
        A = np.load(f'{a.root}/{S}/{kd}.npy', mmap_mode='r')
        for L in LSETS[lset]: parts.append(torch.from_numpy(np.ascontiguousarray(A[LAYS.index(L)])))
    X = torch.cat(parts, 1).to(dev)                   # fp16 (n, d)
    _cache[key] = X
    return X


def auroc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    if y.all() or (~y).all(): return float('nan')
    # average ranks for ties
    order = np.argsort(s, kind='mergesort'); ss = s[order]; rk = np.empty(len(s)); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]: j += 1
        rk[order[i:j + 1]] = (i + j) / 2; i = j + 1
    n1 = y.sum(); n0 = (~y).sum()
    return float((rk[y].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def within(idx, y, s, mask):
    """pair-weighted within-item AUROC over (right, wrong) pairs of the same item"""
    num = 0.0; den = 0
    for it in np.unique(idx[mask]):
        m = mask & (idx == it); yy = y[m].astype(bool); ss = s[m]
        if yy.all() or (~yy).all(): continue
        d = ss[yy][:, None] - ss[~yy][None]
        num += (d > 0).sum() + 0.5 * (d == 0).sum(); den += d.size
    return float(num / den) if den else float('nan')


def metr(S, s):
    M = META[S]; cm = (M['has_hash'] > 0) & (M['lg'] > -29.5)
    return dict(pooled=auroc(M['ok'][cm], s[cm]), within=within(M['idx'], M['ok'], s, cm))


class Net(nn.Module):
    def __init__(self, d, kind):
        super().__init__()
        self.kind = kind; self.idr = nn.Dropout(a.idrop)
        if kind == 'lin': self.f = nn.Linear(d, 1)
        else: self.f = nn.Sequential(nn.Linear(d, a.hid), nn.GELU(), nn.Dropout(a.drop), nn.Linear(a.hid, 1))
        if kind == 'mlpr': self.g = nn.Linear(d, 1)          # wide & deep: linear skip + MLP

    def forward(self, x):
        x = self.idr(x); o = self.f(x)
        if self.kind == 'mlpr': o = o + self.g(x)
        return o.squeeze(-1)


def loss_fn(kind, s, y, item, stream, grp):
    yf = y.float(); L = F.binary_cross_entropy_with_logits(s, yf)
    if kind == 'bce': return L
    if kind == 'pair':
        same = item[:, None] == item[None]; pm = same & (y[:, None] > 0) & (y[None] == 0)
        if pm.any():
            d = s[:, None] - s[None]
            return 0.5 * L + F.softplus(-d[pm]).mean()
        return L
    if kind == 'list':
        v = grp >= 0
        if not v.any(): return L
        key = stream[v] * 64 + grp[v]; uk, inv = torch.unique(key, return_inverse=True)
        sv = s[v]; G = len(uk)
        mx = torch.full((G,), -1e9, device=s.device).scatter_reduce(0, inv, sv, 'amax')
        gs = mx + torch.log(torch.zeros(G, device=s.device).index_add(0, inv, torch.exp(sv - mx[inv])))
        gy = torch.zeros(G, device=s.device).index_add(0, inv, yf[v]) > 0
        gst = uk // 64; us, sinv = torch.unique(gst, return_inverse=True)
        ns = len(us)
        smx = torch.full((ns,), -1e9, device=s.device).scatter_reduce(0, sinv, gs, 'amax')
        lse = smx + torch.log(torch.zeros(ns, device=s.device).index_add(0, sinv, torch.exp(gs - smx[sinv])))
        ng = torch.zeros(ns, device=s.device).index_add(0, sinv, torch.ones(G, device=s.device))
        has = torch.zeros(ns, device=s.device).index_add(0, sinv, gy.float())
        gold = torch.full((ns,), 0.0, device=s.device).index_add(0, sinv, torch.where(gy, gs, torch.zeros_like(gs)))
        ok = (ng > 1) & (has > 0)
        if ok.any(): return 0.5 * L + (lse - gold)[ok].mean()
        return L
    raise ValueError(kind)


def run_cfg(cfg):
    lset, kinds_s, model, lossk = cfg.split(':')
    kinds = tuple(kinds_s)                        # chars: H '####', M line-end mean, A last seg token, E last line end
    Xt = feats('trainC', lset, kinds); d = Xt.shape[1]
    mu = Xt.float().mean(0); sd = Xt.float().std(0) + 1e-3
    Mt = META['trainC']; idx = Mt['idx']; y = torch.tensor(Mt['ok'], device=dev).long()
    item_t = torch.tensor(idx, device=dev).long(); stream_t = torch.tensor(AUX['trainC']['sid'], device=dev).long()
    grp_t = torch.tensor(AUX['trainC']['grp'], device=dev).long()
    fold = idx % 5
    oof = {e: np.zeros(len(idx), np.float32) for e in EP}
    WS = {e: [] for e in EP}
    evs = {S: {e: np.zeros(len(META[S]['idx']), np.float32) for e in EP} for S in ('dev', 'dev2')}
    XE = {S: feats(S, lset, kinds) for S in ('dev', 'dev2')}
    t0 = time.time()

    def score(net, X):
        net.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(X), 8192): out.append(net(((X[i:i + 8192].float() - mu) / sd)))
        net.train(); return torch.cat(out).cpu().numpy()
    for k in range(5):
        torch.manual_seed(a.seed * 100 + k); rng = np.random.default_rng(a.seed * 100 + k)
        tr_items = np.unique(idx[fold != k]); by = {}
        for i, it in enumerate(idx):
            if fold[i] != k: by.setdefault(it, []).append(i)
        net = Net(d, model).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
        nb = int(np.ceil(len(tr_items) / a.bitems)); total = max(EP) * nb
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.1)
        for ep in range(1, max(EP) + 1):
            perm = rng.permutation(tr_items)
            for b in range(nb):
                ii = torch.tensor(np.concatenate([by[it] for it in perm[b * a.bitems:(b + 1) * a.bitems]]), device=dev)
                x = (Xt[ii].float() - mu) / sd
                s = net(x)
                L = loss_fn(lossk, s, y[ii], item_t[ii], stream_t[ii], grp_t[ii])
                opt.zero_grad(); L.backward(); opt.step(); sch.step()
            if ep in EP:
                te = np.where(fold == k)[0]
                oof[ep][te] = score(net, Xt[torch.tensor(te, device=dev)])
                for S in ('dev', 'dev2'): evs[S][ep] += score(net, XE[S]) / 5
                if model == 'lin': WS[ep].append((net.f.weight.detach().double()[0].clone(), float(net.f.bias.detach().double()[0])))
    res = dict(cfg=cfg, d=int(d), secs=round(time.time() - t0, 1), ep={})
    for e in EP:
        res['ep'][e] = dict(trainC=metr('trainC', oof[e]), dev=metr('dev', evs['dev'][e]), dev2=metr('dev2', evs['dev2'][e]))
    best = max(EP, key=lambda e: res['ep'][e]['trainC']['within'])
    res['best_ep'] = best
    if a.save_w and model == 'lin':
        # the 5 fold models are linear in the standardized features: their mean is ONE fixed linear map on the raw features
        wz = torch.stack([w for w, _ in WS[best]]).mean(0); bz = float(np.mean([b for _, b in WS[best]]))
        w = wz / sd.double(); b = bz - float((w * mu.double()).sum())
        chk = {}
        for S in ('dev', 'dev2'):
            v = (XE[S].double() @ w + b).float().cpu().numpy(); chk[S] = float(np.abs(v - evs[S][best]).max())
        print('folded fold-mean vector: max |score - fold-mean score|', chk, flush=True)
        torch.save(dict(w=w.float().cpu(), b=b, layers=LSETS[lset], kinds=kinds_s, cfg=cfg, epoch=best, d=int(d), D=1024,
                        layout='concat over kinds (H then M), each concat over layers of the layer-input state (1024)',
                        mu=mu.cpu(), sd=sd.cpu(), wz_folds=torch.stack([w for w, _ in WS[best]]).float().cpu(),
                        bz_folds=[b for _, b in WS[best]], check=chk), f'{HERE}/out/w_{a.tag}_{cfg.replace(":", "_")}.pt')
    print(f"{cfg:22s} d={d} {res['secs']}s | " + ' | '.join(
        f"ep{e}{'*' if e == best else ''} tC {res['ep'][e]['trainC']['within']:.3f} dev {res['ep'][e]['dev']['pooled']:.3f}/{res['ep'][e]['dev']['within']:.3f} "
        f"dev2 {res['ep'][e]['dev2']['pooled']:.3f}/{res['ep'][e]['dev2']['within']:.3f}" for e in EP), flush=True)
    return res, {f'{cfg}|trainC': oof[best], f'{cfg}|dev': evs['dev'][best], f'{cfg}|dev2': evs['dev2'][best]}


base = {S: dict(lg=metr(S, np.nan_to_num(META[S]['lg'], nan=-30)), lg_ours=metr(S, np.nan_to_num(META[S]['lg_ours'], nan=-30)))
        for S in SETS}
print('baselines (pooled/within):', json.dumps(base), flush=True)
R = dict(args=vars(a), base=base, cfgs=[]); SC = {}
os.makedirs(f'{HERE}/out', exist_ok=True)
for cfg in a.grid.split(','):
    r, sc = run_cfg(cfg); R['cfgs'].append(r); SC.update(sc)
    json.dump(R, open(f'{HERE}/out/train_{a.tag}.json', 'w'), indent=1)
    np.savez_compressed(f'{HERE}/out/sc_{a.tag}.npz', **SC)
print('TRAIN DONE', flush=True)
