"""lane seqver: SEQUENCE side-net verifier over all body-token states (extract_sq.py features). Node GPU.
Token features f_t = [z(x_t) over a layer set (PCA r dims per layer) | z(channels)] standardized with trainC stats.
Channels: lp_row (sampled logp), ent_prev, ent_next, sH, sM (sidenet per-token projections), isnl, calc (wf&1), relpos.
Pooling terms (cfg POOL chars): aK = K attention-pooling heads  Σ_t softmax_t(q_h.f_t + c_h) (v_h.f_t);  m = max-badness
alpha*max_t(w.f_t);  u = token mean u.f_t;  h = linear on the H token (last body token = '####') and on the mean of line-end
tokens (projected-space analogue of sn). MODE alone: s = Σ terms + c, loss 0.5 bce + pair; inc: s = b + Σ terms (b frozen =
max(sn,-12)+max(lg,-12), sn = g4 OOF/fold-mean as in attnver), loss pair; output vectors zero-init so inc starts at b.
cfg = LSET[rR]:FEATS:POOL:MODE  e.g. c4:x:a4:inc, c4r64:xc:a4mh:alone, -:c:a2mu:inc  (FEATS x = states, c = channels).
Committed attempts only (has '####', lg > -29.5). trainC 5-fold item CV (idx % 5) -> OOF; dev/dev2 = 5-fold-model mean.
Epoch chosen on trainC OOF within-item AUROC. Out: out/train_{tag}.json, out/sc_{tag}.npz (cfg|S -> score; for inc: the delta
s - b, so that repick b' = b + delta)."""
import os, sys, json, time, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ap = argparse.ArgumentParser()
ap.add_argument('--root', default='$WSM_DATA/implants/seqver')
ap.add_argument('--grid', default='c4:x:a4:inc')
ap.add_argument('--epochs', default='1,2,4,8,16')
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--wd', type=float, default=1e-2)
ap.add_argument('--idrop', type=float, default=0.1)
ap.add_argument('--bitems', type=int, default=64)
ap.add_argument('--tag', default='q1')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--save_w', type=int, default=0)
ap.add_argument('--perfold', type=int, default=0, help='1: also store fold-0 single-model dev/dev2 scores (cfg#f0)')
ap.add_argument('--tmax', type=float, default=1800, help='do not start a new cfg after this many seconds')
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
SETS = ('trainC', 'dev', 'dev2'); EP = [int(x) for x in a.epochs.split(',')]
LSETS = {'c4': [14, 18, 22, 27], 'l1827': [18, 27], 'l27': [27], 'l18': [18], '-': []}
SN = dict(np.load(f'{HERE}/tables/sc_g4.npz'))
CH = ('lp_row', 'ent_prev', 'ent_next', 'sH', 'sM', 'isnl', 'calc', 'relpos')
D = {}
t0 = time.time()
for S in SETS:
    M = dict(np.load(f'{a.root}/{S}/meta.npz')); TK = dict(np.load(f'{a.root}/{S}/tok.npz'))
    X = np.load(f'{a.root}/{S}/X.npy', mmap_mode='r')
    nL, R0 = X.shape[1], X.shape[2]; LAY = list(M['layers'])
    Xg = torch.empty(X.shape, dtype=torch.float16, device=dev)
    for i in range(0, X.shape[0], 1 << 20): Xg[i:i + (1 << 20)] = torch.from_numpy(np.ascontiguousarray(X[i:i + (1 << 20)])).to(dev)
    n = M['n']; off = M['off']; seg = np.repeat(np.arange(len(n)), n)
    relpos = (np.arange(len(seg)) - off[seg]) / np.maximum(n[seg] - 1, 1)
    chans = np.stack([TK['lp_row'], TK['ent_prev'], TK['ent_next'], TK['sH'], TK['sM'], TK['isnl'].astype(np.float32),
                      (TK['wf'] & 1).astype(np.float32), relpos.astype(np.float32)], 1)
    last = np.zeros(len(seg), bool); last[off + n - 1] = True
    isnl_m = TK['isnl'].astype(bool) & ~(last & (M['has_hash'][seg] > 0))     # line ends before '####' (offline M window)
    lg = np.nan_to_num(M['lg'], nan=-30.0); sn = SN[f'c4:HM:lin:pair|{S}']; assert len(sn) == len(n)
    cm = (M['has_hash'] > 0) & (lg > -29.5)
    D[S] = dict(X=Xg, C=torch.tensor(chans, device=dev), nlm=torch.tensor(isnl_m, device=dev), off=torch.tensor(off, device=dev),
                n=torch.tensor(n, device=dev), idx=M['idx'], ok=M['ok'].astype(bool), cm=cm,
                b=torch.tensor(np.maximum(sn, -12) + np.maximum(lg, -12), device=dev, dtype=torch.float32), LAY=LAY, R0=R0)
    print(S, 'attempts', len(n), 'committed', cm.sum(), 'tokens', len(seg), 'load', round(time.time() - t0, 1), flush=True)


def pairs(S, mask):
    idx = D[S]['idx']; y = D[S]['ok']; R_, W_ = [], []
    for it in np.unique(idx[mask]):
        ii = np.flatnonzero(mask & (idx == it)); yy = y[ii]
        if yy.all() or (~yy).all(): continue
        r, w = ii[yy], ii[~yy]; R_.append(np.repeat(r, len(w))); W_.append(np.tile(w, len(r)))
    return torch.tensor(np.concatenate(R_), device=dev), torch.tensor(np.concatenate(W_), device=dev)


PA = {S: pairs(S, D[S]['cm']) for S in SETS}


def wauc(S, s):
    s = torch.as_tensor(s, device=dev); r, w = PA[S]; d = s[r] - s[w]
    return float(((d > 0).float() + 0.5 * (d == 0).float()).mean())


def pooled_auc(S, s):
    from sklearn.metrics import roc_auc_score
    cm = D[S]['cm']; return float(roc_auc_score(D[S]['ok'][cm], np.asarray(s)[cm]))


BASE = {S: dict(b=wauc(S, D[S]['b'])) for S in SETS}
print('within AUROC b (sn+lg):', json.dumps(BASE), flush=True)


class Seq(nn.Module):
    def __init__(self, d, pool, inc):
        super().__init__()
        self.inc = inc; self.idr = nn.Dropout(a.idrop); self.pool = pool
        K = int(pool[pool.index('a') + 1]) if 'a' in pool else 0; self.K = K
        z = (lambda t: nn.init.zeros_(t)) if inc else (lambda t: nn.init.normal_(t, std=0.01))
        if K:
            self.q = nn.Linear(d, K); nn.init.normal_(self.q.weight, std=0.01); nn.init.zeros_(self.q.bias)
            self.v = nn.Linear(d, K, bias=False); z(self.v.weight)
        if 'm' in pool:
            self.wm = nn.Linear(d, 1, bias=False); nn.init.normal_(self.wm.weight, std=0.01); self.al = nn.Parameter(torch.zeros(1) if inc else torch.full((1,), -1.0))
        if 'u' in pool: self.u = nn.Linear(d, 1, bias=False); z(self.u.weight)
        if 'h' in pool:
            self.wh = nn.Linear(d, 1, bias=False); z(self.wh.weight); self.wl = nn.Linear(d, 1, bias=False); z(self.wl.weight)
        self.c = nn.Parameter(torch.zeros(1))

    def forward(self, f, seg, B, nb, last, nlm):
        f = self.idr(f); s = self.c.expand(B).clone()
        if self.K:
            at = self.q(f); v = self.v(f); K = self.K; sk = seg[:, None].expand(-1, K)
            mx = torch.full((B, K), -1e30, device=f.device).scatter_reduce(0, sk, at, 'amax', include_self=True)
            e = torch.exp(at - mx.detach()[seg])
            den = torch.zeros(B, K, device=f.device).index_add(0, seg, e); num = torch.zeros(B, K, device=f.device).index_add(0, seg, e * v)
            s = s + (num / den).sum(1)
        if 'm' in self.pool:
            w = self.wm(f)[:, 0]
            mx = torch.full((B,), -1e30, device=f.device).scatter_reduce(0, seg, w, 'amax', include_self=True)
            s = s + self.al * mx
        if 'u' in self.pool:
            s = s + torch.zeros(B, device=f.device).index_add(0, seg, self.u(f)[:, 0]) / nb
        if 'h' in self.pool:
            s = s + self.wh(f[last])[:, 0]
            cnt = torch.zeros(B, device=f.device).index_add(0, seg[nlm], torch.ones_like(seg[nlm], dtype=torch.float32))
            s = s + torch.zeros(B, device=f.device).index_add(0, seg[nlm], self.wl(f[nlm])[:, 0]) / cnt.clamp(min=1)
        return s


def run_cfg(cfg):
    lspec, feats, pool, mode = cfg.split(':')
    ls, R = (lspec.split('r') + [None])[:2] if 'r' in lspec and lspec != '-' else (lspec, None)
    lays = LSETS[ls]; R = int(R) if R else D['trainC']['R0']
    li = [D['trainC']['LAY'].index(L) for L in lays]
    use_x = 'x' in feats and len(lays) > 0; use_c = 'c' in feats; inc = mode == 'inc'

    def gather(S, att):
        """att: LongTensor of attempt ids -> token features (T, d), seg (T,), nb (B,), last mask, nlm mask"""
        d = D[S]; nb = d['n'][att]; B = len(att)
        seg = torch.repeat_interleave(torch.arange(B, device=dev), nb)
        st = torch.cumsum(nb, 0) - nb
        tok = d['off'][att][seg] + torch.arange(len(seg), device=dev) - st[seg]
        parts = []
        if use_x: parts.append(d['X'][tok][:, li, :R].float().reshape(len(tok), -1))
        if use_c: parts.append(d['C'][tok])
        f = torch.cat(parts, 1)
        last = torch.zeros(len(tok), dtype=torch.bool, device=dev); last[st + nb - 1] = True
        return f, seg, nb.float(), last, d['nlm'][tok]

    # standardization stats from trainC committed tokens (subsample)
    ct = np.flatnonzero(D['trainC']['cm'])
    rs = np.random.default_rng(0).choice(ct, 4000, replace=False)
    f0 = gather('trainC', torch.tensor(rs, device=dev))[0]
    mu = f0.mean(0); sd = f0.std(0) + 1e-3; dd = f0.shape[1]; del f0
    norm = lambda f: (f - mu) / sd
    Mt = D['trainC']; idx = Mt['idx']; fold = idx % 5; y = torch.tensor(Mt['ok'], device=dev)
    oof = {e: np.zeros(len(idx), np.float32) for e in EP}
    evs = {S: {e: np.zeros(len(D[S]['idx']), np.float32) for e in EP} for S in ('dev', 'dev2')}
    t0 = time.time()

    def score(net, S, att_np):
        net.eval(); out = np.zeros(len(att_np), np.float32)
        with torch.no_grad():
            for i in range(0, len(att_np), 2048):
                at = torch.tensor(att_np[i:i + 2048], device=dev); f, seg, nb, last, nlm = gather(S, at)
                s = net(norm(f), seg, len(at), nb, last, nlm)
                out[i:i + 2048] = s.cpu().numpy()             # inc: delta only (b added at evaluation)
        net.train(); return out
    WS = {e: [] for e in EP}; ev0 = {S: {} for S in ('dev', 'dev2')}
    for k in range(5):
        torch.manual_seed(a.seed * 100 + k); rng = np.random.default_rng(a.seed * 100 + k)
        tr = Mt['cm'] & (fold != k); tr_items = np.unique(idx[tr]); by = {}
        for i in np.flatnonzero(tr): by.setdefault(idx[i], []).append(i)
        net = Seq(dd, pool, inc).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
        nbat = int(np.ceil(len(tr_items) / a.bitems)); total = max(EP) * nbat
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.1)
        for ep in range(1, max(EP) + 1):
            perm = rng.permutation(tr_items)
            for bi in range(nbat):
                ii = np.concatenate([by[it] for it in perm[bi * a.bitems:(bi + 1) * a.bitems]])
                at = torch.tensor(ii, device=dev); f, seg, nb, last, nlm = gather('trainC', at)
                s = net(norm(f), seg, len(at), nb, last, nlm)
                yy = y[at]; it_ = torch.tensor(idx[ii], device=dev)
                if inc: s = s + Mt['b'][at]
                pm = (it_[:, None] == it_[None]) & yy[:, None] & ~yy[None]
                Lp = F.softplus(-(s[:, None] - s[None])[pm]).mean() if pm.any() else s.sum() * 0
                L = Lp if inc else 0.5 * F.binary_cross_entropy_with_logits(s, yy.float()) + Lp
                opt.zero_grad(); L.backward(); opt.step(); sch.step()
            if ep in EP:
                te = np.flatnonzero(Mt['cm'] & (fold == k))
                oof[ep][te] = score(net, 'trainC', te)
                for S in ('dev', 'dev2'):
                    cmS = np.flatnonzero(D[S]['cm']); sc_ = score(net, S, cmS); evs[S][ep][cmS] += sc_ / 5
                    if k == 0: ev0[S][ep] = np.zeros(len(D[S]['idx']), np.float32); ev0[S][ep][cmS] = sc_
                if a.save_w: WS[ep].append({kk: v.detach().cpu().clone() for kk, v in net.state_dict().items()})
    res = dict(cfg=cfg, d=int(dd), secs=round(time.time() - t0, 1), ep={})
    full = lambda S, s: s + (D[S]['b'].cpu().numpy() if inc else 0)
    for e in EP:
        res['ep'][e] = {S: dict(within=wauc(S, full(S, (oof[e] if S == 'trainC' else evs[S][e]))),
                                within_delta_alone=wauc(S, oof[e] if S == 'trainC' else evs[S][e])) for S in SETS}
    best = max(EP, key=lambda e: res['ep'][e]['trainC']['within']); res['best_ep'] = best
    if a.save_w:
        torch.save(dict(states=WS[best], mu=mu.cpu(), sd=sd.cpu(), cfg=cfg, R=R, layers=lays, chans=CH, epoch=best),
                   f'{HERE}/out/w_{a.tag}_{cfg.replace(":", "_")}.pt')
    print(f"{cfg:24s} d={dd} {res['secs']}s | " + ' | '.join(
        f"ep{e}{'*' if e == best else ''} " + ' '.join(f"{S} {res['ep'][e][S]['within']:.4f}" for S in SETS) for e in EP), flush=True)
    out = {f'{cfg}|trainC': oof[best], f'{cfg}|dev': evs['dev'][best], f'{cfg}|dev2': evs['dev2'][best]}
    if a.perfold:
        out.update({f'{cfg}#f0|trainC': oof[best], f'{cfg}#f0|dev': ev0['dev'][best], f'{cfg}#f0|dev2': ev0['dev2'][best]})
        res['f0'] = {S: wauc(S, full(S, ev0[S][best])) for S in ('dev', 'dev2')}
        print('   fold-0 single model:', res['f0'], flush=True)
    return res, out


Rj = dict(args=vars(a), base=BASE, cfgs=[]); SC = {}
os.makedirs(f'{HERE}/out', exist_ok=True)
T0 = time.time()
for cfg in a.grid.split(','):
    if time.time() - T0 > a.tmax: print('TMAX reached, skipping', cfg, flush=True); continue
    r, sc = run_cfg(cfg); Rj['cfgs'].append(r); SC.update(sc)
    json.dump(Rj, open(f'{HERE}/out/train_{a.tag}.json', 'w'), indent=1)
    np.savez_compressed(f'{HERE}/out/sc_{a.tag}.npz', **SC)
print('TRAIN DONE', flush=True)
