"""lane seqver: per-TOKEN projected states of every body token of each rung-3/4 attempt (teacher-forced in the in-stack context
pre + C.q_ids(q) + seg_tokens, as sidenet extract_sn.py), for a sequence side-net verifier.

Body tokens of an attempt = seg_tokens offsets phrase_len <= o <= hash_off (no '####': through the last token), i.e. every
host/calc-written token after the written attempt phrase, up to and including the token completing '####' (= sidenet's H).
Per body token t and layer L in {14,18,22,27} (layer INPUT = HF hidden_states[L]):
  X[t, l, :] = (h_t,L - mu_L) @ P_L  (fp16, r dims). P_L = diag(1/sd_L) V_L[:, :r]: PCA of the per-dim standardized body-token
  states, fit on a fixed trainC subsample (--pca_n attempts) in this job (pass 1). Saved to out/pca.pt (fixed linear maps).
Per-token channels (tok.npz): lp_row (stack's sampled logp from the row, 0 for calc/written tokens), lp_m (teacher-forced host
logp of the token), ent_prev (entropy of the host distribution that produced the token), ent_next (entropy of the next-token
distribution at the token), sH / sM (sidenet g4 per-layer-summed projections wH.h, wM.h of the full 1024-d states), wf, isnl, tok.
meta.npz: off, n (token ranges per attempt, table order), idx, seed, k, ok, lg, has_hash, lg_ours (L18 probe at H, full dims),
sn_ours (g4 side-net recomputed from the per-token sH/sM = sH[H] + mean sM over line ends pl<=o<hash_off + b).
Validation -> out/extract_{set}.json: lg_ours vs register lg; sn_ours vs sidenet fold-mean scores (dev/dev2); lp_m vs lp_row.
"""
import os, sys, json, gzip, time, argparse
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C

ap = argparse.ArgumentParser()
ap.add_argument('--sets', default='dev,dev2,trainC')
ap.add_argument('--data', default=f'{HERE}/tables')
ap.add_argument('--out_root', default='$WSM_DATA/implants/seqver')
ap.add_argument('--ckpt', default='$WSM_DATA/gsm_sel/ckpt_a2s2')
ap.add_argument('--probe', default=f'{HERE}/verifier_probe_L18.pt')
ap.add_argument('--snw', default=f'{HERE}/sn_w_g4.pt')
ap.add_argument('--layers', default='14,18,22,27')
ap.add_argument('--r', type=int, default=256)
ap.add_argument('--pca_n', type=int, default=3000)
ap.add_argument('--pca', default='', help='reuse an existing pca.pt instead of fitting')
ap.add_argument('--tok_budget', type=int, default=131072)
ap.add_argument('--limit_items', type=int, default=0)
ap.add_argument('--tag', default='')
a = ap.parse_args()

from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file
dev = 'cuda'
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
pre, shot_q = C.prefix_ids(tok); Lp = len(pre)
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', dtype=torch.bfloat16, attn_implementation='sdpa').to(dev).eval()
sd = load_file(f'{a.ckpt}/model.safetensors', device='cpu')
msd = m.state_dict()
names = [k for k in msd if k != 'lm_head.weight']
assert all(k in sd for k in names)
with torch.no_grad():
    for k in names: msd[k].copy_(sd[k].float().to(torch.bfloat16).to(dev))
del sd
print('host loaded, Lp', Lp, flush=True)
LAY = [int(x) for x in a.layers.split(',')]; nL = len(LAY); D = m.config.hidden_size; R = a.r
PR = torch.load(a.probe); pw = PR['w'].float().to(dev); pb = float(PR['b']); pL = int(PR.get('layer', 18))
assert pL in LAY
SW = torch.load(a.snw); assert list(SW['layers']) == LAY
_w = SW['w'].float().to(dev).view(2, nL, D)                 # [H|M] x layer x D
WHM = {L: torch.stack([_w[0, j], _w[1, j]], 1) for j, L in enumerate(LAY)}   # (D, 2)
snb = float(SW['b'])
NLTOK = np.array(['\n' in tok.decode([i]) for i in range(len(tok))] + [False] * (m.config.vocab_size - len(tok)))
W_lm = m.lm_head.weight                                        # (V, D) bf16

cap = {}; cur = {}
def _pre(L):
    def f(mod, args, kwargs):
        h = args[0] if args else kwargs['hidden_states']
        cap[L] = h.reshape(-1, D).index_select(0, cur['flat'])
    return f
for L in LAY:
    m.model.layers[L].register_forward_pre_hook(_pre(L), with_kwargs=True)


def read_table(p):
    with gzip.open(p, 'rt') as f: return [json.loads(l) for l in f]


def body(r):
    st = r['seg_tokens']; ho = r.get('hash_off'); pl = int(r.get('phrase_len') or 0)
    e = int(ho) if ho is not None else len(st) - 1
    return pl, e                                                # body offsets pl..e inclusive


def batches(rows, qids, order):
    b0 = 0; n = len(order)
    while b0 < n:
        b1 = b0
        while b1 < n:
            T = Lp + len(qids[rows[order[b1]]['idx']]) + len(rows[order[b1]]['seg_tokens'])
            if (b1 - b0 + 1) * T > a.tok_budget and b1 > b0: break
            b1 += 1
        yield order[b0:b1]; b0 = b1


def forward(rows, qids, bat):
    """returns per-layer full states at body positions (concat over attempts in bat order), final hidden at positions o-1 for
    o in pl..e+1 (n+1 per attempt), and the per-attempt body lengths"""
    seqs = [pre + qids[rows[i]['idx']] + rows[i]['seg_tokens'] for i in bat]
    T = max(map(len, seqs)); B = len(seqs)
    ids = torch.full((B, T), tok.pad_token_id or 0, dtype=torch.long)
    flat, flat2, ns = [], [], []
    for b, (i, s) in enumerate(zip(bat, seqs)):
        ids[b, :len(s)] = torch.tensor(s)
        qb = Lp + len(qids[rows[i]['idx']]); pl, e = body(rows[i])
        flat.extend(range(b * T + qb + pl, b * T + qb + e + 1)); flat2.extend(range(b * T + qb + pl - 1, b * T + qb + e + 1))
        ns.append(e - pl + 1)
    cur['flat'] = torch.tensor(flat, device=dev)
    with torch.no_grad():
        out = m.model(input_ids=ids.to(dev), use_cache=False)
        fin = out.last_hidden_state.reshape(-1, D).index_select(0, torch.tensor(flat2, device=dev))
    return {L: cap[L] for L in LAY}, fin, ns, sum(map(len, seqs))


def lm_stats(fin, toks):
    """fin (n+1 per attempt concat) -> per position: logsumexp-normalized logp of the next token and entropy"""
    lp = torch.empty(len(fin), device=dev); ent = torch.empty(len(fin), device=dev)
    with torch.no_grad():
        for i in range(0, len(fin), 4096):
            lo = (fin[i:i + 4096] @ W_lm.T).float(); ls = torch.log_softmax(lo, -1)
            ent[i:i + 4096] = -(ls.exp() * ls).sum(-1)
            t = toks[i:i + 4096]; v = t >= 0
            g = ls.gather(1, t.clamp(min=0)[:, None])[:, 0]
            lp[i:i + 4096] = torch.where(v, g, torch.zeros_like(g))
    return lp, ent


def fit_pca():
    if a.pca:
        P = torch.load(a.pca); print('PCA reused', a.pca, flush=True); return P
    t0 = time.time()
    rows = read_table(f'{a.data}/attempts_trainC.jsonl.gz')
    rng = np.random.default_rng(0); sel = sorted(rng.choice(len(rows), min(a.pca_n, len(rows)), replace=False).tolist())
    rows = [rows[i] for i in sel]
    qids = {r['idx']: C.q_ids(tok, r['q']) for r in rows}
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]['seg_tokens']))
    S1 = {L: torch.zeros(D, dtype=torch.float64, device=dev) for L in LAY}
    S2 = {L: torch.zeros(D, D, dtype=torch.float64, device=dev) for L in LAY}; N = 0
    for bat in batches(rows, qids, order):
        hs, _, ns, _ = forward(rows, qids, bat)
        for L in LAY:
            x = hs[L].float(); S1[L] += x.sum(0).double(); S2[L] += (x.T @ x).double()
        N += sum(ns)
    P = dict(layers=LAY, r=R, n_tok=N, n_att=len(rows))
    for L in LAY:
        mu = S1[L] / N; cov = S2[L] / N - mu[:, None] * mu[None]
        sdv = cov.diagonal().clamp(min=1e-8).sqrt(); corr = cov / sdv[:, None] / sdv[None]
        ev, V = torch.linalg.eigh(corr); ev, V = ev.flip(0), V.flip(1)
        Pm = (V[:, :R] / sdv[:, None]).float()
        # fraction of total RAW variance captured by the projection subspace (back-projected)
        P[f'mu{L}'] = mu.float().cpu(); P[f'P{L}'] = Pm.cpu(); P[f'ev{L}'] = ev.float().cpu(); P[f'sd{L}'] = sdv.float().cpu()
        P[f'zvar{L}'] = float(ev[:R].sum() / ev.sum())
    P['secs'] = round(time.time() - t0, 1)
    torch.save(P, f'{HERE}/out/pca{a.tag}.pt')
    print('PCA', {f'zvar{L}': round(P[f'zvar{L}'], 4) for L in LAY}, 'n_tok', N, 'secs', P['secs'], flush=True)
    return P


def run_set(S, P):
    t0 = time.time()
    rows = read_table(f'{a.data}/attempts_{S}.jsonl.gz')
    if a.limit_items:
        ks = set(sorted({r['idx'] for r in rows})[:a.limit_items]); rows = [r for r in rows if r['idx'] in ks]
    n = len(rows); od = f'{a.out_root}/{S}{a.tag}'; os.makedirs(od, exist_ok=True)
    qids = {}
    for r in rows:
        if r['idx'] not in qids: qids[r['idx']] = C.q_ids(tok, r['q'])
    nb = np.array([body(r)[1] - body(r)[0] + 1 for r in rows], np.int64); off = np.r_[0, np.cumsum(nb)[:-1]]; Nt = int(nb.sum())
    X = np.lib.format.open_memmap(f'{od}/X.npy', 'w+', np.float16, (Nt, nL, R))
    TK = {k: np.zeros(Nt, np.float32) for k in ('lp_row', 'lp_m', 'ent_prev', 'ent_next', 'sH', 'sM')}
    TK['wf'] = np.zeros(Nt, np.int16); TK['isnl'] = np.zeros(Nt, np.int8); TK['tok'] = np.zeros(Nt, np.int32)
    for i, r in enumerate(rows):
        pl, e = body(r); sl = slice(off[i], off[i] + nb[i])
        TK['tok'][sl] = r['seg_tokens'][pl:e + 1]; TK['wf'][sl] = r['seg_wf'][pl:e + 1]
        TK['lp_row'][sl] = r['seg_logp'][pl:e + 1]; TK['isnl'][sl] = NLTOK[np.array(r['seg_tokens'][pl:e + 1])]
    Pm = {L: P[f'P{L}'].to(dev) for L in LAY}; Mu = {L: P[f'mu{L}'].to(dev) for L in LAY}
    lg_ours = np.full(n, np.nan, np.float32); sn_ours = np.full(n, np.nan, np.float32)
    order = sorted(range(n), key=lambda i: Lp + len(qids[rows[i]['idx']]) + len(rows[i]['seg_tokens']))
    ntok = 0
    for bat in batches(rows, qids, order):
        hs, fin, ns, nt = forward(rows, qids, bat); ntok += nt
        # targets for lm stats: position o-1 predicts token o (o = pl..e), position e predicts the token after (none -> -1)
        tg = []
        for i in bat:
            pl, e = body(rows[i]); st = rows[i]['seg_tokens']; tg.extend(st[pl:e + 1]); tg.append(st[e + 1] if e + 1 < len(st) else -1)
        lp, ent = lm_stats(fin, torch.tensor(tg, device=dev))
        with torch.no_grad():
            Xb = torch.stack([((hs[L].float() - Mu[L]) @ Pm[L]) for L in LAY], 1).to(torch.float16).cpu().numpy()
            sHM = sum(hs[L].float() @ WHM[L] for L in LAY)          # (nt, 2)
            hL = hs[pL].float() @ pw + pb
        sHM = sHM.cpu().numpy(); hL = hL.cpu().numpy(); lp = lp.cpu().numpy(); ent = ent.cpu().numpy()
        p0 = 0; q0 = 0
        for j, i in enumerate(bat):
            k = ns[j]; sl = slice(off[i], off[i] + k)
            X[sl] = Xb[p0:p0 + k]
            TK['sH'][sl] = sHM[p0:p0 + k, 0]; TK['sM'][sl] = sHM[p0:p0 + k, 1]
            TK['lp_m'][sl] = lp[q0:q0 + k]; TK['ent_prev'][sl] = ent[q0:q0 + k]; TK['ent_next'][sl] = ent[q0 + 1:q0 + k + 1]
            lg_ours[i] = hL[p0 + k - 1]
            nl = TK['isnl'][off[i]:off[i] + k - 1].astype(bool)     # line ends before the H token (offline M: pl <= o < hash_off)
            if rows[i].get('hash_off') is None: nl = TK['isnl'][off[i]:off[i] + k].astype(bool)
            sm = sHM[p0:p0 + len(nl), 1][nl]
            sn_ours[i] = sHM[p0 + k - 1, 0] + (sm.mean() if len(sm) else 0.0) + snb
            p0 += k; q0 += k + 1
    X.flush(); del X
    np.savez(f'{od}/tok.npz', **TK)
    meta = dict(off=off, n=nb, idx=np.array([r['idx'] for r in rows], np.int32), seed=np.array([r['seed'] for r in rows], np.int32),
                k=np.array([r['k'] for r in rows], np.int16), ok=np.array([int(r['ok']) for r in rows], np.int8),
                lg=np.array([np.nan if r.get('lg') is None else float(r['lg']) for r in rows], np.float32),
                has_hash=np.array([r.get('hash_off') is not None for r in rows], np.int8), lg_ours=lg_ours, sn_ours=sn_ours,
                layers=np.array(LAY), r=R)
    np.savez(f'{od}/meta.npz', **meta)
    t = meta['lg']; ok_ = meta['has_hash'].astype(bool) & np.isfinite(t) & (t > -29.5)
    d = np.abs(t[ok_] - lg_ours[ok_])
    val = dict(set=S, n=n, n_tok=Nt, n_cmp=int(ok_.sum()), lg_corr=float(np.corrcoef(t[ok_], lg_ours[ok_])[0, 1]), lg_mae=float(d.mean()),
               lg_max=float(d.max()), secs=round(time.time() - t0, 1), tokens=ntok, size_gb=round(Nt * nL * R * 2 / 1e9, 2))
    hm = TK['wf'] == 0
    val['lp_corr_host_tokens'] = float(np.corrcoef(TK['lp_row'][hm], TK['lp_m'][hm])[0, 1])
    val['lp_mae_host_tokens'] = float(np.abs(TK['lp_row'][hm] - TK['lp_m'][hm]).mean())
    if not a.limit_items:
        ref = dict(np.load(f'{HERE}/tables/sc_g4.npz'))[f'c4:HM:lin:pair|{S}']
        if S != 'trainC':
            dd = np.abs(ref[ok_] - sn_ours[ok_])
            val.update(sn_corr=float(np.corrcoef(ref[ok_], sn_ours[ok_])[0, 1]), sn_mae=float(dd.mean()), sn_max=float(dd.max()))
        else:
            val.update(sn_corr_vs_oof=float(np.corrcoef(ref[ok_], sn_ours[ok_])[0, 1]))
    os.makedirs(f'{HERE}/out', exist_ok=True)
    json.dump(val, open(f'{HERE}/out/extract_{S}{a.tag}.json', 'w'), indent=1)
    print('VALIDATION', json.dumps(val), flush=True)


os.makedirs(f'{HERE}/out', exist_ok=True)
P = fit_pca()
for S in a.sets.split(','):
    run_set(S, P)
print('EXTRACT DONE', flush=True)
