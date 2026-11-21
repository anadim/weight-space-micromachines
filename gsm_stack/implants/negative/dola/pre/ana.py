"""DoLa prescreen analysis (local). python3.11 implants/dola/pre/ana.py [--out pre/job/out] -> pre/ana.log (stdout) + ana.json
Per config (premature layer or dyn, alpha, beta):
 forks  : atlas branch pairs (probe/branch/tables/pairs_*, clean): sign flips of D = lp(tok_r) - lp(tok_w) toward right / wrong,
          full-vocab argmax: a0 = wrong -> a' = right (fixes) vs a0 = right -> a' = wrong (breaks); soft trie-fork pairs: weighted acc.
 drift  : host-sampled positions (wf & 3 == 0) of RIGHT attempts (always-right items seed 7; right attempts of mixed items):
          mean dlp = lp'(x) - lp(x), argmax-change rate, APC-excluded sampled tokens.
 seq    : S = sum over host-sampled positions of (lp' - lp) per attempt; within-item AUROC (right vs wrong, mixed items) of S,
          SNIS per-item P'(right) = sum w ok / sum w (w = exp S) vs P(right) (mean over mixed items) and mean ESS/n.
"""
import os, sys, json, gzip, argparse, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); WS = os.path.expanduser('gsm_stack/implants')
ap = argparse.ArgumentParser(); ap.add_argument('--out', default=f'{HERE}/job/out'); a = ap.parse_args()
rd = lambda fn: [json.loads(l) for l in gzip.open(fn, 'rt')]


def within_auroc(g, y, s):
    """mean over items of AUROC(right vs wrong) with ties 0.5, weighted by pairs"""
    num = den = 0.0
    by = collections.defaultdict(list)
    for gi, yi, si in zip(g, y, s): by[gi].append((yi, si))
    for v in by.values():
        r = np.array([s for y, s in v if y]); w = np.array([s for y, s in v if not y])
        if len(r) == 0 or len(w) == 0: continue
        d = r[:, None] - w[None]
        num += (d > 0).sum() + 0.5 * (d == 0).sum(); den += d.size
    return num / den


res = {}
for S in ['dev', 'dev2']:
    A = json.load(open(f'{a.out}/{S}_att.json')); CFG = [tuple(c) for c in A['cfg']]; NC = len(CFG)
    P = dict(np.load(f'{a.out}/{S}_pos.npz'))
    Fk = dict(np.load(f'{a.out}/{S}_fork.npz'))
    ok = np.array(A['ok'], bool); cls = np.array(A['cls']); idx = np.array(A['idx'])
    host = (P['wf'] & 3) == 0
    LPX = P['LPX'].astype(np.float32); lp0 = P['lpx0']
    AMC = np.unpackbits(P['AMC'], axis=1)[:, :NC].astype(bool); AMX = np.unpackbits(P['AMX'], axis=1)[:, :NC].astype(bool)
    att = P['att']
    dl = LPX - lp0[:, None]
    excl = LPX <= -98
    print(f'== {S}: attempts {len(ok)} (mixed {int((cls == "mixed").sum())}, always {int((cls == "always").sum())}), positions {len(att)}, '
          f'host-sampled {int(host.sum())}; lp0 vs stored slp MAE (host) {np.abs(lp0 - P["slp"])[host].mean():.4f}', flush=True)
    # drift on right attempts
    rmask_al = host & (cls[att] == 'always')
    rmask_mx = host & (cls[att] == 'mixed') & ok[att]
    # seq-level
    hostw = host.astype(np.float32)
    Sx = np.zeros((len(ok), NC)); np.add.at(Sx, att, dl * hostw[:, None])
    nX = np.bincount(att, weights=hostw, minlength=len(ok))
    S0 = np.bincount(att, weights=lp0 * hostw, minlength=len(ok))
    mx = cls == 'mixed'
    au0 = within_auroc(idx[mx], ok[mx], S0[mx] / np.maximum(nX[mx], 1))
    # forks / pairs
    row = {f: i for i, f in enumerate(Fk['fid'])}
    pairs = [p for p in rd(f'{WS}/probe/branch/tables/pairs_{S}.jsonl.gz') if p['clean'] and p['fid'] in row]
    forks = {f['fid']: f for f in rd(f'{WS}/probe/branch/tables/forks_{S}.jsonl.gz')}
    def cpos(r, t):
        k = np.where(Fk['cand'][r] == t)[0]; return k[0] if len(k) else -1
    PR = [(row[p['fid']], cpos(row[p['fid']], p['tok_r']), cpos(row[p['fid']], p['tok_w']), p['tok_r'], p['tok_w'], p['n_wrong']) for p in pairs]
    PR = [x for x in PR if x[1] >= 0 and x[2] >= 0]
    pr = np.array([x[0] for x in PR]); kr = np.array([x[1] for x in PR]); kw = np.array([x[2] for x in PR])
    tr = np.array([x[3] for x in PR]); tw = np.array([x[4] for x in PR])
    D0 = Fk['lp0c'][pr, kr] - Fk['lp0c'][pr, kw]; a0 = Fk['a0'][pr]
    FP = []
    for f, r in row.items():
        ch = forks[f]['children']
        for i in range(len(ch)):
            for j in range(i + 1, len(ch)):
                va, vb = ch[i][2] / ch[i][1], ch[j][2] / ch[j][1]
                if va == vb: continue
                ka, kb = cpos(r, ch[i][0]), cpos(r, ch[j][0])
                if ka < 0 or kb < 0: continue
                FP.append((r, ka, kb, 1.0 if va > vb else -1.0, abs(va - vb)))
    FP = np.array(FP)
    fr, fa, fb, fs, fw = FP[:, 0].astype(int), FP[:, 1].astype(int), FP[:, 2].astype(int), FP[:, 3], FP[:, 4]
    fD0 = fs * (Fk['lp0c'][fr, fa] - Fk['lp0c'][fr, fb])
    print(f'  pairs {len(PR)} (host acc {np.mean(D0 > 0):.3f}, argmax=r {np.mean(a0 == tr):.3f}, argmax=w {np.mean(a0 == tw):.3f}); '
          f'soft fork pairs {len(FP)} (host wacc {np.average(fD0 > 0, weights=fw):.3f}); within-item AUROC of mean lp (host conf) {au0:.3f}')
    hdr = (f'  {"cfg":16s} | pairs: D>0  ->R/->W  amax W->R / R->W  dmean | forks wacc | drift always: dlp  amaxchg excl | '
           f'mixed-right: dlp amaxchg | seq: AUROC(S) AUROC(S/n) dP_snis ESS/n')
    print(hdr)
    out = {}
    for ci, c in enumerate(CFG):
        D1 = Fk['FLP'][pr, ci, kr] - Fk['FLP'][pr, ci, kw]; a1 = Fk['FAM'][pr, ci]
        to_r = int(((D0 < 0) & (D1 > 0)).sum()); to_w = int(((D0 > 0) & (D1 < 0)).sum())
        afix = int(((a0 == tw) & (a1 == tr)).sum()); abrk = int(((a0 == tr) & (a1 == tw)).sum())
        fD1 = fs * (Fk['FLP'][fr, ci, fa] - Fk['FLP'][fr, ci, fb])
        wacc = float(np.average(fD1 > 0, weights=fw))
        dla = float(dl[rmask_al, ci].mean()); amca = float(AMC[rmask_al, ci].mean()); exa = float(excl[rmask_al, ci].mean())
        dlm = float(dl[rmask_mx, ci].mean()); amcm = float(AMC[rmask_mx, ci].mean())
        Sc = Sx[mx, ci]
        auS = within_auroc(idx[mx], ok[mx], Sc); auSn = within_auroc(idx[mx], ok[mx], Sc / np.maximum(nX[mx], 1))
        dP, ess = [], []
        for it in np.unique(idx[mx]):
            m_ = idx[mx] == it; s = Sc[m_]; o = ok[mx][m_]
            w = np.exp(s - s.max()); dP.append((w * o).sum() / w.sum() - o.mean()); ess.append(w.sum() ** 2 / (w ** 2).sum() / len(w))
        r = dict(pairs_to_r=to_r, pairs_to_w=to_w, amax_fix=afix, amax_brk=abrk, dmean=float(np.mean(D1 - D0)), fork_wacc=wacc,
                 drift_always_dlp=dla, drift_always_amc=amca, drift_always_excl=exa, drift_mixr_dlp=dlm, drift_mixr_amc=amcm,
                 auroc_S=auS, auroc_Sn=auSn, dP_snis=float(np.mean(dP)), ess=float(np.mean(ess)))
        out['%s_a%s_b%s' % c] = r
        print(f'  {"%s a%s b%s" % c:16s} | {np.mean(D1 > 0):.3f} {to_r:4d}/{to_w:<4d} {afix:4d}/{abrk:<4d} {r["dmean"]:+.3f} | {wacc:.3f} | '
              f'{dla:+.3f} {amca:.3f} {exa:.4f} | {dlm:+.3f} {amcm:.3f} | {auS:.3f} {auSn:.3f} {r["dP_snis"]:+.4f} {r["ess"]:.3f}', flush=True)
    res[S] = dict(n_pairs=len(PR), host_acc=float(np.mean(D0 > 0)), host_conf_auroc=au0, cfg=out)
json.dump(res, open(f'{HERE}/ana.json', 'w'), indent=1)
