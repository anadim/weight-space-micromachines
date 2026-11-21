"""Are forks decisions? Between-child outcome heterogeneity at trie forks vs binomial/permutation null.
python3.11 probe/branch/decisive.py"""
import gzip, json, numpy as np
from scipy.stats import chi2
rng = np.random.default_rng(0)
for S in ['trainC', 'dev', 'dev2']:
    F = [json.loads(l) for l in gzip.open(f'tables/forks_{S}.jsonl.gz', 'rt')]
    for nmin in (3, 5):
        st, df, pv, perm, crit, pos = [], [], [], [], [], []
        for f in F:
            ch = [c for c in f['children'] if c[1] >= nmin]
            if len(ch) < 2 or not f['clean']: continue
            n = np.array([c[1] for c in ch], float); r = np.array([c[2] for c in ch], float)
            vb = r.sum() / n.sum()
            if vb in (0, 1): continue
            X = ((r - n * vb) ** 2 / (n * vb * (1 - vb))).sum()
            st.append(X); df.append(len(ch) - 1); pv.append(chi2.sf(X, len(ch) - 1))
            # exact permutation null of X (shuffle outcomes among the attempts)
            outc = np.r_[np.ones(int(r.sum())), np.zeros(int(n.sum() - r.sum()))]
            Xs = []
            for _ in range(200):
                p = rng.permutation(outc); cs = np.cumsum(np.r_[0, n]).astype(int)
                rr = np.array([p[cs[i]:cs[i + 1]].sum() for i in range(len(n))])
                Xs.append(((rr - n * vb) ** 2 / (n * vb * (1 - vb))).sum())
            perm.append(np.mean(np.array(Xs) >= X - 1e-9)); pos.append(f['L'] - f['plen'])
        st, df, pv, perm = map(np.array, (st, df, pv, perm))
        print(f'{S} nmin {nmin}: forks {len(st)} | mean X/df {np.mean(st / df):.2f} (null 1) | perm p<0.05: {np.mean(perm < 0.05):.3f} '
              f'p<0.01: {np.mean(perm < 0.01):.3f} (null <=0.05/0.01) | median offset {np.median(pos):.0f}')
