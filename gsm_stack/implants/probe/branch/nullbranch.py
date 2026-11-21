"""Null check: host 'higher-prob token = right' accuracy at atlas branch points with real vs shuffled outcome labels
(labels permuted among attempts of the same item, n_right kept). python3.11 probe/branch/nullbranch.py"""
import gzip, json, collections, numpy as np, sys
sys.path.insert(0, '.'); from build import group_of, rd
rng = np.random.default_rng(0)
def lcp(x, y):
    n = min(len(x), len(y)); i = 0
    while i < n and x[i] == y[i]: i += 1
    return i
for S in ['trainC', 'dev', 'dev2']:
    A = rd(f'../tables/attempts_{S}.jsonl.gz'); I = {r['idx']: r for r in rd(f'../tables/items_{S}.jsonl.gz')}
    by = collections.defaultdict(list)
    for a in A:
        if I[a['idx']]['class'] == 'mixed': by[a['idx']].append(a)
    def acc(perm):
        hits = []; inl = []
        for idx, atts in by.items():
            ok = np.array([bool(a['ok']) for a in atts])
            if perm: ok = rng.permutation(ok)
            for i, w in enumerate(atts):
                if ok[i]: continue
                best = None
                for j, r in enumerate(atts):
                    if not ok[j] or group_of(r) != group_of(w): continue
                    L = lcp(w['seg_tokens'], r['seg_tokens'])
                    if L >= min(len(w['seg_tokens']), len(r['seg_tokens'])): continue
                    if best is None or L > best[0] or (L == best[0] and r['seg_logp'][L] > best[1]['seg_logp'][L]): best = (L, r)
                if best is None: continue
                L, r = best
                if (w['seg_wf'][L] | r['seg_wf'][L]) & 3: continue
                hits.append((r['seg_logp'][L] > w['seg_logp'][L]) + 0.5 * (r['seg_logp'][L] == w['seg_logp'][L]))
        return np.mean(hits), len(hits)
    o = acc(False); ns = [acc(True)[0] for _ in range(3)]
    print(f'{S}: per-wrong-attempt host acc real {o[0]:.3f} (n {o[1]}) | shuffled labels {np.mean(ns):.3f} +- {np.std(ns):.3f}', flush=True)
