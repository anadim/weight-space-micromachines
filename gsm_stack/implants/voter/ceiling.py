"""How good must a per-attempt verifier be?  Synthetic score s = d*ok + N(0,1) (pairwise AUROC = Phi(d/sqrt2)), combined
with the answer count as  score(v) = sum_{i in v} exp(beta*s_i)  (beta tuned per row; beta=0 = plain count), averaged over
noise draws, on the dev/dev2 seed streams.  Also: argmax-verifier (pick the single best-scored attempt)."""
import os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vote as V

rng = np.random.default_rng(0)
for s in ['dev', 'dev2']:
    by, items = V.load_set(s)
    st = V.build(V.seed_streams(by), ['lg'])
    for x, (idx, sd, S) in zip(st, V.seed_streams(by)):
        if not any(a['voted'] for a in by[idx] if a['seed'] == sd): x.fixed = items[idx]['vote_ok'][str(sd)]
    probe = np.mean([x.correct(x.pick(np.array([V.sig(a['rlg']) for a in x.S]))) for x in st])
    print(f'{s}: stack probe voter {100*probe:.2f}; oracle@8 {100*np.mean([x.ok.any() if x.fixed is None else x.fixed for x in st]):.2f}')
    for auc in (0.7, 0.77, 0.8, 0.85, 0.9, 0.95, 0.99):
        d = math.sqrt(2) * float(__import__('scipy.stats', fromlist=['norm']).norm.ppf(auc))
        res = {}
        for beta in (0.0, 0.5, 1.0, 2.0, 4.0, 1e3):
            accs = []
            for rep in range(5):
                acc = 0
                for x in st:
                    sc = d * x.ok + rng.standard_normal(len(x.ok))
                    w = np.exp(np.clip(beta * (sc - sc.max() if len(sc) else sc), -700, 0))
                    acc += x.correct(x.pick(w))
                accs.append(acc / len(st))
            res[beta] = 100 * np.mean(accs)
        print(f'  AUROC {auc:.2f}: ' + ' '.join(f'b={b:g}:{v:.2f}' for b, v in res.items()) + f'  best {max(res.values()):.2f}')
