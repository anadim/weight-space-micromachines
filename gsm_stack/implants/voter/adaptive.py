"""Abstain / adaptive-N: what if the stream keeps trying when the vote is uncertain?  Bootstrap from the 32 attempts
per item (dev, dev2): fixed N in {8,12,16,24,32} with the stack probe voter; adaptive: 8 tries, then while the kept
winner's probe-weight share < tau add 4 more tries (cap Nmax).  Reports acc and mean attempts."""
import os, sys, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vote as V


def probe_vote(S):
    i = V.v_probe(S)
    if i is None: return None, 0.0
    keep = V.keep_mask(S); w = [V.sig(a['rlg']) for a in S]
    tot = sum(wi for wi, k in zip(w, keep) if k)
    win = sum(w[j] for j, b in enumerate(S) if keep[j] and V.same_val(S[i]['val'], b['val']))
    return i, win / max(tot, 1e-9)


for s in sys.argv[1:] or ['trainC', 'dev', 'dev2']:
    by, items = V.load_set(s)
    rng = random.Random(0); B = 20
    pools = {idx: [a for a in A if a['committed']] for idx, A in by.items()}
    perms = {idx: [rng.sample(P, len(P)) for _ in range(B)] for idx, P in pools.items()}
    line = []
    for N in (8, 12, 16, 24, 32)[:3 if s == 'trainC' else 5]:
        acc = np.mean([(lambda i, S: S[i]['vok'] if i is not None else 0)(probe_vote(p[:N])[0], p[:N])
                       for idx in perms for p in perms[idx]])
        line.append(f'N={N}:{100*acc:.2f}')
    print(s, 'fixed N, stack probe voter:', ' '.join(line))
    for tau in (0.5, 0.6, 0.7, 0.8, 0.9):
        for nmax in ((16,) if s == 'trainC' else (16, 24)):
            acc = []; natt = []
            for idx in perms:
                for p in perms[idx]:
                    n = 8
                    while True:
                        i, sh = probe_vote(p[:n])
                        if sh >= tau or n >= nmax or n >= len(p): break
                        n += 4
                    acc.append(p[:n][i]['vok'] if i is not None else 0); natt.append(n)
            print(f'   adaptive tau={tau} Nmax={nmax}: acc {100*np.mean(acc):.2f}  mean tries {np.mean(natt):.2f}  '
                  f'extended {100*np.mean(np.array(natt) > 8):.1f}%')
