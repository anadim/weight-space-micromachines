"""Hand-built agreement voters (one knob gamma each, tuned on trainC streams, then applied to dev / dev2):
 R  reasoning agreement: attempts vote for answers AND intermediates; w_i = sig(lg_i) * (1 + g * s_i),
    s_i = mean over x in I_i of the probe-weighted share of the other attempts whose (I_j + answer_j) contains x.
 G  agreement graph: score(v) = sum_{i in v} sig(lg_i) + g * sum_{i in v} sum_{j not in v} sig(lg_j) J_ij,
    J_ij = |I_i & I_j| / |I_i | I_j|  (partial support from attempts that share the reasoning but not the answer).
 E  early-stop veto: w_i = sig(lg_i) * (1 - g * [a_i is an intermediate of a different-answer attempt]).
 P  probe power: w_i = sig(lg_i)^g  (sharpen / flatten the stack weights)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vote as V


def ksets(S):
    S = [V._keys(a) for a in S]
    return [a['_I'] for a in S], [a['_I'] | ({a['_v']} if a['_v'] is not None else set()) for a in S]


def pick_R(S, g):
    I, U = ksets(S); w = [V.sig(a['rlg']) for a in S]; n = len(S)
    ww = []
    for i in range(n):
        tot = sum(w[j] for j in range(n) if j != i)
        s = np.mean([sum(w[j] for j in range(n) if j != i and x in U[j]) / max(tot, 1e-9) for x in I[i]]) if I[i] else 0.0
        ww.append(w[i] * (1 + g * s))
    return V.weighted_pick(S, ww)


def pick_G(S, g):
    I, U = ksets(S); w = [V.sig(a['rlg']) for a in S]; n = len(S); keep = V.keep_mask(S)
    J = [[len(I[i] & I[j]) / max(1, len(I[i] | I[j])) for j in range(n)] for i in range(n)]
    best, bi = -1e18, None
    for i in range(n):
        if not keep[i]: continue
        v = [j for j in range(n) if keep[j] and V.same_val(S[i]['val'], S[j]['val'])]
        sc = sum(w[j] for j in v) + g * sum(w[k] * J[j][k] for j in v for k in range(n) if keep[k] and k not in v)
        if sc > best + 1e-12: best, bi = sc, i
    return bi


def pick_E(S, g):
    I, U = ksets(S); w = []
    for i, a in enumerate(S):
        e = any(a['_v'] is not None and b['_v'] != a['_v'] and a['_v'] in I[j] for j, b in enumerate(S) if j != i)
        w.append(V.sig(a['rlg']) * (1 - g * e))
    return V.weighted_pick(S, w)


def pick_P(S, g): return V.weighted_pick(S, [V.sig(a['rlg']) ** g for a in S])


def acc(streams, f, g, fixed):
    return 100 * np.mean([fixed.get(k) if fixed.get(k) is not None else
                          (lambda i: S[i]['vok'] if i is not None else 0)(f(S, g)) for k, S in streams])


DATA = {}
for s in ['trainC', 'dev', 'dev2']:
    by, items = V.load_set(s)
    fixed = {(idx, sd): items[idx]['vote_ok'][str(sd)] for idx, sd, S in V.seed_streams(by)
             if not any(a['voted'] for a in by[idx] if a['seed'] == sd)}
    seeds = [((idx, sd), S) for idx, sd, S in V.seed_streams(by)]
    boot = [((idx, sd), S) for idx, sd, S in V.random_subsets(by, 8 if s == 'trainC' else 16, seed=1)]
    DATA[s] = (seeds, boot, fixed)

GR = dict(R=[0, 0.25, 0.5, 1, 2, 4], G=[0, 0.1, 0.25, 0.5, 1], E=[0, 0.25, 0.5, 0.75, 1.0], P=[0.5, 1, 1.5, 2, 3, 4])
F = dict(R=pick_R, G=pick_G, E=pick_E, P=pick_P)
for name, grid in GR.items():
    tr = {g: acc(DATA['trainC'][0] + DATA['trainC'][1], F[name], g, DATA['trainC'][2]) for g in grid}
    gb = max(tr, key=lambda g: (tr[g], -abs(g - grid[0])))
    out = ' '.join(f'{g:g}:{v:.2f}' for g, v in tr.items())
    ev = ' '.join(f'{s} seed {acc(DATA[s][0], F[name], gb, DATA[s][2]):.2f} boot {acc(DATA[s][1], F[name], gb, {}):.2f}'
                  for s in ('dev', 'dev2'))
    print(f'{name}: trainC grid {out} -> g={gb:g} | {ev}', flush=True)
