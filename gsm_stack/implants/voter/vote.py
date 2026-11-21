"""Voter study core: streams, the stack's voter (exact port), in-stream context features, voters, evaluation.

The stack voter (gen/stk/serial_stack.py `_act`, vmode 1, san on): keep = sane committed slots (valid & ~bad), or all
valid slots if none is sane; score(slot) = sum of w over kept slots with the same value (|Vi-Vj| <= 1e-6 max(1,|Vj|)),
w = sigmoid(lg); earliest best slot wins; no kept slot -> the stream ends with a bare eos (graded on the last earlier '####').
"""
import os, sys, math, random, pickle, gzip, json, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
NMAX = 8


def load_set(s):
    R = pickle.load(open(os.path.join(HERE, 'cache', f'att_{s}.pkl'), 'rb'))
    items = {}
    p = os.path.join(WS, 'atlas', 'data', f'items_{s}.jsonl.gz')
    if os.path.exists(p):
        for l in gzip.open(p):
            it = json.loads(l); items[it['idx']] = it
    by = collections.defaultdict(list)
    for r in R: by[r['idx']].append(r)
    for v in by.values(): v.sort(key=lambda r: (r['seed'], r['k']))
    return by, items


def seed_streams(by):
    """-> list of (idx, seed, [attempts committed to the register, k order])"""
    out = []
    for idx, A in sorted(by.items()):
        for sd in sorted({a['seed'] for a in A}):
            out.append((idx, sd, [a for a in A if a['seed'] == sd and a['committed']]))
    return out


def same_val(x, y):
    return abs(x - y) <= 1e-6 * max(1.0, abs(y))


def sig(x): return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))


# ---------------------------------------------------------------- generic weighted vote (the stack's rule)
def keep_mask(S, use_san=True):
    ok = [a['valid'] for a in S]
    sane = [o and not a['rbad'] for o, a in zip(ok, S)]
    return sane if (use_san and any(sane)) else ok


def weighted_pick(S, w, keep=None):
    """earliest slot with max sum_{kept j, same value} w_j; None if nothing kept"""
    keep = keep_mask(S) if keep is None else keep
    best, bi = -1e18, None
    for i, a in enumerate(S):
        if not keep[i]: continue
        sc = sum(w[j] for j, b in enumerate(S) if keep[j] and same_val(a['val'], b['val']))
        if sc > best + 1e-12: best, bi = sc, i
    return bi


def v_count(S, **kw): return weighted_pick(S, [1.0] * len(S))
def v_probe(S, **kw): return weighted_pick(S, [sig(a['rlg']) for a in S])


# ---------------------------------------------------------------- in-stream context features (vote time)
def _keys(a):
    if '_I' not in a:
        a['_I'] = {round(float(x), 6) for x in a['inter']}; a['_E'] = {round(float(x), 6) for x in a['eqset']}
        a['_v'] = round(a['val'], 6) if a['val'] is not None else None
    return a


def ctx(S):
    """per-attempt agreement features computed over the attempts of this stream only (register values)."""
    S = [_keys(a) for a in S]
    out = []
    for i, a in enumerate(S):
        oth = [b for j, b in enumerate(S) if j != i]
        no = max(1, len(oth))
        av = a['_v']
        diff = [b for b in oth if b['_v'] is not None and b['_v'] != av]
        U = [b['_I'] | ({b['_v']} if b['_v'] is not None else set()) for b in oth]
        UE = [b['_E'] | ({b['_v']} if b['_v'] is not None else set()) for b in oth]
        Ii, Ei = a['_I'], a['_E']
        agr_i = sum(len(Ii & u) for u in U) / (no * len(Ii)) if Ii else 0.0
        agr_e = sum(len(Ei & u) for u in UE) / (no * len(Ei)) if Ei else 0.0
        Ud = [b['_I'] | {b['_v']} for b in diff]
        agr_i_d = sum(len(Ii & u) for u in Ud) / (len(diff) * len(Ii)) if (Ii and diff) else 0.0
        a_in_o = sum(1 for b in diff if av is not None and av in b['_I']) / no
        o_in_a = sum(1 for b in diff if b['_v'] in Ii) / no
        c_same = sum(1 for b in oth if av is not None and b['_v'] == av) / no
        out.append(dict(c_same=c_same, agr_i=agr_i, agr_e=agr_e, agr_i_d=agr_i_d, a_in_o=a_in_o, o_in_a=o_in_a,
                        n_oth=len(oth)))
    return out


BASE = ['lg', 'sane', 'nrep', 'ntok', 'nsteps', 'lp_mean', 'lp_min', 'lp_n1', 'lp_ans', 'calc_bad', 'qcov',
        'ans_isq', 'ans_int', 'ans_log', 'ans_in_body', 'k']
CTX = ['agr_i', 'agr_e', 'agr_i_d', 'a_in_o', 'o_in_a']


def featvec(a, c, names):
    d = dict(lg=max(-12.0, a['rlg']), sane=float(a['valid'] and not a['rbad']), nrep=float(a['nrep'] > 0),
             ntok=a['ntok'] / 100.0, nsteps=a['nsteps'] / 10.0, lp_mean=a['lp_mean'] * 10, lp_min=a['lp_min'] / 5,
             lp_n1=min(a['lp_n1'], 20) / 5.0, lp_ans=max(-10.0, a['lp_ans']), calc_bad=float(a['calc_bad'] > 0),
             qcov=a['qcov'], ans_isq=float(a['ans_isq']), ans_int=float(a['ans_int']), ans_log=a['ans_log'] / 3,
             ans_in_body=float(a['ans_in_body']), k=a['k'] / 7.0)
    d.update({n: float(a.get(n, 0.0)) for n in names if n.startswith('pr_')})
    d.update(c)
    return [d[n] for n in names]


class Stream:
    """vectorised stream: F (n,d) features, keep mask, same matrix, ok per slot"""
    def __init__(self, S, names, idx=None, base_ok=None):
        self.S = S; self.idx = idx; self.base_ok = base_ok
        C = ctx(S)
        self.F = np.array([featvec(a, c, names) for a, c in zip(S, C)], dtype=float).reshape(len(S), len(names))
        self.keep = np.array(keep_mask(S), dtype=bool) if S else np.zeros(0, bool)
        n = len(S)
        self.same = np.array([[a['val'] is not None and b['val'] is not None and same_val(a['val'], b['val'])
                               for b in S] for a in S], dtype=bool).reshape(n, n)
        self.ok = np.array([a['vok'] for a in S], dtype=float)
        self.rep = np.array([self.keep[i] and not any(self.keep[j] and self.same[i, j] for j in range(i))
                             for i in range(n)], dtype=bool)

    def pick(self, w):
        if not self.keep.any(): return None
        sc = (self.same & self.keep[None, :]).astype(float) @ (w * self.keep)
        sc = np.where(self.keep, sc - 1e-9 * np.arange(len(sc)), -1e18)
        return int(np.argmax(sc))

    fixed = None

    def correct(self, i):
        if self.fixed is not None: return self.fixed
        if i is None: return self.base_ok if self.base_ok is not None else 0
        return int(self.ok[i])


def build(streams, names, base_ok=None):
    return [Stream(S, names, idx=idx, base_ok=(base_ok or {}).get((idx, sd))) for idx, sd, S in streams]


def random_subsets(by, n_sub, size=8, seed=0):
    """bootstrap streams: random size-subsets of each item's committed attempts (independent tries), k re-indexed"""
    rng = random.Random(seed)
    out = []
    for idx, A in sorted(by.items()):
        C = [a for a in A if a['committed']]
        for b in range(n_sub):
            sub = sorted(rng.sample(C, min(size, len(C))), key=lambda a: (a['seed'], a['k']))
            rng.shuffle(sub)
            out.append((idx, -1 - b, sub))
    return out
