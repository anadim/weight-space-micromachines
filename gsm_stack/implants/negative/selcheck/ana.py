"""selcheck step-1 analysis (pure python): within-item AUROC of the check scores vs lg, and the vote simulation.
Inputs: job/out/fz_{set}.jsonl.gz (fz.py), job/out/fzp_{set}.json (fit.py, optional), data/streams_{set}.jsonl.gz (cands.py).
usage: python3 ana.py [--sim KEY[,KEY..]]"""
import gzip, json, math, os, sys, collections, itertools, random
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
from cands import correct, base_vote
SETS = ('trainC', 'dev', 'dev2')
sig = lambda v: 1 / (1 + math.exp(-max(-50, min(50, v))))


def within_auc(s, ok, idx):
    by = {}
    for v, o, i in zip(s, ok, idx):
        if v is None: continue
        by.setdefault(i, ([], []))[0 if o else 1].append(v)
    w = n = 0.0
    for r, wr in by.values():
        for a in r:
            for b in wr: w += (a > b) + 0.5 * (a == b); n += 1
    return w / max(n, 1), int(n)


def load(S):
    fs = [f'{H}/job/out/fz_{S}.jsonl.gz'] if os.path.exists(f'{H}/job/out/fz_{S}.jsonl.gz') else \
        [f'{H}/job/out/fz_{S}_p{i}.jsonl.gz' for i in range(2)]       # same p0, p1 order as fit.py
    rows = [json.loads(l) for f in fs for l in gzip.open(f)]
    p = f'{H}/job/out/fzp_{S}.json'
    P = json.load(open(p)) if os.path.exists(p) else {}
    if os.path.exists(f'{H}/job/out/fzo_{S}.json'): P.update(json.load(open(f'{H}/job/out/fzo_{S}.json')))   # xfit.py OOF probes
    st = [json.loads(l) for l in gzip.open(f'{H}/data/streams_{S}.jsonl.gz')]
    return rows, P, st


def scalars(r, P, i):
    """per-candidate score dict; sampled formats -> list of K."""
    d = {}
    for k in ('v2', 'v3', 'dir_sum', 'dir_mean'):
        if k in r: d[k] = r[k]
    for F in ('fwd', 'bwd'):
        if F in r:
            d[F + '_v'] = [s['v'] for s in r[F]]
            d[F + '_agree'] = [float(s['agree']) for s in r[F]]
            d[F + '_hasy'] = [float(s['y'] is not None) for s in r[F]]
            d[F + '_va'] = [s['v'] + 2 * s['agree'] for s in r[F]]
    for k, v in P.items(): d[k] = v[i]
    return d


def lg_feats(st):
    """candidate-level lg aggregates over all streams of the set: mean / max lg over slots of that value."""
    acc = collections.defaultdict(list)
    for s in st:
        for x, b, lg in s['slots']: acc[(s['idx'], x)].append(lg)
    return {k: (sum(v) / len(v), max(v), len(v)) for k, v in acc.items()}


def auc_table(sets):
    out = {}
    for S in sets:
        rows, P, st = load(S)
        D = [scalars(r, P, i) for i, r in enumerate(rows)]
        ok = [r['ok'] for r in rows]; idx = [r['idx'] for r in rows]
        LG = lg_feats(st)
        res = {}
        res['lg_mean'] = within_auc([LG.get((r['idx'], r['x']), (None,))[0] for r in rows], ok, idx)
        res['lg_max'] = within_auc([LG.get((r['idx'], r['x']), (None, None))[1] for r in rows], ok, idx)
        res['count'] = within_auc([LG.get((r['idx'], r['x']), (None, None, None))[2] for r in rows], ok, idx)
        for k in D[0]:
            v = [d[k] for d in D]
            if isinstance(v[0], list):
                K = len(v[0])
                res[k + '@1'] = (sum(within_auc([x[j] for x in v], ok, idx)[0] for j in range(K)) / K, 0)
                res[k + '@K'] = within_auc([sum(x) / K for x in v], ok, idx)
            else:
                res[k] = within_auc(v, ok, idx)
        out[S] = res
    keys = list(out[sets[0]])
    print('within-item AUROC (right vs wrong candidate of the same item); n pairs in ()')
    print('%-22s' % 'score' + ''.join('%16s' % S for S in sets))
    for k in keys:
        print('%-22s' % k + ''.join('%16s' % ('%.3f (%d)' % out[S][k] if k in out[S] else '-') for S in sets))
    return out


# ---------------- vote simulation ----------------
def groups(sl):
    """kept slots (sane if any sane) -> OrderedDict value -> [count, W=sum sigmoid(lg), max lg]"""
    idx = list(range(len(sl))); s = [i for i in idx if not sl[i][1]]
    if s: idx = s
    g = collections.OrderedDict()
    for i in idx:
        x, b, lg = sl[i]
        if x not in g: g[x] = [0, 0.0, -99.0]
        g[x][0] += 1; g[x][1] += sig(lg); g[x][2] = max(g[x][2], lg)
    return g


def sim(S, key, beta, tau, K, mu, sd, CH, wmode='share'):
    """returns per-stream (base_ok, new_ok, fired)."""
    _, _, st = DATA[S]
    seeds = sorted({s['seed'] for s in st}); si = {s: j for j, s in enumerate(seeds)}
    out = []
    for s in st:
        g = groups(s['slots'])
        if not g: out.append((0, 0, 0)); continue
        items = list(g.items()); tot = sum(v[1] for _, v in items)
        best = max(items, key=lambda kv: round(kv[1][1], 6))
        bok = int(correct(best[0], s['gold']))
        share = best[1][1] / tot
        new = best[0]; fired = 0
        if share < tau and len(items) > 1:
            top = sorted(items, key=lambda kv: -kv[1][1])[:K]
            sc = []
            for x, v in top:
                c = CH.get((s['idx'], x))
                if c is None: z = 0.0
                else:
                    c = c[si[s['seed']] % len(c)] if isinstance(c, list) else c
                    z = (c - mu) / sd
                sc.append((v[1] / tot + beta * z, x))
            new = max(sc, key=lambda t: t[0])[1]; fired = 1
        out.append((bok, int(correct(new, s['gold'])), fired))
    return out


def check_map(S, key):
    rows, P, st = DATA[S]
    m = {}
    for i, r in enumerate(rows):
        d = scalars(r, P, i)
        if key in d: m[(r['idx'], r['x'])] = d[key]
        elif '+' in key:
            ks = key.split('+'); m[(r['idx'], r['x'])] = None
    return m


def run_sim(keys):
    for key in keys:
        CHs = {S: check_map(S, key) for S in SETS}
        vals = [v for c in CHs['trainC'].values() for v in (c if isinstance(c, list) else [c])]
        mu = sum(vals) / len(vals); sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** .5 + 1e-9
        best = None
        for beta, tau, K in itertools.product([0, .05, .1, .2, .3, .5, 1, 2, 5], [.5, .6, .7, .8, .9, 1.01], [2, 3]):
            r = sim('trainC', key, beta, tau, K, mu, sd, CHs['trainC'])
            acc = sum(t[1] for t in r) / len(r)
            if best is None or acc > best[0] + 1e-12: best = (acc, beta, tau, K)
        acc, beta, tau, K = best
        line = f'{key:18s} trainC fit beta={beta} tau={tau} K={K}: trainC {100 * acc:.2f}'
        for S in ('trainC', 'dev', 'dev2'):
            r = sim(S, key, beta, tau, K, mu, sd, CHs[S])
            b = sum(t[0] for t in r) / len(r); n = sum(t[1] for t in r) / len(r)
            fx = sum(1 for t in r if t[1] > t[0]); br = sum(1 for t in r if t[1] < t[0]); fr = sum(t[2] for t in r) / len(r)
            line += f' | {S} {100 * b:.2f}->{100 * n:.2f} ({100 * (n - b):+.2f}, +{fx}/-{br}, fire {100 * fr:.0f}%)'
        print(line, flush=True)


DATA = {}
if __name__ == '__main__':
    for S in SETS:
        if os.path.exists(f'{H}/job/out/fz_{S}.jsonl.gz') or os.path.exists(f'{H}/job/out/fz_{S}_p1.jsonl.gz'): DATA[S] = load(S)
    auc_table([S for S in SETS if S in DATA])
    if '--sim' in sys.argv and all(S in DATA for S in SETS):
        run_sim(sys.argv[sys.argv.index('--sim') + 1].split(','))
