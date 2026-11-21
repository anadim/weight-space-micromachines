"""M5 step 1: how similar are the first steps of the serial attempts, do they predict the answer, and what would a
first-step-distinct constraint do to oracle@8 / vote (resampling 8 of the item's 32 attempts = 4 seeds x 8).
usage: python3 ana1.py [src=fe|GENDIR] [R=200]  -> prints markdown tables"""
import sys, random, math, collections, itertools
sys.path.insert(0, '.')
import m5lib as L

SRC = sys.argv[1] if len(sys.argv) > 1 else 'fe'
R = int(sys.argv[2]) if len(sys.argv) > 2 else 200
SEEDS = [7, 11, 13, 17]


def keys(a):
    txt = L.TK.decode(a['ids'])
    fl = L.TK.decode(a['fl']).strip()
    ws = fl.split()
    return dict(L1=fl, W4=' '.join(ws[:4]), W8=' '.join(ws[:8]), EQ=L.first_eq(txt), N2=L.first_nums(txt, 2),
                L1N=L.first_nums(fl, 3))


KN = ['W4', 'W8', 'L1', 'L1N', 'N2', 'EQ']


def H(c):
    n = sum(c.values()); return -sum(v / n * math.log2(v / n) for v in c.values() if v)


def main():
    for st in ['dev', 'dev2']:
        D = L.load_set(SRC, st, SEEDS)
        n = len(D[SEEDS[0]])
        items = []
        for i in range(n):
            pool = []
            for s in SEEDS:
                r, atts = D[s][i]
                for a in atts:
                    a = dict(a); a['seed'] = s; a['key'] = keys(a); a['ans'] = a['val'] if a['ok'] else ('NA', s, a['k'])
                    pool.append(a)
            items.append(dict(pool=pool, rows=[D[s][i][0] for s in SEEDS]))
        print(f'\n## {st} (src={SRC}, {n} items, {len(SEEDS)} seeds)')
        # A. similarity within an item-seed (8 attempts) + B. predictiveness (pairs within the 32-pool)
        print('\n| key | distinct/8 (mean) | P(pair same key) | P(same ans / same key) | P(same ans / diff key) | H(ans|key)/H(ans) | same (mixed) | P(same ans / same key) mixed | / diff key mixed |')
        print('|---|---|---|---|---|---|---|---|---|')
        for kn in KN:
            dist = []; ps = []; sa_sk = [0, 0]; sa_dk = [0, 0]; hk = h0 = 0; pm = []; m_sk = [0, 0]; m_dk = [0, 0]
            for it in items:
                pool = it['pool']
                nr = sum(a['right'] for a in pool); mixed = 0 < nr < len(pool)
                for s in SEEDS:
                    ks = [a['key'][kn] for a in pool if a['seed'] == s]
                    dist.append(len(set(ks)))
                for a, b in itertools.combinations(pool, 2):
                    sk = a['key'][kn] == b['key'][kn]; sa = a['ans'] == b['ans']
                    ps.append(sk)
                    t = sa_sk if sk else sa_dk; t[0] += sa; t[1] += 1
                    if mixed:
                        pm.append(sk); t = m_sk if sk else m_dk; t[0] += sa; t[1] += 1
                ca = collections.Counter(str(a['ans']) for a in pool); h0 += H(ca)
                byk = collections.defaultdict(list)
                for a in pool: byk[a['key'][kn]].append(str(a['ans']))
                hk += sum(len(v) / len(pool) * H(collections.Counter(v)) for v in byk.values())
            f = lambda t: f'{t[0] / max(1, t[1]):.3f}'
            print(f'| {kn} | {sum(dist) / len(dist):.2f} | {sum(ps) / len(ps):.3f} | {f(sa_sk)} | {f(sa_dk)} | {hk / h0:.3f} | {sum(pm) / max(1, len(pm)):.3f} | {f(m_sk)} | {f(m_dk)} |')
        # C. gap cases: vote of a seed wrong although one of its attempts is right
        print('\n| key | gap cases | some right att key not used by any winner-answer att (same seed) | ... (any att of 32 with winner answer) | right atts with key unique in seed |')
        print('|---|---|---|---|---|')
        for kn in KN:
            g = c1 = c2 = 0; ru = rt = 0
            for it in items:
                for s in SEEDS:
                    atts = [a for a in it['pool'] if a['seed'] == s]
                    w = L.vote(atts)
                    if (w is not None and w['right']) or not any(a['right'] for a in atts): continue
                    g += 1
                    wk = {a['key'][kn] for a in atts if w is not None and a['ans'] == w['ans']}
                    wk32 = {a['key'][kn] for a in it['pool'] if w is not None and a['ans'] == w['ans']}
                    rk = [a['key'][kn] for a in atts if a['right']]
                    c1 += any(k not in wk for k in rk); c2 += any(k not in wk32 for k in rk)
                    cnt = collections.Counter(a['key'][kn] for a in atts)
                    ru += sum(cnt[k] == 1 for k in rk); rt += len(rk)
            print(f'| {kn} | {g} | {c1} ({100 * c1 / max(1, g):.0f}%) | {c2} ({100 * c2 / max(1, g):.0f}%) | {ru}/{rt} |')
        # D. ceiling: resample 8 of 32
        rng = random.Random(0)
        print('\n| policy | vote | oracle@8 | per-att | distinct ans/8 | distinct key/8 |')
        print('|---|---|---|---|---|---|')

        def draw(pool, kn, mode):
            rest = list(range(len(pool))); rng.shuffle(rest); seq = []; used = set()
            for t in range(8):
                pick = None
                if kn and kn != 'ANS' and t > 0:
                    agree = len({str(pool[j]['ans']) for j in seq}) == 1
                    if mode == 'all' or (mode == 'agree' and agree):
                        for j in rest:
                            if pool[j]['key'][kn] not in used: pick = j; break
                if mode and mode.startswith('plur') and t >= 2:
                    tau = float(mode[4:]); c = collections.Counter(str(pool[j]['ans']) for j in seq if pool[j]['ok'])
                    if c:
                        pa_, pc = c.most_common(1)[0]
                        G = [pool[j] for j in seq if str(pool[j]['ans']) == pa_]
                        if pc >= 2 and sum(g['lg'] for g in G) / len(G) < tau:
                            gk = {g['key'][kn] for g in G}
                            for j in rest:
                                if pool[j]['key'][kn] not in gk: pick = j; break
                if kn == 'ANS' and t > 0:
                    for j in rest:
                        if str(pool[j]['ans']) not in {str(pool[x]['ans']) for x in seq}: pick = j; break
                if pick is None: pick = rest[0]
                rest.remove(pick); seq.append(pick)
                if kn and kn != 'ANS': used.add(pool[pick]['key'][kn])
            return [dict(pool[j], k=t) for t, j in enumerate(seq)]

        pols = [('random 8 of 32', None, None)] + [(f'{kn}-distinct', kn, 'all') for kn in KN] + \
               [(f'{kn}-distinct if prev agree', kn, 'agree') for kn in ['W4', 'L1', 'EQ']] + [('answer-distinct (bound)', 'ANS', 'all')] + \
               [(f'{kn}: avoid plurality-group keys if its mean lg < {tau}', kn, f'plur{tau}') for kn in ['EQ', 'L1', 'W4'] for tau in [0.0, 1.0, 2.0, 99.0]]
        for name, kn, mode in pols:
            v = o = pa = da = dk = 0; m = 0
            for it in items:
                for _ in range(R):
                    seq = draw(it['pool'], kn, mode)
                    w = L.vote(seq); v += bool(w and w['right']); o += any(a['right'] for a in seq)
                    pa += sum(a['right'] for a in seq) / 8; da += len({str(a['ans']) for a in seq})
                    dk += len({str(a['key'][kn if kn in KN else 'L1']) for a in seq}); m += 1
            print(f'| {name} | {100 * v / m:.2f} | {100 * o / m:.2f} | {100 * pa / m:.2f} | {da / m:.2f} | {dk / m:.2f} |')
        # actual seeds
        v = sum(bool(L.vote([a for a in it['pool'] if a['seed'] == s]) and L.vote([a for a in it['pool'] if a['seed'] == s])['right'])
                for it in items for s in SEEDS) / (len(items) * len(SEEDS))
        o = sum(any(a['right'] for a in it['pool'] if a['seed'] == s) for it in items for s in SEEDS) / (len(items) * len(SEEDS))
        print(f'| actual seeds (serial order) | {100 * v:.2f} | {100 * o:.2f} | | | |')


main()
