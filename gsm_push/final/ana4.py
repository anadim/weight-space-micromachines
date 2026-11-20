"""S3-serial analysis. usage: python ana4.py ROWS_DIR SET FILE.jsonl [FILE2 ...]
ROWS_DIR holds fwd/<set>_S0.jsonl and fgtw/<set>_S2F_fwntf.jsonl (references, same item order)."""
import sys, os, json, numpy as np
D, SET = sys.argv[1], sys.argv[2]
L = lambda f: [json.loads(l) for l in open(f)]
S0 = L(f'{D}/fwd/{SET}_S0.jsonl')
fF = f'{D}/fgtw/{SET}_S2F_fwntf.jsonl'
S2F = L(fF) if os.path.exists(fF) else None


def boot(d, B=4000):
    rng = np.random.RandomState(0); d = np.asarray(d, float); n = len(d)
    m = np.array([d[rng.randint(0, n, n)].mean() for _ in range(B)])
    return [round(100 * float(np.percentile(m, 2.5)), 2), round(100 * float(np.percentile(m, 97.5)), 2)]


for fn in sys.argv[3:]:
    R = L(fn); n = len(R); assert n == len(S0)
    ok = np.array([x['ok'] for x in R], float); o0 = np.array([x['ok'] for x in S0], float)
    N = max(x['natt'] for x in R)
    g = np.array([float(x['gold']) for x in R]); V = np.array([x['val'] for x in R]); O = np.array([x['rok'] for x in R], bool)
    right = O & (np.abs(V - g[:, None]) <= 1e-6 * np.maximum(1, np.abs(g[:, None])))
    k = min(N, V.shape[1])
    dist = [len({round(v, 6) for v, o in zip(V[j, :k], O[j, :k]) if o}) for j in range(n)]
    res = dict(file=os.path.basename(fn), n=n, acc=round(100 * ok.mean(), 2), tok_q=round(float(np.mean([x['ntok'] for x in R])), 1),
               vs_S0=dict(delta=round(100 * (ok - o0).mean(), 2), ci=boot(ok - o0), fixes=int(((ok == 1) & (o0 == 0)).sum()), breaks=int(((ok == 0) & (o0 == 1)).sum())),
               attempt_acc=[round(100 * float(right[:, i].mean()), 1) for i in range(k)], oracle=round(100 * float(right[:, :k].any(1).mean()), 2),
               all_same=round(100 * float(np.mean([d <= 1 for d in dist])), 1), repairs_per_q=round(float(np.mean([x['nseg'] - x['natt'] for x in R])), 3),
               c2_rows=round(100 * float(np.mean([x['c2fires'] > 0 for x in R])), 1))
    if S2F is not None:
        of = np.array([x['ok'] for x in S2F], float)
        res['S2F_acc'] = round(100 * of.mean(), 2)
        res['vs_S2F'] = dict(delta=round(100 * (ok - of).mean(), 2), ci=boot(ok - of), fixes=int(((ok == 1) & (of == 0)).sum()), breaks=int(((ok == 0) & (of == 1)).sum()))
    print(json.dumps(res))
