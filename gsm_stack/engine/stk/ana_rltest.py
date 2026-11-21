"""RL_STACK test analysis. usage: python ana_rltest.py BASE.jsonl FILE.jsonl [FILE2 ...]
BASE = the all-weights u0 stack (same item order). Files from sw_eval_hook.py (serial: sym/rok per slot) or
rlb_test_hook.py (B: one attempt). Prints accuracy, paired-bootstrap 95% CI vs BASE, fixes / breaks, tok / item,
all-same rate, per-attempt accuracy, oracle."""
import sys, os, json, numpy as np
L = lambda f: [json.loads(l) for l in open(f)]
PAD = 12


def val(s):
    t = ''.join('0123456789-.'[x] for x in s if x != PAD)
    try: return float(t)
    except ValueError: return None


def boot(d, B=4000):
    rng = np.random.RandomState(0); d = np.asarray(d, float); n = len(d)
    m = np.array([d[rng.randint(0, n, n)].mean() for _ in range(B)])
    return [round(100 * float(np.percentile(m, 2.5)), 2), round(100 * float(np.percentile(m, 97.5)), 2)]


def boot1(d, B=4000):
    return boot(d, B)


base = L(sys.argv[1]); ob = np.array([x['ok'] for x in base], float)
for fn in sys.argv[2:]:
    R = L(fn); n = len(R); assert n == len(base), (fn, n, len(base))
    assert all(str(a['gold']) == str(b['gold']) for a, b in zip(R, base))
    ok = np.array([x['ok'] for x in R], float)
    res = dict(file=os.path.basename(fn), n=n, acc=round(100 * ok.mean(), 2), acc_ci=boot1(ok),
               tok_q=round(float(np.mean([x['ntok'] for x in R])), 1),
               vs_base=dict(delta=round(100 * (ok - ob).mean(), 2), ci=boot(ok - ob),
                            fixes=int(((ok == 1) & (ob == 0)).sum()), breaks=int(((ok == 0) & (ob == 1)).sum())))
    if 'sym' in R[0]:
        N = max(x['natt'] for x in R)
        g = [float(x['gold']) for x in R]
        V = [[val(x['sym'][i]) if x['rok'][i] else None for i in range(N)] for x in R]
        right = np.array([[v is not None and abs(v - gg) <= 1e-6 * max(1, abs(gg)) for v in vs] for vs, gg in zip(V, g)])
        dist = [len({v for v in vs if v is not None}) for vs in V]
        res.update(N=N, attempt_acc=[round(100 * float(right[:, i].mean()), 1) for i in range(N)],
                   oracle=round(100 * float(right.any(1).mean()), 2), all_same=round(100 * float(np.mean([d <= 1 for d in dist])), 1),
                   repairs_per_q=round(float(np.mean([x['nseg'] - x['natt'] for x in R])), 3),
                   no_winner=int(sum(x['win'] < 0 for x in R)))
    elif 'fire' in R[0]:
        res['fire_rate'] = round(100 * float(np.mean([x['fire'] for x in R])), 1)
    print(json.dumps(res))
