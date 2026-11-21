"""task [10](2) offline controls from serial-stack rows (final_eval fe_hook S rows: sym / rok / bad / lg per slot).
python3.11 offctl.py FILE.jsonl ... -> per file: stack vote (row 'ok'), voter off = last attempt only, first attempt only,
plain majority over valid attempt answers (ties: earliest), probe-argmax over valid attempts; 95% bootstrap CIs; paired
delta of each control vs the stack vote."""
import sys, json, numpy as np
from collections import Counter
PAD = 12


def val(s):
    t = ''.join('0123456789-.'[x] for x in s if x != PAD)
    try: return float(t)
    except ValueError: return None


def boot(d, B=4000):
    rng = np.random.RandomState(0); d = np.asarray(d, float); n = len(d)
    m = np.array([d[rng.randint(0, n, n)].mean() for _ in range(B)])
    return [round(100 * float(np.percentile(m, 2.5)), 2), round(100 * float(np.percentile(m, 97.5)), 2)]


def ok(v, g): return v is not None and abs(v - g) <= 1e-6 * max(1, abs(g))


for fn in sys.argv[1:]:
    R = [json.loads(l) for l in open(fn)]
    res = {k: [] for k in ('stack', 'last', 'first', 'majority', 'probe_argmax')}
    for x in R:
        g = float(x['gold']); n = x['natt']
        V = [val(x['sym'][i]) if x['rok'][i] else None for i in range(n)]
        res['stack'].append(int(x['ok']))
        res['last'].append(int(ok(V[n - 1], g)))
        res['first'].append(int(ok(V[0], g)))
        vv = [v for v in V if v is not None]
        res['majority'].append(int(ok(Counter(vv).most_common(1)[0][0], g)) if vv else 0)
        lg = [(x['lg'][i], -i) for i in range(n) if V[i] is not None]
        res['probe_argmax'].append(int(ok(V[-max(lg)[1]], g)) if lg else 0)
    st = np.array(res['stack'], float)
    print(f'{fn.split("/")[-1]} n={len(R)}')
    for k, a in res.items():
        a = np.array(a, float)
        print(f'  {k:13s} {100*a.mean():6.2f} {boot(a)}' + ('' if k == 'stack' else f'  Δ vs stack {100*(a-st).mean():+.2f} {boot(a-st)}'))
