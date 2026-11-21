"""one-parameter voter families on (v2 = hash_mnl_c4, lg): choose on trainC (OOF v2), report dev/dev2."""
import sys, os, numpy as np
sys.argv = [sys.argv[0]]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'offline.py')).read().split("NAMES = ")[0])
NAMES = ['lg', 'sane', 'pr_hash_mnl_c4']
D = {S: make(S, 0, names=NAMES)[0] for S in ('trainC', 'dev', 'dev2')}
def acc(st, f): return 100 * np.mean([x.correct(x.pick(f(x))) for x in st])
sig = lambda z: 1 / (1 + np.exp(-z))
def row(name, f): print(f'{name:40s}', ' '.join(f'{S} {acc(D[S], lambda x: f(x) if len(x.S) else np.zeros(0)):6.2f}' for S in D), flush=True)
row('stack sigmoid(lg)', lambda x: sig(x.F[:, 0]))
row('count', lambda x: np.ones(len(x.S)))
for a in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
    row(f'exp({a}*v2)', lambda x, a=a: np.exp(np.clip(a * x.F[:, 2], -30, 30)))
for c in (-2, -1, 0, 1, 2, 3):
    row(f'sigmoid(v2-{c})', lambda x, c=c: sig(x.F[:, 2] - c))
zb = lambda x: 0.3 * (np.maximum(x.F[:, 0], -10) - 1.287) / 1.528 + 0.7 * (x.F[:, 2] - 2.006) / 1.634
for a in (0.5, 1.0, 2.0, 3.0):
    row(f'exp({a}*blend)', lambda x, a=a: np.exp(np.clip(a * zb(x), -30, 30)))
for c in (-1, 0, 1):
    row(f'sigmoid(blend*1.5-{c})', lambda x, c=c: sig(1.5 * zb(x) - c))
