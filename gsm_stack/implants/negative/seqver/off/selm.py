"""lane seqver: selection-relevant metrics per stream (seed, idx) on committed attempts of the attempts tables.
python3.11 off/selm.py off/sc_inc_TAG.npz [cfg ...]
 top1: accuracy of the argmax-score attempt of each stream (best-of-N by verifier alone, no votes), streams with >=1 right & >=1 wrong
 grp:  pairwise accuracy right answer-group vs each wrong answer-group of the same stream, group score = mean member score
 grpsum: same with group score = Σ sigmoid(score/2) (the writer's ssb feature; votes + verifier)"""
import sys, gzip, json, os, numpy as np
H = os.path.dirname(os.path.abspath(__file__))
SN = dict(np.load(f'{H}/../job/tables/sc_g4.npz'))
sig = lambda x: 1 / (1 + np.exp(-x))
T = {}
for S in ('trainC', 'dev', 'dev2'):
    rows = [json.loads(l) for l in gzip.open(f'{H}/../job/tables/attempts_{S}.jsonl.gz', 'rt')]
    lg = np.array([-30 if r['lg'] is None else r['lg'] for r in rows], float)
    cm = np.array([r['hash_off'] is not None and r['lg'] is not None and r['lg'] > -29.5 for r in rows])
    st = {}
    for i, r in enumerate(rows):
        if cm[i]: st.setdefault((r['seed'], r['idx']), []).append(i)
    ans = [r['ans'] for r in rows]; ok = np.array([r['ok'] for r in rows], bool)
    ss = []
    for key, ii in st.items():
        ii = np.array(ii); y = ok[ii]
        if not (y.any() and (~y).any()): continue
        g = {}
        for j in ii: g.setdefault(ans[j], []).append(j)
        ss.append((ii, y, [np.array(v) for v in g.values()], [ok[v[0]] for v in g.values()]))
    T[S] = dict(b=np.maximum(SN[f'c4:HM:lin:pair|{S}'], -12) + np.maximum(lg, -12), ss=ss)


def metrics(S, s):
    t1 = gm = gs = 0; ng = 0
    for ii, y, G, gy in T[S]['ss']:
        t1 += y[np.argmax(s[ii])]
        mr = [s[g].mean() for g, o in zip(G, gy) if o]; mw = [s[g].mean() for g, o in zip(G, gy) if not o]
        sr = [sig(s[g] / 2).sum() for g, o in zip(G, gy) if o]; sw = [sig(s[g] / 2).sum() for g, o in zip(G, gy) if not o]
        for a in mr:
            for b in mw: gm += (a > b) + .5 * (a == b); ng += 1
        for a in sr:
            for b in sw: gs += (a > b) + .5 * (a == b)
    n = len(T[S]['ss'])
    return t1 / n, gm / ng, gs / ng


sc = dict(np.load(sys.argv[1])); cfgs = sys.argv[2:] or sorted({k.split('|')[0] for k in sc})
print('streams with right & wrong:', {S: len(T[S]['ss']) for S in T})
print(f'{"score":40s} | top1 trainC dev dev2 | grp-mean trainC dev dev2 | grp-Σσ trainC dev dev2')
fmt = lambda r: ' | '.join(' '.join(f'{x[i]:.4f}' for x in r) for i in range(3))
print(f'{"b (rung 4)":40s} | ' + fmt([metrics(S, T[S]['b']) for S in T]), flush=True)
for c in cfgs:
    print(f'{c:40s} | ' + fmt([metrics(S, T[S]['b'] + sc[f'{c}|{S}']) for S in T]), flush=True)
