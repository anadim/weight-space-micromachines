"""lane sidenet: largest |R_sn - offline| attempts and [4]-mismatch rows. python3.11 aw/sn_tail.py SET"""
import gzip, json, glob, re, sys, os
import numpy as np, torch
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, '..'))
import repick_sn as RS
S = sys.argv[1]
P = torch.load(os.path.join(H, 'stk', 'sn_w.pt')); th = np.array(P['th']); c0 = P['c0']; names = P['feats']
sc = np.load(os.path.join(H, '..', 'job/out/sc_g4.npz'))[f'c4:HM:lin:pair|{S}']
tab = [json.loads(l) for l in gzip.open(os.path.join(H, '..', 'tables', f'attempts_{S}.jsonl.gz'), 'rt')]
off = {(t['seed'], t['idx'], t['k']): (float(v), t) for t, v in zip(tab, sc)}
L = []
for f in sorted(glob.glob(os.path.join(H, 'out', f'sn_{S}', '*.jsonl.gz'))):
    sd = int(re.search(r'seed(\d+)', f).group(1))
    for r in map(json.loads, gzip.open(f, 'rt')):
        for k in range(r['natt']):
            v = off.get((sd, r['idx'], k))
            if v and v[1]['hash_off'] is not None and r['rok'][k] and r['lg'][k] > -29.5:
                L.append((abs(r['sn'][k] - v[0]), sd, r['idx'], k, r['sn'][k], v[0], len(v[1]['seg_tokens']), v[1]['hash_off'], v[1]['nrep']))
        so = RS.St(r, [off.get((sd, r['idx'], k), (-30.0, 0))[0] for k in range(len(r['lg']))])
        po = RS.pick(so, names, th, c0)
        if po != r['pick']:
            print('PICKDIFF seed', sd, 'idx', r['idx'], 'in-stack', r['pick'], r['sym'][r['pick']] if r['pick'] >= 0 else None, 'offline', po,
                  'ok', r['ok'], 'sn', [round(x, 2) for x in r['sn'][:r['natt']]])
L.sort(reverse=True)
d = np.array([x[0] for x in L]); print('n', len(d), '>1:', (d > 1).sum(), '>0.5:', (d > .5).sum(), '>0.3:', (d > .3).sum())
for x in L[:5]: print([round(y, 3) if isinstance(y, float) else y for y in x])


def margin(r, sn):
    s = RS.St(r, sn)
    g = s.G(names) @ th + c0; o = np.sort(g)[::-1]
    return float(o[0] - o[1]) if len(o) > 1 else float('inf')


if len(sys.argv) > 2:     # margins of the in-stack / offline group scores on the [4]-mismatch rows
    for f in sorted(glob.glob(os.path.join(H, 'out', f'sn_{S}', '*.jsonl.gz'))):
        sd = int(re.search(r'seed(\d+)', f).group(1))
        for r in map(json.loads, gzip.open(f, 'rt')):
            sno = [off.get((sd, r['idx'], k), (-30.0, 0))[0] for k in range(len(r['lg']))]
            po = RS.pick(RS.St(r, sno), names, th, c0)
            if po != r['pick']:
                print('MARGIN seed', sd, 'idx', r['idx'], 'in-stack margin', round(margin(r, r['sn']), 4), 'offline margin', round(margin(r, sno), 4),
                      'in-stack ok', r['ok'], 'offline-pick ok', RS.RF.acc_row(RS.St(r, sno), po))
