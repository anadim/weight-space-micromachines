"""lane sidenet: in-stack AW_SN rows vs the offline side-net / writer (local). python3.11 aw/sn_check.py SN_ROWDIR REF_ROWDIR SET SC.npz CFG
[1] stream identity vs the reference (avf) rows: token-identical rows, and rows identical up to the final '####' line
[2] side-net register R_sn (committed slots with '####' and lg set) vs the offline extractor+side-net score of the same attempt
    (attempts table of the reference rows, keyed (seed, idx, k)): corr, MAE, p99, max
[3] in-stack pick vs the offline BLD writer recomputed from the row's own registers (repick_sn.St/pick, theta of sn_w.pt)
[4] pick agreement vs the offline re-pick that uses the OFFLINE side-net scores on the reference rows"""
import gzip, json, glob, os, re, sys
import numpy as np, torch
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, '..'))
import repick_sn as RS

D, REF, S, SC, CFG = sys.argv[1:6]
P = torch.load(os.path.join(H, 'stk', 'sn_w.pt')); th = np.array(P['th']); c0 = P['c0']; names = P['feats']


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, '*.jsonl.gz'))):
        m = re.match(r'(dev2|dev)_.*_seed(\d+)', os.path.basename(f))
        if m and m.group(1) == S: out[int(m.group(2))] = {r['idx']: r for r in map(json.loads, gzip.open(f, 'rt'))}
    return out


C, R = load(D), load(REF)
sc = np.load(SC)[f'{CFG}|{S}']
tab = [json.loads(l) for l in gzip.open(os.path.join(H, '..', 'tables', f'attempts_{S}.jsonl.gz'), 'rt')]
off = {(t['seed'], t['idx'], t['k']): (float(v), t['hash_off'] is not None and t['lg'] is not None and t['lg'] > -29.5) for t, v in zip(tab, sc)}
fin = lambda r: max(i for i, w in enumerate(r['wf']) if w & 8)          # start of the last written queue (the final '####' line)
ident = pre = 0; xs, ys = [], []; mm = n = mo = 0
for sd in sorted(C):
    for i, c in C[sd].items():
        r = R[sd][i]; n += 1
        ident += c['tokens'] == r['tokens']
        pre += c['tokens'][:fin(c)] == r['tokens'][:fin(r)] and c['natt'] == r['natt']
        for k in range(c['natt']):
            v = off.get((sd, i, k))
            if v and v[1] and c['rok'][k] and c['lg'][k] > -29.5: xs.append(c['sn'][k]); ys.append(v[0])
        st = RS.St(c, c['sn'])
        if c['pick'] != RS.pick(st, names, th, c0): mm += 1
        so = RS.St(r, [off.get((sd, i, k), (-30.0, 0))[0] for k in range(len(r['lg']))])
        if c['pick'] != RS.pick(so, names, th, c0): mo += 1
xs, ys = np.array(xs), np.array(ys); d = np.abs(xs - ys)
print(f'SNCHK {S} rows {n} | [1] token-identical {ident}, identical before the final line (and natt) {pre}')
print(f'SNCHK {S} [2] R_sn vs offline n={len(xs)} corr {np.corrcoef(xs, ys)[0, 1]:.5f} MAE {d.mean():.4f} p99 {np.quantile(d, .99):.4f} max {d.max():.4f}')
print(f'SNCHK {S} [3] pick vs offline BLD writer on the row registers: mismatches {mm}/{n}')
print(f'SNCHK {S} [4] pick vs offline re-pick with offline side-net scores: mismatches {mo}/{n}')
