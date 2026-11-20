"""Serial tries + voter: accuracy for N = 1..4 from one N=4 stream (attempt k's text does not depend on N, so the first
k attempts of the N=4 stream ARE an N=k stream; the final line is the voter over the first k register slots).
usage: python ser_analyze.py OUTDIR set mode [greedy_dump.pt]"""
import sys, json, numpy as np, torch
O, nm, mode = sys.argv[1:4]
import os; SUF = os.environ.get('SUF', '')
NN = int(os.environ.get('NN', 4))
D = torch.load(f'{O}/ser_{nm}_{mode}_N{NN}_v1{SUF}.pt'); R = D['reg']
gold = np.array([float(r['gold']) for r in D['rows']]); n = len(gold)
val, ok, bad, lg, end = R['val'].numpy(), R['ok'].numpy(), R['bad'].numpy(), R['lg'].numpy(), R['end'].numpy()
right = ok & (np.abs(val - gold[:, None]) <= 1e-6 * np.maximum(1, np.abs(gold[:, None])))


def vote(j, k, vm, sanity=True):
    idx = [i for i in range(k) if ok[j, i]]
    if sanity:
        s = [i for i in idx if not bad[j, i]]
        if s: idx = s
    if not idx: return None
    if vm == 2: return idx[-1]
    w = {i: (1 / (1 + np.exp(-lg[j, i])) if vm == 1 else 1.0) for i in idx}
    sc = {}
    for i in idx:
        sc[i] = sum(w[t] for t in idx if abs(val[j, t] - val[j, i]) <= 1e-6 * max(1, abs(val[j, i])))
    return max(idx, key=lambda i: (sc[i], -i))


res = dict(n=n, set=nm, mode=mode)
gd = torch.load(sys.argv[4]) if len(sys.argv) > 4 else None
if gd is not None: res['plain_greedy'] = 100 * float(np.mean(gd['ok'])); res['plain_greedy_tok'] = float(np.mean(gd['ntok']))
for k in ([1, 2, 3, 4] if NN == 4 else [1, 2, 4, 8]):
    for vm, vn in ((0, 'count'), (1, 'probe'), (2, 'last')):
        for san in (True, False):
            pk = [vote(j, k, vm, san) for j in range(n)]
            acc = np.mean([p is not None and right[j, p] for j, p in enumerate(pk)])
            res[f'N{k}_{vn}' + ('_san' if san else '')] = round(100 * float(acc), 2)
    res[f'N{k}_oracle'] = round(100 * float(right[:, :k].any(1).mean()), 2)
    res[f'N{k}_tok'] = round(float(np.mean(end[:, k - 1] + (6 if k > 1 else 0))), 1)
    dist = [len({round(v, 6) for v, o in zip(val[j, :k], ok[j, :k]) if o}) for j in range(n)]
    res[f'N{k}_distinct'] = round(float(np.mean(dist)), 3)
    res[f'N{k}_all_same'] = round(100 * float(np.mean([d <= 1 for d in dist])), 1)
res['attempt_acc'] = [round(100 * float(right[:, i].mean()), 2) for i in range(NN)]
res['same_as_first'] = [round(100 * float(np.mean([ok[j, i] and ok[j, 0] and abs(val[j, i] - val[j, 0]) < 1e-6 for j in range(n)])), 1) for i in range(NN)]
res['instream_final_acc'] = round(100 * float(np.mean([r['ok_last'] for r in D['rows']])), 2)
pk4 = [vote(j, NN, 1, True) for j in range(n)]
res['instream_vs_offline_agree'] = int(sum((r['ok_last'] == int(p is not None and right[j, p])) for j, (r, p) in enumerate(zip(D['rows'], pk4))))
res['tok_total_mean'] = float(np.mean([r['ntok'] for r in D['rows']]))
res['probe_auroc_by_attempt'] = []
from sklearn.metrics import roc_auc_score
for i in range(NN):
    m = ok[:, i]
    try: res['probe_auroc_by_attempt'].append(round(float(roc_auc_score(right[m, i], lg[m, i])), 3))
    except ValueError: res['probe_auroc_by_attempt'].append(None)
# failure buckets at N=4 (for the diagnosis track)
allw = ~right.any(1); agree_w = allw & np.array([len({round(v, 6) for v, o in zip(val[j], ok[j]) if o}) <= 1 for j in range(n)])
res['all_wrong'] = int(allw.sum()); res['all_wrong_and_agree'] = int(agree_w.sum())
print(json.dumps(res, indent=1))
json.dump(res, open(f'{O}/ser_res_{nm}_{mode}{SUF}.json', 'w'), indent=1)
np.save(f'{O}/ser_allwrong_{nm}_{mode}.npy', np.nonzero(allw)[0])
