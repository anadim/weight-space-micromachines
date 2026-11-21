"""lane prmstep: per-line logistic step probe on line-end residuals (4 layers x 1024), trained on trainC labeled lines only.
5-fold item CV on trainC (idx % 5): OOF per-line logits for trainC, fold-mean logits for dev/dev2 (held out; their labels are used
only for a line-level diagnostic). Grid l2 x steps; out/sc_prm.npz: 'L2@STEP|S' -> per-line logits (all lines), plus line meta;
out/train_pl.json: per-config line AUROC (trainC OOF, dev, dev2 on labeled lines)."""
import os, json, argparse, time
import numpy as np, torch, torch.nn.functional as F
ap = argparse.ArgumentParser()
ap.add_argument('--root', default='$WSM_DATA/implants/prmstep')
ap.add_argument('--l2', default='1e-4,1e-3,1e-2,1e-1')
ap.add_argument('--steps', default='50,100,200,400,800')
ap.add_argument('--lr', type=float, default=1e-3)
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); dev = os.environ.get('PL_DEV', 'cuda'); OUT = os.environ.get('PL_OUT', HERE + '/out')
SETS = ('trainC', 'dev', 'dev2'); X, LN, MT = {}, {}, {}
for S in SETS:
    LN[S] = dict(np.load(f'{a.root}/{S}/lines.npz')); MT[S] = dict(np.load(f'{a.root}/{S}/meta.npz'))
    X[S] = torch.from_numpy(np.load(f'{a.root}/{S}/X.npy')).to(dev).view(len(LN[S]['att']), -1)
    print(S, X[S].shape, 'labeled', (LN[S]['lab'] >= 0).sum(), 'good', (LN[S]['lab'] == 1).sum(), flush=True)
lab = LN['trainC']['lab']; it_line = MT['trainC']['idx'][LN['trainC']['att']]; fold = it_line % 5
L = np.flatnonzero(lab >= 0)
Xl = X['trainC'][torch.tensor(L, device=dev)].float()
mu = Xl.mean(0); sd = Xl.std(0) + 1e-3
Z = (Xl - mu) / sd; del Xl
y = torch.tensor(lab[L], device=dev, dtype=torch.float32)


def score(S, w, c):
    out = []
    for i in range(0, X[S].shape[0], 65536):
        out.append(((X[S][i:i + 65536].float() - mu) / sd) @ w + c)
    return torch.cat(out).cpu().numpy()


def auc(s, yy):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    p = yy.sum(); n = len(yy) - p
    return float((r[yy == 1].sum() - p * (p + 1) / 2) / (p * n))


STEPS = sorted(map(int, a.steps.split(','))); RES = {}; SC = {}
for l2 in map(float, a.l2.split(',')):
    oof = {s: np.zeros(len(lab), np.float32) for s in STEPS}
    fm = {(S, s): np.zeros(X[S].shape[0], np.float32) for S in ('dev', 'dev2') for s in STEPS}
    for k in range(5):
        tr = torch.tensor(fold[L] != k, device=dev)
        w = torch.zeros(Z.shape[1], device=dev, requires_grad=True); c = torch.zeros((), device=dev, requires_grad=True)
        opt = torch.optim.Adam([w, c], lr=a.lr); st = 0
        for s in STEPS:
            while st < s:
                opt.zero_grad(); lo = F.binary_cross_entropy_with_logits(Z[tr] @ w + c, y[tr]) + l2 * (w * w).sum(); lo.backward(); opt.step(); st += 1
            with torch.no_grad():
                full = score('trainC', w, c); hm = np.flatnonzero(fold == k); oof[s][hm] = full[hm]
                for S in ('dev', 'dev2'): fm[(S, s)] += score(S, w, c) / 5
    for s in STEPS:
        key = f'{l2:g}@{s}'
        r = dict(trainC=auc(oof[s][L], lab[L]))
        for S in ('dev', 'dev2'):
            l_ = np.flatnonzero(LN[S]['lab'] >= 0); r[S] = auc(fm[(S, s)][l_], LN[S]['lab'][l_])
        RES[key] = r; SC[f'{key}|trainC'] = oof[s]; SC[f'{key}|dev'] = fm[('dev', s)]; SC[f'{key}|dev2'] = fm[('dev2', s)]
        print('line AUROC', key, json.dumps(r), flush=True)
for S in SETS:
    for f_ in ('att', 'lab', 'start'): SC[f'{f_}|{S}'] = LN[S][f_]
np.savez_compressed(f'{OUT}/sc_prm.npz', **SC)
json.dump(RES, open(f'{OUT}/train_pl.json', 'w'), indent=1)
print('TRAIN DONE', flush=True)
