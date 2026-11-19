"""Train the call-gate probe: a single linear reader of the host's post-norm hidden state at the '<tool_call>' step.
Fit L2 logistic regression on OWN opener-step states (label 1 = Team A/B tool question, 0 = our no-tool question that
spuriously opened), standardization + threshold folded into one (w, b) so the test-time gate is: veto iff w.h + b < 0.
Threshold chosen on TRAIN to keep >=97.5% of legitimate (positive) calls. AUROC reported on held-out own questions.

python cg_train_probe.py OUT_probe.pt PCT STATES1.pt [STATES2.pt ...]   (later files override by (qid, src))
"""
import sys, torch

OUT, PCT = sys.argv[1], float(sys.argv[2])
byk = {}
for fi, f in enumerate(sys.argv[3:]):
    for j, r in enumerate(torch.load(f)):
        byk[(r['qid'], r['src']) if r['label'] == 1 else (fi, j)] = r      # only re-collected positives override
recs = list(byk.values())
op = [r for r in recs if r['opened'] and r['h'] is not None]
X = torch.stack([r['h'] for r in op]).float()
y = torch.tensor([float(r['label']) for r in op])
split = [r['split'] for r in op]
tr = torch.tensor([s == 'train' for s in split]); te = ~tr
print(f'opened states: {len(op)}  (train {int(tr.sum())}, test {int(te.sum())})  pos {int(y.sum())} neg {int((1-y).sum())}')
for sp, m in (('train', tr), ('test', te)):
    print(f'  {sp}: pos={int(y[m].sum())} neg={int((1-y[m]).sum())}')

mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6
Xn = (X - mu) / sd
w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
opt = torch.optim.LBFGS([w, b], lr=0.5, max_iter=300, line_search_fn='strong_wolfe')
Xtr, ytr = Xn[tr], y[tr]
LAM = float(__import__("os").environ.get("CG_LAM", "1e-2"))
def closure():
    opt.zero_grad()
    z = Xtr @ w + b
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, ytr) + LAM * (w * w).sum()
    loss.backward(); return loss
opt.step(closure)

@torch.no_grad()
def auroc(scores, labels):
    order = torch.argsort(scores); ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float)
    pos = labels == 1; n1 = int(pos.sum()); n0 = len(labels) - n1
    if n1 == 0 or n0 == 0: return float('nan')
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

with torch.no_grad():
    z_all = Xn @ w + b
for sp, m in (('train', tr), ('test', te)):
    print(f'AUROC {sp}: {auroc(z_all[m], y[m]):.4f}  (n={int(m.sum())})')

# choose threshold theta on TRAIN: keep >=97.5% of positives (legit calls pass). veto iff z < theta.
with torch.no_grad():
    zt = z_all[tr]; pos_z = zt[y[tr] == 1].sort().values
    theta = float(pos_z[max(0, int(PCT / 100 * len(pos_z)) - 1)])      # PCT-th percentile of positive scores
    def rates(m):
        zz = z_all[m]; yy = y[m]
        pos_pass = float(((zz >= theta) & (yy == 1)).sum()) / max(1, int((yy == 1).sum()))
        neg_veto = float(((zz < theta) & (yy == 0)).sum()) / max(1, int((yy == 0).sum()))
        return pos_pass, neg_veto
    for sp, m in (('train', tr), ('test', te)):
        pp, nv = rates(m); print(f'  {sp} @theta={theta:.3f}: pos-pass={pp:.3f}  neg-veto={nv:.3f}')
    for s_ in sorted(set(r['src'] for r in op if r['label'] == 1)):
        idx = torch.tensor([i for i, r in enumerate(op) if r['src'] == s_ and r['split'] == 'test'])
        if len(idx): print(f'    pos-pass[{s_}] test = {float((z_all[idx] >= theta).float().mean()):.3f} (n={len(idx)})')
    # per-source neg veto on test
    for s in sorted(set(r['src'] for r in op if r['label'] == 0)):
        idx = torch.tensor([i for i, r in enumerate(op) if r['src'] == s and r['split'] == 'test'])
        if len(idx):
            v = float((z_all[idx] < theta).float().mean()); print(f'    neg-veto[{s}] test = {v:.3f} (n={len(idx)})')

# fold standardization + threshold: z - theta = (X-mu)/sd . w + b - theta = X.(w/sd) + (b - theta - (mu/sd).w)
with torch.no_grad():
    W = (w / sd).clone()
    B = float(b) - theta - float((mu / sd * w).sum())
    # check: X@W + B ~= z_all - theta
    chk = (X @ W + B) - (z_all - theta)
    print('fold max abs err', float(chk.abs().max()))
torch.save({'w': W, 'b': B, 'theta': theta, 'auroc_test': auroc(z_all[te], y[te]),
            'pct': PCT, 'files': sys.argv[3:], 'note': 'gate vetoes <tool_call> iff w.h + b < 0 (h = post-norm last hidden, fp32)'}, OUT)
print('saved probe to', OUT)
