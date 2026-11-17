"""Detector for the detect-and-rethink machine on Qwen3-0.6B-Base.
Hidden states at the '####' token of the model's own solution (every layer), for train / test / hard, for the 'base' and
'ocalc' solutions. Detectors fitted on TRAIN only:
  meandiff L : unit(mean(wrong) - mean(right)) at layer L, a closed-form direction (no training loop)
  probe L    : logistic regression (for comparison; this one IS fitted)
  margin     : low top-2 margin of the first answer digit (no fit at all)
Gate: apply the rethink where score >= threshold; threshold chosen on TRAIN (max train accuracy), then applied unchanged.
python scripts/gate/fit_detector.py GEN.json OUT.json   (GEN.json from gen_rethink.py)"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import json, sys, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from micromachines.calcfmt import strip_solution
G = json.load(open(sys.argv[1])); OUT = sys.argv[2]
MP = 'Qwen/Qwen3-0.6B-Base'; dev = 'cuda'
tok = AutoTokenizer.from_pretrained(MP); model = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.bfloat16).to(dev).eval()
shots = json.load(open('data/gsm8k/shots_v1.json')); P = ''
for s in shots:
    body, fin = strip_solution(s['solution'] + f"\n#### {s['answer']}"); P += f"Question: {s['question']}\nAnswer: {body}\n#### {fin}\n\n"
pid = tok(P, return_tensors='pt').input_ids.to(dev)
with torch.no_grad(): pkv = model(pid, use_cache=True).past_key_values
import copy
@torch.no_grad()
def feats(qs, sols):
    H, M = [], []
    for q, s in zip(qs, sols):
        if '####' not in s: H.append(None); M.append(None); continue
        body = s.split('####')[0] + '####'
        ids = tok(f"Question: {q}\nAnswer:" + body, return_tensors='pt', add_special_tokens=False).input_ids.to(dev)
        o = model(ids, past_key_values=copy.deepcopy(pkv), output_hidden_states=True, use_cache=True)
        H.append(torch.stack([h[0, -1].float() for h in o.hidden_states]).cpu())
        nx = model(tok(' ', return_tensors='pt', add_special_tokens=False).input_ids.to(dev), past_key_values=o.past_key_values).logits[0, -1].float()
        t2 = nx.topk(2).values; M.append(float(t2[0] - t2[1]))
    return H, M
out = {}
F = {}
for split in ('train', 'test', 'hard'):
    if split not in G: continue
    for src in ('base', 'ocalc'):
        F[split, src] = feats(G[split]['q'], G[split][src]); print('feats', split, src, flush=True)
def XY(split, src):
    H, M = F[split, src]; keep = [i for i, h in enumerate(H) if h is not None]
    return keep, torch.stack([H[i] for i in keep]).numpy(), np.array([not G[split][src + '_ok'][i] for i in keep]), np.array([M[i] for i in keep])
for src, rt in (('base', 'rethink'), ('ocalc', 'ocalc_rt')):
    kt, Xt, yt, Mt = XY('train', src); nL = Xt.shape[1]
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    dets = {}
    for L in range(nL):
        d = (Xt[yt, L].mean(0) - Xt[~yt, L].mean(0)); dets[f'meandiff L{L}'] = ('md', L, d)
    for L in range(4, nL, 4):
        clf = LogisticRegression(C=0.01, max_iter=3000).fit((Xt[:, L] - mu[L]) / sd[L], yt); dets[f'probe L{L}'] = ('lr', L, clf)
    def score(X, M, det):
        k, L, d = det
        if k == 'md': return X[:, L] @ d
        return d.predict_proba((X[:, L] - mu[L]) / sd[L])[:, 1]
    def gated(split, keep, s, thr):
        b, r = G[split][src + '_ok'], G[split][rt + '_ok']; on = set(np.array(keep)[s >= thr].tolist())
        return sum(r[i] if i in on else b[i] for i in range(len(b))) / len(b), len(on) / len(b)
    tr_auc = {name: roc_auc_score(yt, score(Xt, Mt, det)) for name, det in dets.items()}
    tr_auc['margin'] = roc_auc_score(yt, -Mt)
    best_md = max((n for n in tr_auc if n.startswith('meandiff')), key=tr_auc.get)
    best_lr = max((n for n in tr_auc if n.startswith('probe')), key=tr_auc.get)
    R = {'train_auc': {k: round(v, 3) for k, v in tr_auc.items()}}
    for name in (best_md, best_lr, 'margin'):
        sc_tr = -Mt if name == 'margin' else score(Xt, Mt, dets[name])
        ths = np.quantile(sc_tr, np.linspace(0, 1, 41)); bt = max(ths, key=lambda t: gated('train', kt, sc_tr, t)[0])
        R[name] = {'train': gated('train', kt, sc_tr, bt), 'thr': float(bt), 'train_scores': [float(x) for x in sc_tr]}
        for split in ('test', 'hard'):
            if (split, src) not in F: continue
            k2, X2, y2, M2 = XY(split, src); s2 = -M2 if name == 'margin' else score(X2, M2, dets[name])
            R[name][split] = gated(split, k2, s2, bt); R[name][split + '_auc'] = roc_auc_score(y2, s2)
            R[name][split + '_scores'] = {'keep': list(map(int, k2)), 's': [float(x) for x in s2]}
    for split in ('train', 'test', 'hard'):
        if split in G:
            b, r = G[split][src + '_ok'], G[split][rt + '_ok']; n = len(b)
            R[split + '_ref'] = {'no_rethink': sum(b) / n, 'always': sum(r) / n, 'oracle': sum(x or (y and not x) for x, y in zip(b, r)) / n}
    out[src] = R; print(src, json.dumps(R, indent=0)[:3000], flush=True)
json.dump(out, open(OUT, 'w'), indent=1)
