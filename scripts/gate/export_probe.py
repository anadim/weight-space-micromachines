"""Fit and save the probe that gates the rethink implant (the FITTED part of that implant).

Same features, same fit and same threshold rule as fit_detector.py ('probe L28' for the 'base' solutions): the host's
layer-28 state (Hugging Face hidden_states[28] = the final normed state, the LM head's input) at the '####' token of its
own greedy solution, standardized with train mean/std, scikit-learn LogisticRegression(C=0.01) on 1,500 GSM8K TRAINING
problems, label = first answer wrong; threshold = the train quantile (41-point grid) that maximizes TRAIN accuracy of
"rethink iff score >= thr". Test numbers are printed for comparison with results/gate/detector_summary.json only.

python scripts/gate/export_probe.py GEN.json results/rethink/rethink_probe_L28.pt   (GEN.json from gen_rethink.py)
"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import json, copy, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from micromachines.calcfmt import strip_solution
G = json.load(open(sys.argv[1])); OUT = sys.argv[2]; L = 28
MP = 'Qwen/Qwen3-0.6B-Base'; dev = 'cuda'
tok = AutoTokenizer.from_pretrained(MP); model = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.bfloat16).to(dev).eval()
shots = json.load(open('data/gsm8k/shots_v1.json')); P = ''
for s in shots:
    body, fin = strip_solution(s['solution'] + f"\n#### {s['answer']}"); P += f"Question: {s['question']}\nAnswer: {body}\n#### {fin}\n\n"
pid = tok(P, return_tensors='pt').input_ids.to(dev)
with torch.no_grad(): pkv = model(pid, use_cache=True).past_key_values


@torch.no_grad()
def feats(qs, sols):                                  # as in fit_detector.py, layer L only
    H = []
    for q, s in zip(qs, sols):
        if '####' not in s: H.append(None); continue
        body = s.split('####')[0] + '####'
        ids = tok(f"Question: {q}\nAnswer:" + body, return_tensors='pt', add_special_tokens=False).input_ids.to(dev)
        o = model(ids, past_key_values=copy.deepcopy(pkv), output_hidden_states=True, use_cache=True)
        H.append(o.hidden_states[L][0, -1].float().cpu())
    return H


def XY(split):
    H = feats(G[split]['q'], G[split]['base']); keep = [i for i, h in enumerate(H) if h is not None]
    return keep, torch.stack([H[i] for i in keep]).numpy(), np.array([not G[split]['base_ok'][i] for i in keep])


def gated(split, keep, s, thr):
    b, r = G[split]['base_ok'], G[split]['rethink_ok']; on = set(np.array(keep)[s >= thr].tolist())
    return sum(r[i] if i in on else b[i] for i in range(len(b))) / len(b), len(on) / len(b)


kt, Xt, yt = XY('train')
mu, sd = Xt.mean(0), Xt.std(0) + 1e-6                 # fit_detector.py standardizes per layer with the same mean/std
clf = LogisticRegression(C=0.01, max_iter=3000).fit((Xt - mu) / sd, yt)
sc = clf.predict_proba((Xt - mu) / sd)[:, 1]
ths = np.quantile(sc, np.linspace(0, 1, 41)); thr = max(ths, key=lambda t: gated('train', kt, sc, t)[0])
rep = {'train_auc': roc_auc_score(yt, sc), 'thr': float(thr), 'train': gated('train', kt, sc, thr)}
if 'test' in G:
    k2, X2, y2 = XY('test'); s2 = clf.predict_proba((X2 - mu) / sd)[:, 1]
    rep.update(test_auc=roc_auc_score(y2, s2), test=gated('test', k2, s2, thr),
               test_scores={'keep': list(map(int, k2)), 's': [float(x) for x in s2]})
print(json.dumps({k: v for k, v in rep.items() if k != 'test_scores'}))
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
torch.save({'w': torch.tensor(clf.coef_[0], dtype=torch.float64), 'b': float(clf.intercept_[0]),
            'mu': torch.tensor(mu, dtype=torch.float64), 'sd': torch.tensor(sd, dtype=torch.float64), 'thr': float(thr), 'layer': L,
            'fit': 'sklearn LogisticRegression(C=0.01) on GSM8K train (1,500 problems, gen_rethink.py), threshold max train accuracy',
            'report': rep}, OUT)
