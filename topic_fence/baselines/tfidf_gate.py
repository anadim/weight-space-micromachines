"""TF-IDF + LogReg topic-blocker gate baseline, trained on the SAME dev rows + role labels as the state probe,
threshold picked the SAME way (LOSO-by-style OOF: max TPR s.t. FPR_neg<=1%, FPR_hard<=5%). Gate-level fire rates on test.
usage: tfidf_gate.py F   (math|ml)   CPU only."""
import json, sys, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack

F = sys.argv[1]; other = 'ml' if F == 'math' else 'math'
dev = [json.loads(l) for l in open('data/dev_all.jsonl')]
test = [json.loads(l) for l in open('data/test.jsonl')]
mm = [json.loads(l) for l in open('data/mmlu_ret.jsonl')]

def role_dev(r):                                  # identical to s2_sweep.role
    c = r['clf_' + F]
    if r['label'] == 'drift': return None
    if r['fence'] == F and r['label'] == 'pos': return 'pos' if c > 0 else None
    if r['source'] == 'gsm8k_train': return 'pos' if F == 'math' else ('neg' if c < 0 else None)
    if r['source'] == 'alpaca_train': return 'pos' if c > 2 else ('neg' if c < -2 else None)
    if r['fence'] == F and r['label'] == 'hard': return 'hard' if c < 0 else None
    return 'neg' if c < 0 else None

R = [role_dev(r) for r in dev]; keep = [i for i, x in enumerate(R) if x is not None]
texts = [dev[i]['text'] for i in keep]; kind = np.array([R[i] for i in keep])
y = (kind == 'pos').astype(float); sty = np.array([dev[i]['style'] for i in keep])
print(F, 'dev pos', int((kind == 'pos').sum()), 'neg', int((kind == 'neg').sum()), 'hard', int((kind == 'hard').sum()), flush=True)

# TF-IDF features: word 1-2 grams + char_wb 3-5 grams (fit on DEV ONLY)
vw = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=60000)
vc = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=60000)
Xw = vw.fit_transform(texts); Xc = vc.fit_transform(texts); X = hstack([Xw, Xc]).tocsr()
STY = sorted(set(sty))

def fitLR(Xtr, ytr, C): return LogisticRegression(C=C, class_weight='balanced', max_iter=3000, solver='liblinear').fit(Xtr, ytr)

def oof_for(C):
    oof = np.zeros(len(y))
    for s in STY:
        te = sty == s
        clf = fitLR(X[~te], y[~te], C); oof[te] = clf.decision_function(X[te])
    return oof

best = None
for C in [0.3, 1.0, 3.0, 10.0]:
    oof = oof_for(C)
    a_all = roc_auc_score(y, oof)
    mh = (kind == 'pos') | (kind == 'hard'); a_h = roc_auc_score(y[mh], oof[mh])
    print('C', C, 'OOF AUC pos-vs-neghard', round(a_all, 4), 'pos-vs-hard', round(a_h, 4), flush=True)
    if best is None or a_all + a_h > best[0]: best = (a_all + a_h, C, oof)
_, C, oof = best
print('BEST C', C, flush=True)

def rates(t): return ((oof[kind == 'pos'] >= t).mean(), (oof[kind == 'neg'] >= t).mean(), (oof[kind == 'hard'] >= t).mean())
cands = np.sort(oof)
ok = [t for t in cands if rates(t)[1] <= 0.01 and rates(t)[2] <= 0.05]
thr = float(min(ok)) if len(ok) else float(cands[-1])
tpr, fn, fh = rates(thr)
print('threshold(logit)', round(thr, 4), 'OOF tpr', round(tpr, 4), 'fpr_neg', round(fn, 4), 'fpr_hard', round(fh, 4), flush=True)

# refit on ALL dev, apply to test
clf = fitLR(X, y, C)
def gate(tlist):
    if not tlist: return np.array([])
    Z = hstack([vw.transform(tlist), vc.transform(tlist)]).tocsr()
    return clf.decision_function(Z) >= thr

def role_test(r):                                 # from s6_judge.role (alpaca needs clf we don't have -> bucket all alpaca as 'ret_all')
    if r['source'] == 'alpaca_test': return 'ret_all'
    if r['fence'] == F: return {'pos': 'pos', 'neg': 'neg_nbr', 'hard': 'hard', 'drift': 'drift'}[r['label']]
    if r['fence'] == 'general': return 'neg_gen'
    if r['label'] == 'drift': return None
    return 'neg_other'

buck = {}
for r in test:
    ro = role_test(r)
    if ro is None: continue
    buck.setdefault(ro, []).append(r['text'])
    if ro == 'pos': buck.setdefault('pos:' + r.get('style', '?'), []).append(r['text'])
res = {}
for ro, ts in buck.items():
    g = gate(ts); res[ro] = (float(g.mean()), len(ts))
gm = gate([r['text'] for r in mm]); res['mmlu'] = (float(gm.mean()), len(mm))
print('\n=== TF-IDF gate fire rates, fence', F, '===')
for k in ['pos', 'pos:pub_gsm8k', 'pos:pub_math500', 'pos:pub_mmlu_mc', 'hard', 'neg_nbr', 'neg_gen', 'neg_other', 'ret_all', 'drift', 'mmlu']:
    if k in res: print('%-16s fire %.4f  n=%d' % (k, res[k][0], res[k][1]))
json.dump({'fence': F, 'C': C, 'thr_logit': thr, 'oof_tpr': tpr, 'oof_fpr_neg': fn, 'oof_fpr_hard': fh,
           'fire': {k: res[k] for k in res}}, open(f'results/tfidf_gate_{F}.json', 'w'), indent=1)
print('saved results/tfidf_gate_%s.json' % F, flush=True)
