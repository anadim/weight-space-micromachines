"""Line-rewind gate (GPU, the B200 node). (1) refit the step-error detector (hdh = [h, h - h_prev_line_end], layers 18+22, logistic,
l2=0.1, trainC, as probe/divprobe.py task a) and save it; (2) score every reasoning line end (step >= 1, before '####') of
every dev/dev2 attempt; (3) for the first flagged line of each attempt (thresholds at train-OOF line FPR 1/2/5%) and for
random control lines, teacher-force pre + C.q_ids(q) + seg_tokens[:prev_line_end+1] through the host (HF Qwen3-0.6B-Base +
ckpt_a2s2) and sample K continuations at T=1 (plain HF generate, NO calc implant) up to the '#### n' line / blank line;
grade with C.correct. Outputs -> ./out/.
"""
import os, sys, json, gzip, time, argparse, collections, re
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C, fit

ap = argparse.ArgumentParser()
ap.add_argument('--sets', default='dev,dev2')
ap.add_argument('--ckpt', default='$WSM_DATA/gsm_sel/ckpt_a2s2')
ap.add_argument('--K', type=int, default=4)
ap.add_argument('--max_new', type=int, default=400)
ap.add_argument('--nctrl', type=int, default=500)
ap.add_argument('--tok_budget', type=int, default=500000)
ap.add_argument('--max_rows', type=int, default=1024)
ap.add_argument('--gen_secs', type=float, default=2700)
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--only_ctrl', type=int, default=0)
ap.add_argument('--out_name', default='gen')
a = ap.parse_args()
T0 = time.time(); LS = [18, 22]; FPRS = (0.01, 0.02, 0.05)
def log(*x): print(f'[{time.time() - T0:6.0f}s]', *x, flush=True)
os.makedirs(f'{HERE}/out', exist_ok=True)
dev = 'cuda'


def prep(S, labels=True):
    A = dict(np.load(f'{fit.ROOT}/{S}/attempts.npz'))
    L_ = [json.loads(l) for l in gzip.open(f'{fit.ROOT}/{S}/attempts_light.jsonl.gz', 'rt')]
    pl = np.array([int(r.get('phrase_len') or 0) for r in L_]); gid = [r['gid'] for r in L_]
    pm = np.load(f'{fit.ROOT}/{S}/pos_meta.npz'); k = pm['kind'] == 1; att = pm['att'][k]; off = pm['off'][k]
    ho = A['hash_off'][att]; mask = (off >= pl[att]) & ((ho < 0) | (off < ho))
    X, m = fit.load(S, LS, 'nl', rows=mask); X = X.reshape(len(X), -1)
    att = m['att'].astype(int); off = m['off'].astype(int)
    per = collections.defaultdict(list)
    for r in np.lexsort((off, att)): per[int(att[r])].append(int(r))
    step = np.zeros(len(att), int); prev = np.full(len(att), -1)
    for i, rr in per.items():
        for t, r in enumerate(rr): step[r] = t; prev[r] = rr[t - 1] if t else -1
    ok = A['ok'].astype(int); idx = A['idx']
    pr = collections.defaultdict(list)
    for i, o in zip(idx, ok): pr[int(i)].append(o)
    mixed = {i for i, v in pr.items() if 0 < np.mean(v) < 1}
    D = {}
    if labels:
        gi = {g: i for i, g in enumerate(gid)}
        for l in gzip.open(f'{HERE}/tables/divlab_{S}.jsonl.gz', 'rt'):
            d = json.loads(l); i = gi.get(d['gid'])
            if i is None or i not in per: continue
            offs = off[per[i]]; e1 = np.flatnonzero(offs >= d['dt']); e0 = np.flatnonzero(offs < d['dl'])
            if not len(e1): continue
            D[i] = (int(e0[-1]) if len(e0) else -1, int(e1[0]), d['cat'])
    log(S, 'rows', len(att), 'attempts', len(gid), 'wrong with D', len(D), 'mixed items', len(mixed))
    return dict(X=X, att=att, off=off, per=per, step=step, prev=prev, ok=ok, idx=idx, mixed=mixed, D=D, gid=gid, pl=pl)


def hdh(d, rows):
    h = d['X'][rows].astype(np.float32)
    return np.concatenate([h, h - d['X'][d['prev'][rows]].astype(np.float32)], 1)


# ---------------- (1) detector on trainC
d = prep('trainC'); rows, y = [], []
for i, rr in d['per'].items():
    if d['idx'][i] not in d['mixed']: continue
    if d['ok'][i]: rows += rr[1:]; y += [0] * (len(rr) - 1)
    elif i in d['D']:
        t0, t1, _ = d['D'][i]
        if t1 < 1: continue
        neg = [rr[t] for t in range(1, t0 + 1)]; rows += neg + [rr[t1]]; y += [0] * len(neg) + [1]
rows = np.array(rows); y = np.array(y, np.float32); X = hdh(d, rows)
w = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean())); g = d['idx'][d['att'][rows]]
ug = np.unique(g); rng = np.random.default_rng(0); rng.shuffle(ug); fm = {q: i % 5 for i, q in enumerate(ug)}
f = np.array([fm[q] for q in g]); oof = np.zeros(len(y), np.float32)
for k in range(5):
    p, _, _ = fit.fit_logreg(X[f != k], y[f != k], l2=0.1, w=w[f != k]); oof[f == k] = p(X[f == k])
p, W, b = fit.fit_logreg(X, y, l2=0.1, w=w)
thr = {fp: float(np.quantile(oof[y == 0], 1 - fp)) for fp in FPRS}
log('detector trainC rows', len(y), 'pos', int(y.sum()), 'oof auc', fit.auroc(y, oof), 'thr', thr)
torch.save(dict(W=W.float(), b=b, thr=thr, layers=LS, feat='hdh: [h18,h22, h18-h18prev, h22-h22prev] (layer inputs, nl rows)',
                oof_auc=fit.auroc(y, oof), l2=0.1), f'{HERE}/out/detector.pt')
del d, X

# ---------------- (2) score dev/dev2 lines
SC = {}; JOBS = []
rng = np.random.default_rng(1)
for S in a.sets.split(','):
    d = prep(S, labels=False)
    rr = np.flatnonzero(d['step'] >= 1); s = p(hdh(d, rr))
    tabs = {}
    for l in gzip.open(f'{HERE}/tables/attempts_{S}.jsonl.gz', 'rt'):
        r = json.loads(l); tabs[r['gid']] = r
    np.savez_compressed(f'{HERE}/out/scores_{S}.npz', att=d['att'][rr], step=d['step'][rr], off=d['off'][rr],
                        prev_off=d['off'][d['prev'][rr]], score=s, ok=d['ok'][d['att'][rr]], gids=np.array(d['gid']))
    byatt = collections.defaultdict(list)
    for j, r in enumerate(rr): byatt[int(d['att'][r])].append((int(d['step'][r]), int(d['off'][d['prev'][r]]), float(s[j])))
    want = collections.defaultdict(set)                          # (att, prev_off) -> tags
    for i, L in byatt.items():
        L.sort()
        for fp in FPRS:
            hit = [x for x in L if x[2] > thr[fp]]
            if hit: want[(i, hit[0][1])].add(f'f{fp}')
    unfl = [i for i, L in byatt.items() if not any(x[2] > thr[0.05] for x in L)]
    for i in rng.choice(unfl, min(a.nctrl, len(unfl)), replace=False):
        L = byatt[int(i)]; x = L[rng.integers(len(L))]; want[(int(i), x[1])].add('ctrl')
    for (i, po), tags in want.items():
        t = tabs[d['gid'][i]]; seg = json.loads(t['seg_tokens']) if isinstance(t['seg_tokens'], str) else t['seg_tokens']
        pri = 0 if 'f0.02' in tags else (1 if 'ctrl' in tags else 2)
        JOBS.append(dict(set=S, att=i, gid=d['gid'][i], idx=int(t['idx']), seed=t['seed'], k=t['k'], q=t['q'], gold=t['gold'],
                         ok=int(t['ok']), ans=t['ans'], prev_off=po, tags=sorted(tags), pri=pri, prefix=seg[:po + 1], seg_len=len(seg)))
    log(S, 'scored', len(rr), 'jobs so far', len(JOBS), {fp: sum(any(x[2] > thr[fp] for x in L) for L in byatt.values()) for fp in FPRS})
    del d

# ---------------- (3) generate
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from safetensors.torch import load_file
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
pre, _ = C.prefix_ids(tok); Lp = len(pre)
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', dtype=torch.bfloat16, attn_implementation='sdpa').to(dev).eval()
sd = load_file(f'{a.ckpt}/model.safetensors', device='cpu'); msd = m.state_dict()
with torch.no_grad():
    for k_ in [k_ for k_ in msd if k_ != 'lm_head.weight']: msd[k_].copy_(sd[k_].float().to(torch.bfloat16).to(dev))
    if 'lm_head.weight' in sd and m.lm_head.weight.data_ptr() != m.model.embed_tokens.weight.data_ptr():
        m.lm_head.weight.copy_(sd['lm_head.weight'].float().to(torch.bfloat16).to(dev))
del sd
log('host loaded Lp', Lp, 'jobs', len(JOBS), collections.Counter(j['pri'] for j in JOBS))
V = len(tok); NL = torch.zeros(m.config.vocab_size, dtype=torch.bool)
for t_ in range(V):
    if '\n' in tok.decode([t_]): NL[t_] = True
NL = NL.to(dev); PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
END = re.compile(r'####\s*-?[\d,]*\.?\d+[^\n]*\n|\n\s*\n')


class Stop(StoppingCriteria):
    def __init__(s, plen): s.plen = plen; s.done = None
    def __call__(s, ids, scores, **kw):
        if s.done is None: s.done = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
        cand = NL[ids[:, -1]] & ~s.done
        for r in torch.nonzero(cand).flatten().tolist():
            if END.search(tok.decode(ids[r, s.plen:])): s.done[r] = True
        return s.done.clone()


order = sorted(range(len(JOBS)), key=lambda i: (JOBS[i]['pri'], len(C.q_ids(tok, JOBS[i]['q'])) + len(JOBS[i]['prefix'])))
if a.only_ctrl: order = [i for i in order if 'ctrl' in JOBS[i]['tags']]
if a.limit: order = order[:a.limit]
fo = gzip.open(f'{HERE}/out/{a.out_name}.jsonl.gz', 'wt'); b0 = 0; ndone = 0; tg = time.time()
while b0 < len(order):
    if time.time() - tg > a.gen_secs: log('time budget hit; stopping at', b0, 'of', len(order)); break
    b1 = b0; pri = JOBS[order[b0]]['pri']
    while b1 < len(order) and JOBS[order[b1]]['pri'] == pri:
        j = JOBS[order[b1]]; T = Lp + len(C.q_ids(tok, j['q'])) + len(j['prefix']) + a.max_new
        if ((b1 - b0 + 1) * a.K * T > a.tok_budget or (b1 - b0 + 1) * a.K > a.max_rows) and b1 > b0: break
        b1 += 1
    bat = order[b0:b1]; b0 = b1
    seqs = [pre + C.q_ids(tok, JOBS[i]['q']) + JOBS[i]['prefix'] for i in bat]; Tm = max(map(len, seqs))
    ids = torch.full((len(seqs), Tm), PAD, dtype=torch.long); am = torch.zeros((len(seqs), Tm), dtype=torch.long)
    for r, s_ in enumerate(seqs): ids[r, Tm - len(s_):] = torch.tensor(s_); am[r, Tm - len(s_):] = 1
    st = Stop(Tm)
    with torch.no_grad():
        out = m.generate(input_ids=ids.to(dev), attention_mask=am.to(dev), do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                         min_p=None, repetition_penalty=1.0, max_new_tokens=a.max_new, num_return_sequences=a.K,
                         stopping_criteria=StoppingCriteriaList([st]), pad_token_id=PAD)
    for r, i in enumerate(bat):
        j = JOBS[i]; res = []
        for kk in range(a.K):
            new = out[r * a.K + kk, Tm:].tolist()
            txt = tok.decode(new); mm = END.search(txt); cut = txt[:mm.end()] if mm else txt
            g_ = re.search(C.NUM, cut); ans = g_.group(1).replace(',', '') if g_ else None
            res.append(dict(ans=ans, ok=int(C.correct(ans, j['gold'])), ntok=len(tok(cut, add_special_tokens=False)['input_ids'])))
        rec = {k_: v for k_, v in j.items() if k_ not in ('prefix', 'q')}
        rec.update(prefix_len=len(j['prefix']), cont=res, ex=tok.decode(new)[:300] if r < 2 else None)
        fo.write(json.dumps(rec) + '\n'); ndone += 1
    fo.flush()
    log('batch', len(bat), 'pri', pri, 'done', ndone, '/', len(order), 'Tm', Tm)
fo.close()
log('GATE DONE', ndone)
