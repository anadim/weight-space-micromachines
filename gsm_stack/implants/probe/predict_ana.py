"""Local analysis of the probe scores written by predict_fit.py (python3.11: numpy + sklearn; no states needed).

    python3.11 predict_ana.py --tag cvdev --train dev --test dev2         # -> out/ana_cvdev.json (+ printed tables)
    python3.11 predict_ana.py --tag trainC --train trainC --test dev,dev2 --scores   # also writes out/scores_{set}.jsonl.gz

Q1 value probe along the trajectory (line ends before '####'), Q3 '####' verifiers, Q4 value-gated early-restart simulation,
Q5 per-attempt scores for the voter lane. Within-item AUROC = pooled concordance over (right, wrong) attempt pairs of the
same item (mixed items only), at a matched position (same step index / same relative-position decile / same divergence slot).
"""
import os, json, gzip, argparse, collections, math
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='cvdev'); ap.add_argument('--train', default='dev'); ap.add_argument('--test', default='dev2')
ap.add_argument('--sims', type=int, default=300); ap.add_argument('--scores', action='store_true')
ap.add_argument('--skip_sim', action='store_true')
ap.add_argument('--hash_col', default='', help='force the per-attempt hash column of the scores file (default: best train OOF)')
a = ap.parse_args()
TEST = [s for s in a.test.split(',') if s]; SETS = [a.train] + TEST
H = os.path.dirname(os.path.abspath(__file__)); PD = f'{H}/out/pred_{a.tag}'; ATL = f'{H}/../atlas/data'
WAIT_LEN = 9                                  # "Wait, let me solve it again.\n" queue tokens (cost normalisation for k=0)
RES = {}


def auroc(y, s):
    y = np.asarray(y).astype(bool); s = np.asarray(s, float)
    if y.all() or (~y).all() or len(y) == 0: return float('nan')
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def wauc(y, s, key):
    """pooled within-group concordance over (y=1, y=0) pairs sharing `key` (item or (item, slot)). -> (auc, n_pairs)"""
    y = np.asarray(y).astype(np.int8); s = np.asarray(s, float); key = np.asarray(key)
    o = np.lexsort((s, key)); y, s, key = y[o], s[o], key[o]
    cuts = np.flatnonzero(key[1:] != key[:-1]) + 1
    num = den = 0.0
    for ys, ss in zip(np.split(y, cuts), np.split(s, cuts)):
        n1 = ys.sum(); n0 = len(ys) - n1
        if not n1 or not n0: continue
        from scipy.stats import rankdata
        r = rankdata(ss); num += r[ys == 1].sum() - n1 * (n1 + 1) / 2; den += n1 * n0
    return (num / den if den else float('nan')), int(den)


def load_attempts(S):
    p = f'{H}/tables/attempts_{S}.jsonl.gz'
    rows = [json.loads(l) for l in gzip.open(p, 'rt')]
    by = collections.defaultdict(list)
    for i, r in enumerate(rows): by[(r['idx'], r['seed'])].append(i)
    st = np.zeros(len(rows), np.int64)
    for v in by.values():
        v.sort(key=lambda i: rows[i]['k']); prev = -1
        for i in v:
            st[i] = prev + 1
            if rows[i].get('reg_end') is not None: prev = rows[i]['reg_end']
    A = dict(idx=np.array([r['idx'] for r in rows]), seed=np.array([r['seed'] for r in rows]), k=np.array([r['k'] for r in rows]),
             ok=np.array([int(r['ok']) for r in rows]), lg=np.array([-30.0 if r.get('lg') is None else r['lg'] for r in rows]),
             valid=np.array([bool(r.get('reg_ok')) and r.get('complete', True) for r in rows]),
             bad=np.array([bool(r.get('reg_bad')) for r in rows]),
             val=np.array([np.nan if r.get('reg_val') is None else r['reg_val'] for r in rows]),
             phrase_len=np.array([int(r.get('phrase_len') or 0) for r in rows]),
             hash_off=np.array([-1 if r.get('hash_off') is None else r['hash_off'] for r in rows]),
             seg_len=np.array([len(r['seg_tokens']) for r in rows]), pre=np.array([r['seg_start'] for r in rows]) - st,
             gid=np.array([r['gid'] for r in rows]), nrep=np.array([r['nrep'] for r in rows]))
    A['pre'] = A['pre'] + np.where(A['k'] == 0, WAIT_LEN, 0)          # tokens before the final segment (+ wait phrase if k=0)
    A['cost'] = A['pre'] + A['seg_len']
    lp = [np.array(r['seg_logp'], np.float32) for r in rows]; wf = [np.array(r['seg_wf']) for r in rows]
    return rows, A, lp, wf


def mixed_items(A):
    m = collections.defaultdict(list)
    for i, o in zip(A['idx'], A['ok']): m[i].append(o)
    return {i for i, v in m.items() if 0 < np.mean(v) < 1}


DATA = {}
for S in SETS:
    rows, A, lp, wf = load_attempts(S)
    nl = dict(np.load(f'{PD}/nl_{S}.npz')); hs = dict(np.load(f'{PD}/hash_{S}.npz'))
    # derived per-row fields
    att, off = nl['att'].astype(int), nl['off'].astype(int)
    o = np.lexsort((off, att)); assert (o == np.arange(len(o))).all() or True
    step = np.zeros(len(att), int); last = -1; c = 0
    for r in np.argsort(att * 100000 + off, kind='stable'):
        if att[r] != last: c = 0; last = att[r]
        step[r] = c; c += 1
    ho = A['hash_off'][att]; pl = A['phrase_len'][att]
    den = np.where(ho > 0, ho, A['seg_len'][att]) - pl
    rel = (off - pl + 1) / np.maximum(den, 1)
    # text baselines at each line end: running mean / min of sampled-token logp, tokens so far, step index
    mlp = np.zeros(len(att)); mnp = np.zeros(len(att)); llp = np.zeros(len(att))
    for r in range(len(att)):
        i = att[r]; x = lp[i][pl[r]:off[r] + 1]; w = wf[i][pl[r]:off[r] + 1] == 0; x = x[w]
        mlp[r] = x.mean() if len(x) else 0.0; mnp[r] = x.min() if len(x) else 0.0
    nl.update(att=att, off=off, step=step, rel=rel, mlp=mlp, mnp=mnp, ntok=off - pl + 1, idx=A['idx'][att], y=A['ok'][att])
    DATA[S] = dict(rows=rows, A=A, nl=nl, hs=hs, mixed=mixed_items(A))
    print(S, 'attempts', len(rows), 'nl rows', len(att), 'hash rows', len(hs['att']), 'mixed items', len(DATA[S]['mixed']), flush=True)

# text-only combined baseline: logistic on [mean logp, min logp, log ntok, step] fit on train rows
from sklearn.linear_model import LogisticRegression
def tfeat(nl): return np.stack([nl['mlp'], nl['mnp'], np.log1p(nl['ntok']), nl['step']], 1)
def gfolds(g, k=5):
    ug = np.unique(g); rng = np.random.default_rng(0); rng.shuffle(ug); m = {x: i % k for i, x in enumerate(ug)}
    return np.array([m[x] for x in g])
tr = DATA[a.train]['nl']; Xt = tfeat(tr); f = gfolds(tr['idx']); oof = np.zeros(len(Xt))
for k in range(5):
    oof[f == k] = LogisticRegression(max_iter=1000).fit(Xt[f != k], tr['y'][f != k]).decision_function(Xt[f == k])
tr['text'] = oof; lr = LogisticRegression(max_iter=1000).fit(Xt, tr['y'])
for S in TEST: DATA[S]['nl']['text'] = lr.decision_function(tfeat(DATA[S]['nl']))
for S in SETS: DATA[S]['nl']['ntok_neg'] = -DATA[S]['nl']['ntok']

NLC = list(DATA[a.train]['nl']['cols'])
def col(S, c):
    nl = DATA[S]['nl']
    return nl['S'][:, NLC.index(c)] if c in NLC else nl[c]

# ---------------------------------------------------------------- Q1: value probe along the trajectory
def per_slot(S, slot, score, only_mixed=True):
    """slot: (n,) int bucket per row (-1 = unused); takes per (attempt, slot) the last row. -> all-AUROC, within-item AUROC, pairs"""
    nl = DATA[S]['nl']; keep = slot >= 0
    r = np.flatnonzero(keep); key = nl['att'][r] * 1000 + slot[r]
    o = np.lexsort((nl['off'][r], key)); r = r[o]; key = key[o]
    lastm = np.r_[key[1:] != key[:-1], True]; r = r[lastm]
    y = nl['y'][r]; s = score[r]; it = nl['idx'][r]
    auc_all = auroc(y, s)
    mm = np.isin(it, list(DATA[S]['mixed']))
    w, npair = wauc(y[mm], s[mm], it[mm] * 1000 + slot[r][mm])
    return auc_all, w, npair, len(r)


def decile_slot(S, d):
    """rows whose relative position <= (d+1)/10 (the latest such line end per attempt is used)"""
    nl = DATA[S]['nl']; return np.where(nl['rel'] <= (d + 1) / 10 + 1e-9, 0, -1)


q1 = {}
for S in SETS:
    nl = DATA[S]['nl']; q1[S] = {}
    # (0) per-layer summary: within-item matched by step (steps 0..11 pooled) and at 30% of the attempt
    stepslot = np.where(nl['step'] < 12, nl['step'], -1)
    for c in NLC + ['mlp', 'mnp', 'ntok_neg', 'text']:
        sc = col(S, c)
        au_all = auroc(nl['y'], sc)
        _, w_step, _, _ = per_slot(S, stepslot, sc)
        a30, w30, _, _ = per_slot(S, decile_slot(S, 2), sc)
        a60, w60, _, _ = per_slot(S, decile_slot(S, 5), sc)
        q1[S][c] = dict(auc_rows=au_all, w_step=w_step, auc30=a30, w30=w30, auc60=a60, w60=w60)
RES['q1_layers'] = q1
# pick the best value column on the TRAIN set by within-item AUROC at 30%
best_v = max([c for c in NLC], key=lambda c: q1[a.train][c]['w30'])
RES['best_value_col'] = best_v
SHOW = sorted({best_v, 'c4', '18', '22', '27'} & set(NLC), key=NLC.index) + ['mlp', 'mnp', 'text']
by_step = {}; by_dec = {}
for S in SETS:
    nl = DATA[S]['nl']; by_step[S] = {}; by_dec[S] = {}
    for s in list(range(0, 10)) + [10]:
        slot = np.where((nl['step'] == s) if s < 10 else (nl['step'] >= 10), 0, -1)
        by_step[S][s] = {c: per_slot(S, slot, col(S, c)) for c in SHOW}
    for d in range(10):
        by_dec[S][d] = {c: per_slot(S, decile_slot(S, d), col(S, c)) for c in SHOW}
RES['q1_by_step'] = by_step; RES['q1_by_decile'] = by_dec

# (c) relative to the semantic divergent step (if the divergence lane wrote it)
def load_div(S):
    p = f'{ATL}/diverge_{S}.jsonl.gz'
    if not os.path.exists(p): return None
    return [json.loads(l) for l in gzip.open(p, 'rt')]
RES['q1_divergence'] = {}
for S in SETS:
    dv = load_div(S)
    if dv is None: continue
    D = DATA[S]; A = D['A']; nl = D['nl']; gi = {g: i for i, g in enumerate(A['gid'])}
    at_step = {}
    for r in range(len(nl['att'])): at_step[(int(nl['att'][r]), int(nl['step'][r]))] = r
    nsteps = collections.Counter(int(x) for x in nl['att'])
    right_of = collections.defaultdict(list)
    for i in range(len(A['idx'])):
        if A['ok'][i]: right_of[A['idx'][i]].append(i)
    W = [(gi[d['gid']], d) for d in dv if d['rec'] == 'att' and not d['ok'] and d.get('div_line_tok') is not None and d['gid'] in gi]
    out = dict(n_wrong_with_div=len(W), div_rel_median=float(np.median([d['div_rel'] for _, d in W])),
               div_rel_le30=float(np.mean([d['div_rel'] <= 0.3 for _, d in W])),
               branch_vs_div=dict(collections.Counter(d.get('b_vs_div', 'na') for _, d in W)),
               cats=dict(collections.Counter(d.get('cat') for _, d in W)), rel={})
    sb = {}
    for i, d in W:
        sb[i] = sum(1 for st in range(nsteps[i]) if nl['off'][at_step[(i, st)]] < d['div_line_tok']) - 1
    for rs in range(-3, 4):
        res = {}
        for c in SHOW:
            sc = col(S, c); num = den = 0.0
            for i, d in W:
                t = sb[i] + rs
                if t < 0 or (i, t) not in at_step: continue
                sw = sc[at_step[(i, t)]]
                rr = [sc[at_step[(j, t)]] for j in right_of[A['idx'][i]] if (j, t) in at_step]
                if not rr: continue
                rr = np.array(rr); num += (rr > sw).sum() + 0.5 * (rr == sw).sum(); den += len(rr)
            res[c] = (num / den if den else float('nan'), int(den))
        out['rel'][rs] = res
    RES['q1_divergence'][S] = out
    print(S, 'divergence-relative within-item AUROC (rs=0: last line end before the divergent line; 1: its end)', flush=True)
    for rs, v in out['rel'].items(): print('  ', rs, {c: round(x[0], 3) for c, x in v.items()}, list(v.values())[0][1])

# ---------------------------------------------------------------- Q3: '####' verifiers
q3 = {}
HC = list(DATA[a.train]['hs']['cols'])
for S in SETS:
    hs = DATA[S]['hs']; A = DATA[S]['A']; att = hs['att'].astype(int); y = A['ok'][att]; it = A['idx'][att]
    mm = np.isin(it, list(DATA[S]['mixed'])); q3[S] = {}
    for c in HC + ['lg']:
        s = A['lg'][att] if c == 'lg' else hs['S'][:, HC.index(c)]
        q3[S][c] = dict(auc=auroc(y, s), w=wauc(y[mm], s[mm], it[mm])[0])
    q3[S]['_n'] = dict(n=len(att), mixed_rows=int(mm.sum()))
RES['q3'] = q3
multi = [c for c in HC if not c.startswith('L')]
best_h = max(multi, key=lambda c: q3[a.train][c]['auc'])
if a.hash_col: best_h = a.hash_col
RES['best_hash_col'] = best_h

# ---------------------------------------------------------------- Q4: value-gated early restart (offline simulation)
def sig(x): return 1 / (1 + np.exp(-x))


def build_pool(S, vcol, smax, wcol=None):
    """per item: list of attempts (cost_full, gate rows [(cost_at, score)], vote fields)"""
    D = DATA[S]; A = D['A']; nl = D['nl']; sc = col(S, vcol)
    rows_of = collections.defaultdict(list)
    sel = np.flatnonzero(nl['step'] < smax)
    for r in sel: rows_of[nl['att'][r]].append(r)
    W = sig(A['lg']) if wcol is None else wcol
    items = collections.defaultdict(list)
    for i in range(len(A['idx'])):
        rr = sorted(rows_of.get(i, []), key=lambda r: nl['off'][r])
        gate = np.array([[A['pre'][i] + nl['off'][r] + 1, sc[r]] for r in rr]).reshape(-1, 2)
        items[A['idx'][i]].append((A['cost'][i], gate, A['valid'][i], A['bad'][i], A['val'][i], W[i], A['ok'][i]))
    return items


def vote(slots):
    keep = [s for s in slots if s[2]]
    sane = [s for s in keep if not s[3]]
    if sane: keep = sane
    if not keep: return 0
    tot = collections.defaultdict(float); first = {}
    for j, s in enumerate(keep):
        v = round(float(s[4]), 6); tot[v] += s[5]; first.setdefault(v, j)
    v = max(tot, key=lambda v: (tot[v], -first[v]))
    return int(next(s[6] for s in keep if round(float(s[4]), 6) == v))


def stream(pool, B, thr, rng):
    """fill budget B with i.i.d. attempts; an attempt whose gate score drops below thr at a line end is abandoned there
    (cost up to that line end, no vote); an attempt that would overrun B is cut (no vote)."""
    used = 0; slots = []; ab = 0; P = len(pool)
    while len(slots) < 64:
        at = pool[rng.integers(0, P)]; g = at[1]
        hit = np.flatnonzero(g[:, 1] < thr) if thr is not None and len(g) else []
        c = g[hit[0], 0] if len(hit) else at[0]
        if used + c > B: break
        used += c
        if len(hit): ab += 1
        else: slots.append(at)
    return slots, used, ab


def simulate(items, thr, sims, seed=0, n=8):
    """plain = 8 i.i.d. attempts (budget B = their tokens); ctrl = ungated stream filling B (same cut rule as gated);
    gated = value-gated stream filling B."""
    rng = np.random.default_rng(seed); acc = collections.Counter(); cnt = 0
    for it, pool in items.items():
        P = len(pool)
        for _ in range(sims):
            d = rng.integers(0, P, n); B = sum(pool[j][0] for j in d)
            acc['plain'] += vote([pool[j] for j in d]); acc['tok_p'] += B
            sl, u, _ = stream(pool, B, None, rng); acc['ctrl'] += vote(sl); acc['votes_ctrl'] += len(sl)
            sl, u, ab = stream(pool, B, thr, rng); acc['gated'] += vote(sl); acc['votes'] += len(sl); acc['aborts'] += ab
            acc['tok_g'] += u; cnt += 1
    r = {k: v / cnt for k, v in acc.items() if not k.startswith('tok')}; r['tok_ratio'] = acc['tok_g'] / acc['tok_p']
    return r


if not a.skip_sim:
    q4 = dict(select={}, test={})
    cands = sorted({best_v, 'c4', '22'} & set(NLC), key=NLC.index)
    trs = DATA[a.train]['nl']
    best = None
    for vc in cands:
        for smax in (2, 3, 5, 99):
            items = build_pool(a.train, vc, smax)
            gs = col(a.train, vc)[trs['step'] < smax]
            for qf in (0.01, 0.03, 0.06, 0.1, 0.2):
                thr = float(np.quantile(gs, qf))
                r = simulate(items, thr, sims=40, seed=1)
                q4['select'][f'{vc}|s<{smax}|q{qf}'] = dict(thr=thr, **r)
                print('sel', vc, smax, qf, r, flush=True)
                if best is None or r['gated'] - r['ctrl'] > best[0]: best = (r['gated'] - r['ctrl'], vc, smax, thr, qf)
    _, vc, smax, thr, qf = best; q4['chosen'] = dict(col=vc, smax=smax, thr=thr, q=qf)
    for S in SETS:
        items = build_pool(S, vc, smax)
        q4['test'][S] = dict(chosen=simulate(items, thr, sims=a.sims, seed=2))
        for smax2 in (3, 99):                      # fixed reference configs (thr = train quantile 5%)
            gs = col(a.train, vc)[trs['step'] < smax2]; t2 = float(np.quantile(gs, 0.05))
            q4['test'][S][f's<{smax2}|q0.05'] = simulate(build_pool(S, vc, smax2), t2, sims=a.sims // 2, seed=3)
        print('test', S, q4['test'][S], flush=True)
    RES['q4'] = q4

# ---------------------------------------------------------------- Q5: per-attempt scores for the voter lane
if a.scores:
    os.makedirs(f'{H}/out', exist_ok=True)
    for S in SETS:
        D = DATA[S]; A = D['A']; nl = D['nl']; hs = D['hs']
        hcol = {int(t): r for r, t in enumerate(hs['att'])}
        vs = col(S, best_v); v4 = col(S, 'c4')
        per = collections.defaultdict(list)
        for r in np.argsort(nl['off'], kind='stable'): per[int(nl['att'][r])].append(r)
        fitsrc = 'oof5' if S == a.train else f'fit_{a.train}'
        with gzip.open(f'{H}/out/scores_{S}.jsonl.gz', 'wt') as f:
            for i in range(len(A['idx'])):
                rr = sorted(per.get(i, []), key=lambda r: nl['off'][r]); h = hcol.get(i)
                e = dict(gid=str(A['gid'][i]), fit=f'{a.tag}:{fitsrc}', hash_col=best_h,
                         hash=None if h is None else round(float(hs['S'][h, HC.index(best_h)]), 4),
                         hash_stack_lg=None if h is None or 'stack_lg' not in HC else round(float(hs['S'][h, HC.index('stack_lg')]), 4),
                         val_col=best_v, val_mean=round(float(vs[rr].mean()), 4) if rr else None,
                         val_min=round(float(vs[rr].min()), 4) if rr else None,
                         val_c4_mean=round(float(v4[rr].mean()), 4) if rr else None,
                         val_steps=[round(float(vs[r]), 3) for r in rr[:5]],
                         val_at30=next((round(float(vs[r]), 4) for r in rr[::-1] if nl['rel'][r] <= 0.3), None))
                f.write(json.dumps(e) + '\n')
        print('wrote', f'{H}/out/scores_{S}.jsonl.gz', flush=True)

json.dump(RES, open(f'{H}/out/ana_{a.tag}.json', 'w'), indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x))
print('best value col', best_v, 'best hash col', best_h)
for S in SETS:
    print(S, 'Q1 per layer (auc_rows / w_step / auc30 / w30):')
    for c, v in q1[S].items(): print(f'  {c:9s} {v["auc_rows"]:.3f} {v["w_step"]:.3f} {v["auc30"]:.3f} {v["w30"]:.3f} {v["w60"]:.3f}')
    print(S, 'Q3:', {c: (round(v['auc'], 3), round(v['w'], 3)) for c, v in q3[S].items() if c != '_n'})
