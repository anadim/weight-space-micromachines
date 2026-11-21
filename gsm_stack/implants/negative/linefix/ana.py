"""Gate analysis (local): per-attempt accuracy change from a 1-rewind line repair at the first detector flag, and a rough
plurality-vote simulation. Inputs: gate/out/{gen.jsonl.gz, scores_*.npz}, gate/tables/attempts_*.jsonl.gz,
../../probe/tables/divlab_*.jsonl.gz (D lines).   python3 ana.py > gate_results.md
"""
import gzip, json, collections, sys
import numpy as np
G = 'gate_full_out'; FPRS = ('0.01', '0.02', '0.05')
THR = {'0.01': 2.238588333129883, '0.02': 1.8208669424057007, '0.05': 1.1829034090042114}   # trainC OOF line FPR (gate log)
import os
gen = [json.loads(l) for l in gzip.open(f'{G}/gen.jsonl.gz', 'rt')]
CT = [json.loads(l) for l in gzip.open('gate/out/ctrl.jsonl.gz', 'rt')] if os.path.exists('gate/out/ctrl.jsonl.gz') else []
bykey = {(r['set'], r['gid'], r['prev_off']): r for r in gen}
rng = np.random.default_rng(0)


def num(x):
    try: return float(x)
    except (TypeError, ValueError): return None


def vote(ans):
    c = collections.Counter(a for a in ans if a is not None)
    if not c: return None
    best = max(c.values())
    for a in ans:
        if a is not None and c[a] == best: return a


print('# linefix gate results\n'); FRESH = []
for S in ('dev', 'dev2'):
    sc = np.load(f'{G}/scores_{S}.npz'); gids = sc['gids']
    T = {}
    for l in gzip.open(f'gate/tables/attempts_{S}.jsonl.gz', 'rt'):
        r = json.loads(l); T[r['gid']] = dict(idx=r['idx'], seed=r['seed'], k=r['k'], ok=int(r['ok']), ans=r['ans'], gold=r['gold'],
                                              steps=r['steps'])
    dl = {}
    for l in gzip.open(f'../../probe/tables/divlab_{S}.jsonl.gz', 'rt'):
        d = json.loads(l); dl[d['gid']] = (d['dl'], d['dt'], d['cat'])
    N = len(T); natt_ok = np.mean([t['ok'] for t in T.values()])
    # per attempt: line list (step, off, prev_off, score)
    per = collections.defaultdict(list)
    for a_, st, of, po, s in zip(sc['att'], sc['step'], sc['off'], sc['prev_off'], sc['score']): per[gids[a_]].append((st, of, po, s))
    ctrl = [r for r in gen + CT if r['set'] == S and 'ctrl' in r['tags']]
    PR = {}
    for l in gzip.open(f'../../probe/tables/items_{S}.jsonl.gz', 'rt'):
        it = json.loads(l); PR[it['idx']] = it['n_right'] / 32
    print(f'## {S}: {N} attempts, per-attempt acc {100 * natt_ok:.2f}\n')
    print('| thr (train line FPR) | flagged att | gen coverage | flagged wrong (n, P_fix) | ... at D line | before D | after D / no D | flagged right (n, P_keep) | per-att delta (pts, covered) | delta scaled to full coverage | delta, breaks in excess of ctrl only (scaled) |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    sims = {}
    for fp in FPRS:
        fl = {}; flagged = set()
        for g, L in per.items():
            hit = [x for x in sorted(L) if x[3] > THR[fp]]
            if not hit: continue
            flagged.add(g); r = bykey.get((S, g, int(hit[0][2])))
            if r is not None and f'f{fp}' in r['tags']: fl[g] = hit[0]
        W, R = [], []; wpos = collections.defaultdict(list)
        for g, x in fl.items():
            r = bykey[(S, g, int(x[2]))]; p = np.mean([c['ok'] for c in r['cont']]); t = T[g]
            if t['ok']: R.append(p)
            else:
                W.append(p)
                if g in dl:
                    d0, d1, _ = dl[g]; lo, hi = int(x[2]) + 1, int(x[1])     # flagged line tokens (lo..hi]
                    pos = 'D' if lo <= d1 <= hi else ('before' if hi < d1 else 'after')
                else: pos = 'after'
                wpos[pos].append(p)
        n = len(fl)
        delta = (sum(W) - (len(R) - sum(R))) / N
        cR = [np.mean([c['ok'] for c in r['cont']]) for r in ctrl if r['ok']]
        keep_ctrl = np.mean(cR) if cR else float('nan')
        adj = (sum(W) - sum(max(0.0, (1 - p) - (1 - keep_ctrl)) for p in R)) / N if cR else float('nan')
        zb = sum(W) / N
        pw = np.mean([PR[T[g]['idx']] for g in fl if not T[g]['ok']]); pr_ = np.mean([PR[T[g]['idx']] for g in fl if T[g]['ok']])
        FRESH.append(f'| {S} | {fp} | {pw:.3f} | {np.mean(W):.3f} | {pr_:.3f} | {np.mean(R):.3f} | {100 * zb * len(flagged) / max(n, 1):+.2f} |')
        f = lambda L: f'{len(L)}, {np.mean(L):.3f}' if L else '0, -'
        print(f'| {fp} | {n} ({100 * n / N:.1f}%) | {n}/{len(flagged)} | {f(W)} | {f(wpos["D"])} | {f(wpos["before"])} | {f(wpos["after"])} | {f(R)} | {100 * delta:+.2f} | {100 * delta * len(flagged) / max(n, 1):+.2f} | {100 * adj * len(flagged) / max(n, 1):+.2f} |')
        sims[fp] = fl
    cW = [np.mean([c['ok'] for c in r['cont']]) for r in ctrl if not r['ok']]
    cR = [np.mean([c['ok'] for c in r['cont']]) for r in ctrl if r['ok']]
    print(f'\ncontrol (random line of unflagged attempts at 5% thr): right n={len(cR)} P_keep={np.mean(cR):.3f}; wrong n={len(cW)} P_fix={np.mean(cW) if cW else float("nan"):.3f}\n')
    # vote simulation: plurality over the 8 attempts of (idx, seed); flagged attempts' answers replaced by a random continuation
    grp = collections.defaultdict(dict)
    for g, t in T.items(): grp[(t['idx'], t['seed'])][t['k']] = g
    print('| thr | plurality vote base | after rewind (mean of 50 draws) | delta (pts) |')
    print('|---|---|---|---|')
    for fp, fl in sims.items():
        base, new, pab, pan = [], [], [], []
        for d_ in range(50):
            nv = []
            for (i, s), ks in grp.items():
                gs = [ks[k] for k in sorted(ks)]; gold = T[gs[0]]['gold']
                A0 = [T[g]['ans'] for g in gs]; A1 = []
                for g in gs:
                    if g in fl:
                        c = bykey[(S, g, int(fl[g][2]))]['cont']; A1.append(c[rng.integers(len(c))]['ans'])
                    else: A1.append(T[g]['ans'])
                if d_ == 0:
                    v0 = vote(A0); base.append(v0 is not None and abs(float(v0) - float(gold)) <= 1e-6 * max(1, abs(float(gold))))
                v1 = vote(A1); nv.append(v1 is not None and num(v1) is not None and abs(float(v1) - float(gold)) <= 1e-6 * max(1, abs(float(gold))))
            new.append(np.mean(nv))
        print(f'| {fp} | {100 * np.mean(base):.2f} | {100 * np.mean(new):.2f} | {100 * (np.mean(new) - np.mean(base)):+.2f} |')
    print()
print('Fresh-attempt baseline: item p_right (n_right/32) of the flagged attempts vs rewind outcome; zero-break bound = gain if no flagged right attempt broke (scaled to full coverage)\n')
print('| set | thr | flagged wrong: item p_right | P_fix | flagged right: item p_right | P_keep | zero-break bound (pts) |')
print('|---|---|---|---|---|---|---|')
print('\n'.join(FRESH))
