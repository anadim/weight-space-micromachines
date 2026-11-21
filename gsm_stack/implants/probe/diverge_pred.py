"""Predictability AT the first divergent step D (lane 2, key question). Local (python3.11, numpy/scipy), no states needed.

    python3.11 diverge_pred.py --tag trainC --sets dev,dev2      # -> out/divpred_{tag}.json + printed tables

Inputs: out/pred_{tag}/nl_{set}.npz (line-end value-probe logits; att = row of tables/attempts_{set}), atlas diverge_{set}
(D per wrong attempt of a mixed item: div_line_tok = first token of the D line, div_tok = token of the divergent value;
token branch point vs closest right attempt with lp_w / lp_r), tables/attempts_{set} (seg_logp, seg_wf).
(a) 'knows after': step-wise logit change across the D line (end of D line minus end of the line before it) vs all other
    line-to-line changes (right attempts of mixed items; wrong attempts before D / after D). AUROC of -delta.
(b) 'knows before': within-item AUROC at matched step index t (right attempts of the same item having a line end at t)
    for t = step of the line end just before the D line (rs=0), and rs = -2..+3 around it (+1 = end of the D line).
(c) text: branch-token probabilities and a min-p proxy (sampled p < thr) on wrong vs right attempts.
"""
import os, json, gzip, argparse, collections
import numpy as np
from scipy.stats import rankdata

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='trainC'); ap.add_argument('--sets', default='dev,dev2')
ap.add_argument('--cols', default='18,22,27,c4,w22,wc4,we22')
a = ap.parse_args()
H = os.path.dirname(os.path.abspath(__file__)); PD = f'{H}/out/pred_{a.tag}'; ATL = f'{H}/../atlas/data'
SETS = a.sets.split(','); RES = {}


def auroc(y, s):
    y = np.asarray(y).astype(bool); s = np.asarray(s, float)
    if len(y) == 0 or y.all() or (~y).all(): return float('nan')
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def rd(p): return [json.loads(l) for l in gzip.open(p, 'rt')]


for S in SETS:
    rows = rd(f'{H}/tables/attempts_{S}.jsonl.gz'); gi = {r['gid']: i for i, r in enumerate(rows)}
    nl = dict(np.load(f'{PD}/nl_{S}.npz')); cols = [str(c) for c in nl['cols']]
    C = [c for c in a.cols.split(',') if c in cols]
    att = nl['att'].astype(int); off = nl['off'].astype(int)
    per = collections.defaultdict(list)
    for r in np.lexsort((off, att)): per[int(att[r])].append(int(r))
    sc = {c: nl['S'][:, cols.index(c)] for c in C}
    ok = np.array([r['ok'] for r in rows]); idx = np.array([r['idx'] for r in rows])
    pr = collections.defaultdict(list)
    for i, o in zip(idx, ok): pr[i].append(o)
    mixed = {i for i, v in pr.items() if 0 < np.mean(v) < 1}
    right_of = collections.defaultdict(list)
    for i in range(len(rows)):
        if ok[i] and idx[i] in mixed: right_of[idx[i]].append(i)
    dv = [d for d in rd(f'{ATL}/diverge_{S}.jsonl.gz') if d['rec'] == 'att' and not d['ok'] and d['gid'] in gi]
    out = dict(n_wrong_mixed=len(dv))

    # ---------------- (a)/(b) hidden state around D
    W = []; why = collections.Counter()
    for d in dv:
        if d.get('div_kind') in (None, 'none'): why['no_D'] += 1; continue
        i = gi[d['gid']]; rr = per.get(i, [])
        if not rr: why['no_lines'] += 1; continue
        offs = off[rr]
        e1 = np.flatnonzero(offs >= d['div_tok']); e0 = np.flatnonzero(offs < d['div_line_tok'])
        if not len(e1): why['D_in_answer_line'] += 1; continue
        t1 = int(e1[0]); t0 = int(e0[-1]) if len(e0) else -1
        if t0 < 0: why['D_in_first_line'] += 1
        W.append((i, d, t0, t1))
    out['coverage'] = dict(usable=len(W), with_pre_line=sum(t0 >= 0 for _, _, t0, _ in W), **why)
    Dset = {i: (t0, t1) for i, _, t0, t1 in W}

    # (a) delta across the D line vs other line-to-line deltas
    resA = {}
    for c in C:
        s = sc[c]; dD, dR, dWpre, dWpost = [], [], [], []
        for i in list(Dset) + [j for v in right_of.values() for j in v]:
            rr = per.get(i, []); v = s[rr]
            if i in Dset:
                t0, t1 = Dset[i]
                if t0 >= 0: dD.append(v[t1] - v[t0])
                for t in range(1, len(v)):
                    if t <= t0: dWpre.append(v[t] - v[t - 1])
                    elif t > t1: dWpost.append(v[t] - v[t - 1])
            else:
                dR += list(np.diff(v))
        dD, dR, dWpre, dWpost = map(np.array, (dD, dR, dWpre, dWpost))
        neg = np.r_[dR, dWpre]
        resA[c] = dict(n_D=len(dD), mean_dD=float(dD.mean()), mean_dR=float(dR.mean()), mean_dWpre=float(dWpre.mean()),
                       mean_dWpost=float(dWpost.mean()), frac_drop_D=float((dD < 0).mean()), frac_drop_R=float((dR < 0).mean()),
                       auc_D_vs_rightAndPre=auroc(np.r_[np.ones(len(dD)), np.zeros(len(neg))], -np.r_[dD, neg]),
                       auc_D_vs_right=auroc(np.r_[np.ones(len(dD)), np.zeros(len(dR))], -np.r_[dD, dR]),
                       auc_D_vs_wpre=auroc(np.r_[np.ones(len(dD)), np.zeros(len(dWpre))], -np.r_[dD, dWpre]),
                       sd_dR=float(dR.std()))
    out['a_delta'] = resA

    # (b) within-item matched-step AUROC around D (rs=0: line end before D line, +1: end of the D line)
    resB = {}
    for c in C:
        s = sc[c]; resB[c] = {}
        for rs in range(-2, 4):
            num = den = 0.0; nW = 0
            for i, d, t0, t1 in W:
                t = (t0 + rs) if rs <= 0 else (t1 + rs - 1)
                if t < 0 or t >= len(per[i]) or (rs <= 0 and t0 < 0): continue
                sw = s[per[i][t]]
                rv = np.array([s[per[j][t]] for j in right_of[idx[i]] if len(per.get(j, [])) > t])
                if not len(rv): continue
                num += (rv > sw).sum() + 0.5 * (rv == sw).sum(); den += len(rv); nW += 1
            resB[c][rs] = dict(auc=num / den if den else float('nan'), n_wrong=nW, pairs=int(den))
    out['b_within'] = resB
    # (b') per category at rs=0 / +1 for the first column
    c0 = C[0] if 'c4' not in C else 'c4'; s = sc[c0]; bycat = {}
    for cat in sorted({d['cat'] for _, d, _, _ in W}):
        r_ = {}
        for rs in (0, 1):
            num = den = 0.0
            for i, d, t0, t1 in W:
                if d['cat'] != cat: continue
                t = t0 if rs == 0 else t1
                if t < 0: continue
                sw = s[per[i][t]]
                rv = np.array([s[per[j][t]] for j in right_of[idx[i]] if len(per.get(j, [])) > t])
                num += (rv > sw).sum() + 0.5 * (rv == sw).sum(); den += len(rv)
            r_[rs] = round(num / den, 3) if den else None
        bycat[cat] = dict(n=sum(d['cat'] == cat for _, d, _, _ in W), **{f'rs{k}': v for k, v in r_.items()})
    out['b_by_cat_' + c0] = bycat

    # ---------------- (c) text: branch token probabilities + min-p proxy
    br = [d for d in dv if d.get('lp_w') is not None]
    pw = np.exp([d['lp_w'] for d in br]); prr = np.exp([d['lp_r'] for d in br])
    crit = np.array([d.get('b_vs_div') == 'in_line' for d in br])
    resC = dict(n_branch=len(br), n_crit=int(crit.sum()))
    TH = (0.01, 0.05, 0.1, 0.2)
    for th in TH:
        resC[f'thr{th}'] = dict(pw_lt=float((pw < th).mean()), pr_lt=float((prr < th).mean()),
                                pw_lt_pr_ge=float(((pw < th) & (prr >= th)).mean()),
                                crit_pw_lt=float((pw[crit] < th).mean()) if crit.any() else None,
                                crit_pw_lt_pr_ge=float(((pw < th) & (prr >= th))[crit].mean()) if crit.any() else None)
    # any low-prob sampled token in an attempt: wrong (mixed) / right (mixed) / right (all items); before D for wrong
    dmap = {gi[d['gid']]: d for d in dv}
    anyl = {k: collections.Counter() for k in ('wrong_mixed', 'wrong_mixed_preD', 'right_mixed', 'right_all', 'all')}
    nlow = {k: collections.defaultdict(list) for k in anyl}; n = collections.Counter()
    for i, r in enumerate(rows):
        lp = np.array(json.loads(r['seg_logp']) if isinstance(r['seg_logp'], str) else r['seg_logp'], np.float32)
        wf = np.array(json.loads(r['seg_wf']) if isinstance(r['seg_wf'], str) else r['seg_wf'])
        pl = int(r.get('phrase_len') or 0); ho = r.get('hash_off')
        end = len(lp) if ho is None else ho
        m = np.zeros(len(lp), bool); m[pl:end] = True; m &= (wf == 0)
        p = np.exp(lp[m]); pos = np.flatnonzero(m)
        keys = ['all']
        if r['ok']: keys.append('right_all')
        if r['idx'] in mixed: keys.append('right_mixed' if r['ok'] else 'wrong_mixed')
        if i in dmap and dmap[i].get('div_kind') not in (None, 'none'):
            keys.append('wrong_mixed_preD'); pre = pos < dmap[i]['div_line_tok']
        for k in keys:
            n[k] += 1; pp = p[pre] if k == 'wrong_mixed_preD' else p
            for th in TH:
                c_ = int((pp < th).sum()); anyl[k][th] += c_ > 0; nlow[k][th].append(c_)
    resC['any_low'] = {k: {f'thr{th}': dict(frac_any=anyl[k][th] / max(1, n[k]), mean_count=float(np.mean(nlow[k][th])) if nlow[k][th] else None)
                           for th in TH} | dict(n=n[k]) for k in anyl}
    out['c_text'] = resC

    # ---------------- (d) text baseline for the step-level error detector: line logprob (D line vs other line ends)
    vt = {}
    for l in gzip.open(f'{H}/tables/valtok_{S}.jsonl.gz', 'rt'):
        v = json.loads(l)
        if v['gid'] in gi: vt[gi[v['gid']]] = np.array(v['eq'], int)
    LP = {}
    def line_feats(i):
        if i not in LP:
            r = rows[i]; lp = np.array(json.loads(r['seg_logp']) if isinstance(r['seg_logp'], str) else r['seg_logp'], np.float32)
            wf = np.array(json.loads(r['seg_wf']) if isinstance(r['seg_wf'], str) else r['seg_wf'])
            offs = off[per[i]]; lo = np.r_[int(r.get('phrase_len') or 0) - 1, offs[:-1]]; f = []
            eqs = vt.get(i, np.zeros(0, int))
            for l_, h_ in zip(lo, offs):
                m = np.zeros(len(lp), bool); m[l_ + 1:h_ + 1] = True; m &= wf == 0; x = lp[m]
                f.append((-x.min() if len(x) else 0.0, -x.mean() if len(x) else 0.0, bool(((eqs > l_) & (eqs <= h_)).any())))
            LP[i] = f
        return LP[i]
    resD = {}
    for eq_only in (False, True):
        pos, neg, wpairs = [], [], [0.0, 0.0, 0.0, 0.0]
        for i, d, t0, t1 in W:
            if t1 < 1: continue
            f = line_feats(i)
            if eq_only and not f[t1][2]: continue
            pos.append(f[t1][:2]); neg += [f[t][:2] for t in range(1, t0 + 1) if f[t][2] or not eq_only]
            for j in right_of[idx[i]]:
                g_ = line_feats(j)
                if len(g_) > t1 and (g_[t1][2] or not eq_only):
                    for k in (0, 1):
                        wpairs[2 * k] += (f[t1][k] > g_[t1][k]) + 0.5 * (f[t1][k] == g_[t1][k]); wpairs[2 * k + 1] += 1
        for j in [j for v in right_of.values() for j in v]:
            neg += [x[:2] for t, x in enumerate(line_feats(j)) if t >= 1 and (x[2] or not eq_only)]
        pos, neg = np.array(pos), np.array(neg); yy = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        resD['eq' if eq_only else 'all'] = dict(n_pos=len(pos), n_neg=len(neg), auc_neg_minlogp=auroc(yy, np.r_[pos[:, 0], neg[:, 0]]),
                                                 auc_neg_meanlogp=auroc(yy, np.r_[pos[:, 1], neg[:, 1]]),
                                                 w_minlogp=wpairs[0] / max(1, wpairs[1]), w_meanlogp=wpairs[2] / max(1, wpairs[3]))
    out['d_text_line'] = resD
    print('(d) text baseline D-line detector', json.dumps(resD))
    RES[S] = out

    # ---------------- print
    print(f'== {S} ({a.tag})  wrong attempts of mixed items {len(dv)}; coverage {out["coverage"]}')
    print('(a) delta across D line: col | n | mean dD | mean dR (right) | mean dWpre | mean dWpost | P(dD<0) | P(dR<0) | AUROC D vs right+pre | vs right | vs pre')
    for c, v in resA.items():
        print(f'  {c:5s} {v["n_D"]:5d} {v["mean_dD"]:+.3f} {v["mean_dR"]:+.3f} {v["mean_dWpre"]:+.3f} {v["mean_dWpost"]:+.3f} '
              f'{v["frac_drop_D"]:.2f} {v["frac_drop_R"]:.2f} {v["auc_D_vs_rightAndPre"]:.3f} {v["auc_D_vs_right"]:.3f} {v["auc_D_vs_wpre"]:.3f}')
    print('(b) within-item AUROC (right > wrong at same step) by rs (0 = line end before D line, 1 = end of D line):')
    for c, v in resB.items():
        print(f'  {c:5s} ' + ' '.join(f'{rs:+d}:{x["auc"]:.3f}({x["n_wrong"]})' for rs, x in v.items()))
    print('(b) by category', c0, json.dumps(bycat))
    print('(c) branch tokens', json.dumps({k: v for k, v in resC.items() if k != 'any_low'}))
    for k, v in resC['any_low'].items():
        print('   any-low', k, v['n'], ' '.join(f'{t}:{v[t]["frac_any"]:.3f}/{v[t]["mean_count"]:.2f}' for t in v if t != 'n'))

json.dump(RES, open(f'{H}/out/divpred_{a.tag}.json', 'w'), indent=1, default=float)
