"""Render notes/atlas.md tables from data/diverge_{set}.jsonl.gz (+ hand labels, attempts for baselines).
usage: python3 atlas/report.py > notes/atlas_tables.md   (atlas.md = hand-written text + these tables)"""
import os, sys, json, gzip, math, collections, statistics as S
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from diverge import rd, CATS, BANDS, band          # noqa: E402
from calib import load as load_calib, CATS as _C   # noqa: E402

SETS = ('dev', 'dev2'); SEEDS = ('7', '11', '13', '17')
BN = [f'{a}-{b}' for a, b in BANDS]
pc = lambda a, b: f'{100 * a / b:.0f}%' if b else '-'


def q(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float('nan')


def load(s):
    R = rd(f'{HERE}/data/diverge_{s}.jsonl.gz')
    return [r for r in R if r['rec'] == 'att'], [r for r in R if r['rec'] == 'item']


def cat_tables(D):
    print('### Heuristic categories (all wrong attempts of mixed items; item = dominant category of its wrong attempts)\n')
    print('| category | ' + ' | '.join(f'{s} att | {s} item' for s in SETS) + ' |')
    print('|---|' + '---|' * (2 * len(SETS)))
    for c in CATS:
        row = []
        for s in SETS:
            A, I = D[s]; W = [r for r in A if not r['ok']]
            a = sum(r['cat'] == c for r in W); i = sum(r['dominant_cat'] == c for r in I)
            row += [f'{a} ({pc(a, len(W))})', f'{i} ({pc(i, len(I))})']
        print(f'| {c} | ' + ' | '.join(row) + ' |')
    print('\nBy n_right band (dev+dev2, attempt-level %, n = wrong attempts):\n')
    print('| category | ' + ' | '.join(BN) + ' |'); print('|---|' + '---|' * len(BN))
    W = [r for s in SETS for r in D[s][0] if not r['ok']]
    nr = {(r['set'], r['idx']): r['n_right'] for s in SETS for r in D[s][1]}
    byb = {b: [r for r in W if band(nr[(r['set'], r['idx'])]) == b] for b in BN}
    for c in CATS:
        print(f'| {c} | ' + ' | '.join(pc(sum(r['cat'] == c for r in byb[b]), len(byb[b])) for b in BN) + ' |')
    print('| n | ' + ' | '.join(str(len(byb[b])) for b in BN) + ' |\n')


def hand_tables(D):
    S_, H, R = load_calib()
    nmix = collections.Counter((s, it['band']) for s in SETS for it in D[s][1])
    ns = collections.Counter((k[0], S_[k]['band']) for k in H)
    print('### Hand labels (stratified sample: 120 mixed items, 20 dev + 10 dev2 per band; the attractor-answer attempt)\n')
    print('| category | dev n | dev2 n | ' + ' | '.join(BN) + ' | weighted % (all mixed) |')
    print('|---|---|---|' + '---|' * len(BN) + '---|')
    tot = sum(nmix.values())
    for c in CATS:
        ks = [k for k in H if H[k][0] == c]
        if not ks: continue
        w = sum(nmix[(k[0], S_[k]['band'])] / ns[(k[0], S_[k]['band'])] for k in ks)
        bc = [sum(S_[k]['band'] == b for k in ks) for b in BN]
        print(f"| {c} | {sum(k[0] == 'dev' for k in ks)} | {sum(k[0] == 'dev2' for k in ks)} | " +
              ' | '.join(f'{x}/30' for x in bc) + f' | {100 * w / tot:.0f}% |')
    P = [(H[k][0], R[S_[k]['wrong']]['cat']) for k in H]
    print(f"\nHeuristic vs hand: exact agreement {sum(a == b for a, b in P)}/120; per hand category (recall):",
          ', '.join(f"{c} {sum(a == b == c for a, b in P)}/{sum(a == c for a, b in P)}" for c in CATS if any(a == c for a, b in P)))
    print()


def position_tables(D):
    print('### Where the first divergent step sits\n')
    print('| set | wrong att with D | D = 1st step | D rel. position median [IQR] | D step index median | no D (all values right) |')
    print('|---|---|---|---|---|---|')
    for s in SETS:
        W = [r for r in D[s][0] if not r['ok'] and r['ans'] is not None]
        d = [r for r in W if r.get('div_kind') not in (None, 'none')]
        rel = [r['div_rel'] for r in d]; si = [r['div_si'] for r in d]
        print(f"| {s} | {len(d)}/{len(W)} | {pc(sum(x == 0 for x in si), len(d))} | {S.median(rel):.2f} [{q(rel, .25):.2f}-{q(rel, .75):.2f}] "
              f"| {S.median(si):.0f} | {sum(r.get('div_kind') == 'none' for r in W)} |")
    W = [r for s in SETS for r in D[s][0] if not r['ok'] and r.get('div_kind') not in (None, 'none')]
    print('\nMedian relative position of D by category (dev+dev2): ' + ', '.join(
        f"{c} {S.median([r['div_rel'] for r in W if r['cat'] == c]):.2f}" for c in CATS
        if sum(r['cat'] == c for r in W) >= 10) + '\n')


def attractor_tables(D):
    print('### Wrong-answer structure (per mixed item)\n')
    print('| set | items | distinct wrong answers (median) | attractor share of wrong (median) | attractor >= 50% of wrong | attractor count > n_right |')
    print('|---|---|---|---|---|---|')
    for s in SETS:
        I = D[s][1]
        print(f"| {s} | {len(I)} | {S.median([i['n_distinct_wrong'] for i in I]):.0f} | {S.median([i['attractor_share_wrong'] for i in I]):.2f} "
              f"| {pc(sum(i['attractor_share_wrong'] >= .5 for i in I), len(I))} | {sum(i['attractor_n'] > i['n_right'] for i in I)} |")
    print('\nBy band (dev+dev2): distinct wrong / attractor share of wrong / attractor type\n')
    T = ['right_canonical_intermediate', 'right_intermediate', 'right_rare_value', 'question_number', 'novel', 'none']
    print('| band | items | distinct wrong | attr. share | ' + ' | '.join(t.replace('right_', 'R_') for t in T) + ' |')
    print('|---|---|---|---|' + '---|' * len(T))
    I = [i for s in SETS for i in D[s][1]]
    for b in BN + ['all']:
        J = [i for i in I if b == 'all' or i['band'] == b]
        print(f"| {b} | {len(J)} | {S.median([i['n_distinct_wrong'] for i in J]):.0f} | {S.median([i['attractor_share_wrong'] for i in J]):.2f} | " +
              ' | '.join(pc(sum(i['attractor_type'] == t for i in J), len(J)) for t in T) + ' |')
    print()


def baseline(s):
    """logp distribution of all sampled (non-forced, non-calc) tokens of mixed-item attempts"""
    mixed = {it['idx'] for it in D_[s][1]}
    n = n5 = n1 = 0
    with gzip.open(f'{HERE}/data/attempts_{s}.jsonl.gz', 'rt') as f:
        for l in f:
            a = json.loads(l)
            if a['idx'] not in mixed: continue
            for lp, wf in zip(a['seg_logp'][a['phrase_len']:], a['seg_wf'][a['phrase_len']:]):
                if wf & 1: continue
                n += 1; n5 += lp < math.log(.5); n1 += lp < math.log(.1)
    return n, n5, n1


def branch_tables(D):
    print('### Token branch points (wrong attempt vs right attempt with the longest common token prefix, same context group)\n')
    print('| set | wrong w/ partner | branch rel. pos median | p_w median | p_w<0.5 | p_w<0.1 | p_w<p_r | b in D-line | b before D-line | b after D |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for s in SETS:
        B = [r for r in D[s][0] if not r['ok'] and 'b_tok' in r and r['prefix_maxdlp'] < 0.5]
        W = [r for r in D[s][0] if not r['ok']]
        pw = [math.exp(r['lp_w']) for r in B]
        bd = collections.Counter(r.get('b_vs_div') for r in B)
        nb = sum(bd[k] for k in ('in_line', 'before', 'after'))
        print(f"| {s} | {len(B)}/{len(W)} | {S.median([r['b_rel'] for r in B]):.2f} | {S.median(pw):.2f} | {pc(sum(p < .5 for p in pw), len(B))} "
              f"| {pc(sum(p < .1 for p in pw), len(B))} | {pc(sum(r['lp_w'] < r['lp_r'] for r in B), len(B))} "
              f"| {pc(bd['in_line'], nb)} | {pc(bd['before'], nb)} | {pc(bd['after'], nb)} |")
        n, n5, n1 = baseline(s)
        print(f"| {s} baseline: all sampled tokens | {n} | | | {pc(n5, n)} | {pc(n1, n)} | | | | |")
    B = [r for s in SETS for r in D[s][0] if not r['ok'] and 'b_tok' in r and r['prefix_maxdlp'] < 0.5]
    print('\nBranch token type (dev+dev2) and probabilities:\n')
    print('| subset | n | type: word / digit / punct+space+newline / operator | p_w median | p_r median | p_w<0.1 | p_w+p_r median |')
    print('|---|---|---|---|---|---|---|')
    for nm, X in (('all branches', B), ('in D-line ("critical token")', [r for r in B if r.get('b_vs_div') == 'in_line']),
                  ('before D-line', [r for r in B if r.get('b_vs_div') == 'before']),
                  ('no D (answer = right value)', [r for r in B if r.get('div_kind') == 'none'])):
        if not X: continue
        t = collections.Counter(r['b_type'] for r in X)
        pw = [math.exp(r['lp_w']) for r in X]; pr = [math.exp(r['lp_r']) for r in X]
        print(f"| {nm} | {len(X)} | {pc(t['word'], len(X))} / {pc(t['digit'], len(X))} / {pc(t['punct'] + t['space'] + t['newline'], len(X))} / {pc(t['operator'], len(X))} "
              f"| {S.median(pw):.2f} | {S.median(pr):.2f} | {pc(sum(p < .1 for p in pw), len(X))} | {S.median([a + b for a, b in zip(pw, pr)]):.2f} |")
    IL = [r for r in B if r.get('b_vs_div') == 'in_line']
    print('\nCritical (in-D-line) branches by category: ' + ', '.join(
        f"{c} {sum(r['cat'] == c for r in IL)}/{sum(r['cat'] == c for r in B)} (p_w med {S.median([math.exp(r['lp_w']) for r in IL if r['cat'] == c]):.2f})"
        for c in CATS if sum(r['cat'] == c for r in IL) >= 5))
    hist = collections.Counter(min(4, int(-r['lp_w'] / math.log(10) * 2)) for r in IL)
    print('\np_w histogram of critical tokens: ' + ', '.join(
        f"{lab} {pc(hist[i], len(IL))}" for i, lab in enumerate(['>0.32', '0.1-0.32', '0.032-0.1', '0.01-0.032', '<0.01'])) + '\n')
    return IL


def gap_tables(D):
    print('### Selection gap: seed votes (N=8) that fail although an attempt of that seed is right\n')
    rows = []
    for s in SETS:
        A, I = D[s]
        byi = collections.defaultdict(list)
        for r in A: byi[r['idx']].append(r)
        for it in I:
            for sd in SEEDS:
                if it['vote_ok'][sd] or not it['n_right_seed'][sd]: continue
                atts = [r for r in byi[it['idx']] if str(r['seed']) == sd]
                win = it['vote_ans'][sd]; wa = [r for r in atts if r['ans'] == win and not r['ok']]
                cnt = collections.Counter(r['ans'] for r in atts)
                nright = sum(r['ok'] for r in atts)
                rows.append(dict(set=s, idx=it['idx'], seed=sd, win=win, nwin=cnt[win], nright=nright,
                                 is_attr=win == it['attractor'], atype=it['attractor_type'] if win == it['attractor'] else 'other-wrong',
                                 cat=collections.Counter(r['cat'] for r in wa).most_common(1)[0][0] if wa else 'none',
                                 none=win is None, band=it['band']))
    print('| set | gap cases (item x seed) | per seed | winner = 32-attempt attractor | winner count > right count | tie (=) | winner null |')
    print('|---|---|---|---|---|---|---|')
    for s in SETS:
        X = [r for r in rows if r['set'] == s]
        print(f"| {s} | {len(X)} | {len(X) / 4:.1f} | {pc(sum(r['is_attr'] for r in X), len(X))} | {pc(sum(r['nwin'] > r['nright'] for r in X), len(X))} "
              f"| {pc(sum(r['nwin'] == r['nright'] for r in X), len(X))} | {sum(r['none'] for r in X)} |")
    print('\nWinner type / category (dev+dev2):', ', '.join(f'{k} {v}' for k, v in collections.Counter(r['atype'] for r in rows).most_common()),
          '| cat:', ', '.join(f'{k} {v}' for k, v in collections.Counter(r['cat'] for r in rows).most_common()),
          '| band:', ', '.join(f'{k} {v}' for k, v in sorted(collections.Counter(r['band'] for r in rows).items())))
    print('\nRight count in the failing seed:', ', '.join(f'{k}: {v}' for k, v in sorted(collections.Counter(r['nright'] for r in rows).items())))
    print()
    return rows


def examples(D):
    S_, H, R = load_calib()
    print('### Examples (hand-labelled; W = wrong divergent line, R = closest right attempt)\n')
    by = collections.defaultdict(list)
    for k, (c, n) in H.items(): by[c].append(k)
    for c in CATS:
        if c not in by: continue
        print(f'**{c}**\n')
        for k in by[c][:8]:
            r = R[S_[k]['wrong']]
            w = (r.get('div_line') or '')[:110]; rl = (r.get('right_line') or '')[:110]
            print(f"- {k[0]} {k[1]} ({H[k][1]}): W `{w}` / R `{rl}`")
        print()


if __name__ == '__main__':
    D_ = {s: load(s) for s in SETS}
    which = sys.argv[1:] or ['cat', 'hand', 'pos', 'attr', 'branch', 'gap', 'ex']
    F = dict(cat=cat_tables, hand=hand_tables, pos=position_tables, attr=attractor_tables, branch=branch_tables,
             gap=gap_tables, ex=examples)
    for w in which: F[w](D_)
