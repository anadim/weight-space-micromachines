"""All-weights failure atlas: item classes, per-attempt-index accuracy, vote-error split, selection gap (atlas-aw).

usage: python3 atlas/analysis_aw.py DATA_DIR [--seeds 7,11,13,17] [--label r1] > section.md
DATA_DIR holds attempts_{set}.jsonl.gz / items_{set}.jsonl.gz from atlas/parse.py (gen-format aw rows).
Writes DATA_DIR/summary_{set}.json (numbers used by notes/atlas_aw.md).
"""
import sys, json, gzip, argparse, collections, statistics as st


def rd(fn):
    with gzip.open(fn, 'rt') as f:
        return [json.loads(l) for l in f]


def pct(a, n): return 100.0 * a / n if n else float('nan')


def run(d, s, seeds, label):
    A = [a for a in rd(f'{d}/attempts_{s}.jsonl.gz') if not seeds or a['seed'] in seeds]
    I = {it['idx']: it for it in rd(f'{d}/items_{s}.jsonl.gz')}
    seeds = sorted({a['seed'] for a in A})
    by = collections.defaultdict(list)
    for a in A: by[(a['idx'], a['seed'])].append(a)
    byi = collections.defaultdict(list)
    for a in A: byi[a['idx']].append(a)
    n = len(I); out = dict(set=s, label=label, seeds=seeds, n=n)
    # ---- classes over all attempts present
    cls = collections.Counter()
    for i, L in byi.items():
        r = sum(a['ok'] for a in L); cls['always' if r == len(L) else 'never' if r == 0 else 'mixed'] += 1
    out['classes'] = dict(cls); out['att_per_item'] = st.mean(len(L) for L in byi.values())
    print(f'\n### {label} {s}: {n} items, seeds {seeds}, {len(A)} attempts ({out["att_per_item"]:.1f}/item)\n')
    print(f'- classes: always {cls["always"]} ({pct(cls["always"], n):.1f}%), mixed {cls["mixed"]} ({pct(cls["mixed"], n):.1f}%), '
          f'never {cls["never"]} ({pct(cls["never"], n):.1f}%)')
    # ---- per-k
    K = collections.defaultdict(lambda: collections.Counter())
    for a in A:
        c = K[a['k']]; c['n'] += 1; c['ok'] += a['ok']; c['na'] += a['ans'] is None; c['rep'] += a['nrep'] > 0
        c['bad'] += a['bad'] and a['ans'] is not None
    print('\n| k | n | acc % | no answer % | repaired % | bad (non-int/neg) % |\n|---|---|---|---|---|---|')
    for k in sorted(K):
        c = K[k]; print(f"| {k} | {c['n']} | {pct(c['ok'], c['n']):.1f} | {pct(c['na'], c['n']):.1f} | {pct(c['rep'], c['n']):.1f} | {pct(c['bad'], c['n']):.1f} |")
    out['perk'] = {k: dict(n=K[k]['n'], acc=round(pct(K[k]['ok'], K[k]['n']), 2), na=round(pct(K[k]['na'], K[k]['n']), 2)) for k in sorted(K)}
    # per-k on a fixed item set (items whose stream reached k in every seed) to remove selection effects of adaptive N
    # ---- position bins
    P = collections.defaultdict(lambda: collections.Counter())
    for a in A:
        b = min(a['seg_start'] // 500, 7); P[b]['n'] += 1; P[b]['ok'] += a['ok']; P[b]['na'] += a['ans'] is None
    print('\n| attempt start (stream token) | n | acc % | no answer % |\n|---|---|---|---|')
    for b in sorted(P):
        c = P[b]; print(f"| {b * 500}{'+' if b == 7 else '-' + str(b * 500 + 499)} | {c['n']} | {pct(c['ok'], c['n']):.1f} | {pct(c['na'], c['n']):.1f} |")
    out['pos'] = {b * 500: dict(n=P[b]['n'], acc=round(pct(P[b]['ok'], P[b]['n']), 2), na=round(pct(P[b]['na'], P[b]['n']), 2)) for b in sorted(P)}
    # ---- per seed: vote, oracle@8, split of vote errors
    print('\n| seed | vote % | oracle (all attempts of the stream) % | oracle@8 (k<8) % | vote wrong, some attempt right | vote wrong, none right | mean attempt acc % | attempts/stream |')
    print('|---|---|---|---|---|---|---|---|')
    T = collections.Counter(); gaps = []
    for sd in seeds:
        v = o = o8 = g = nn = 0; ma = na_ = 0
        for i, it in I.items():
            L = by[(i, sd)]
            if not L: continue
            vk = it['vote_ok'].get(str(sd), 0); anyr = any(a['ok'] for a in L)
            v += vk; o += anyr; o8 += any(a['ok'] for a in L if a['k'] < 8)
            ma += sum(a['ok'] for a in L); na_ += len(L)
            if not vk and anyr: g += 1; gaps.append((i, sd))
            if not vk and not anyr: nn += 1
        T.update(v=v, o=o, o8=o8, g=g, nn=nn, ma=ma, na=na_)
        print(f'| {sd} | {pct(v, n):.2f} | {pct(o, n):.2f} | {pct(o8, n):.2f} | {g} ({pct(g, n):.2f}%) | {nn} ({pct(nn, n):.2f}%) | {pct(ma, na_):.2f} | {na_ / n:.2f} |')
    m = len(seeds)
    print(f"| mean | {pct(T['v'] / m, n):.2f} | {pct(T['o'] / m, n):.2f} | {pct(T['o8'] / m, n):.2f} | {T['g'] / m:.1f} ({pct(T['g'] / m, n):.2f}%) | "
          f"{T['nn'] / m:.1f} ({pct(T['nn'] / m, n):.2f}%) | {pct(T['ma'], T['na']):.2f} | {T['na'] / n / m:.2f} |")
    out.update(vote=round(pct(T['v'] / m, n), 2), oracle=round(pct(T['o'] / m, n), 2), oracle8=round(pct(T['o8'] / m, n), 2),
               gap=round(pct(T['g'] / m, n), 2), none=round(pct(T['nn'] / m, n), 2), att_acc=round(pct(T['ma'], T['na']), 2))
    # ---- selection gap: right count vs winner count, probe lg of right vs winner
    rows = []
    for i, sd in gaps:
        L = by[(i, sd)]; it = I[i]; wans = it['vote_ans'].get(str(sd))
        win = [a for a in L if a['voted']]
        valid = [a for a in L if a['ans'] is not None and not a['bad']]
        cnt = collections.Counter(a['ans'] for a in valid)
        right = [a for a in L if a['ok']]
        try: wkey = float(wans) if wans is not None else None
        except ValueError: wkey = None
        wsl = [a for a in valid if wkey is not None and abs(float(a['ans']) - wkey) < 1e-6]
        lg_r = max((a['lg'] for a in right if a['lg'] is not None), default=None)
        lg_w = max((a['lg'] for a in wsl if a['lg'] is not None), default=None)
        rb = sum(a['bad'] for a in right)
        rows.append(dict(set=s, idx=i, seed=sd, gold=it['gold'], winner=wans, n_right=len(right), n_right_valid=len(right) - rb,
                         n_winner=len(wsl), n_valid=len(valid), n_distinct=len(cnt), lg_right_max=lg_r, lg_winner_max=lg_w,
                         lg_right_mean=st.mean([a['lg'] for a in right]) if right else None,
                         lg_winner_mean=st.mean([a['lg'] for a in wsl]) if wsl else None,
                         right_k=[a['k'] for a in right], winner_k=[a['k'] for a in wsl], right_bad=rb))
    if rows:
        rel = collections.Counter('right>winner' if r['n_right_valid'] > r['n_winner'] else 'tie' if r['n_right_valid'] == r['n_winner']
                                  else 'winner>right' for r in rows)
        rc = collections.Counter(min(r['n_right_valid'], 4) for r in rows)
        lgc = [r for r in rows if r['lg_right_max'] is not None and r['lg_winner_max'] is not None]
        lgbeat = sum(r['lg_right_max'] > r['lg_winner_max'] for r in lgc)
        lgm = sum(r['lg_right_mean'] > r['lg_winner_mean'] for r in lgc)
        tie_lg = [r for r in lgc if r['n_right_valid'] == r['n_winner']]
        allbad = sum(r['n_right_valid'] == 0 for r in rows)
        print(f'\nSelection gap ({len(rows)} item x seed cases, {len(rows) / m:.1f}/seed):')
        print(f'- right count (valid) vs winner count: {dict(rel)}; right attempts all flagged bad (impossible) in {allbad}')
        print(f'- valid right count in the failing stream: ' + ', '.join(f"{k}{'+' if k == 4 else ''}: {rc[k]}" for k in sorted(rc)))
        print(f'- probe lg: max lg(right) > max lg(winner) in {lgbeat}/{len(lgc)} ({pct(lgbeat, len(lgc)):.0f}%), mean lg(right) > mean lg(winner) '
              f'in {lgm}/{len(lgc)}; median lg right-max {st.median([r["lg_right_max"] for r in lgc]):.2f} vs winner-max {st.median([r["lg_winner_max"] for r in lgc]):.2f}'
              if lgc else '- probe lg: n/a')
        if tie_lg:
            tb = sum(r['lg_right_mean'] > r['lg_winner_mean'] for r in tie_lg)
            print(f'- ties (right count == winner count): {len(tie_lg)}, of which mean lg(right) > mean lg(winner) in {tb} (the voter weights by lg: these are lost on lg)')
        # oracle: what would a perfect tie-breaker / plurality-with-lg give
        out['gap_rows'] = len(rows); out['gap_rel'] = dict(rel); out['gap_rc'] = dict(rc); out['gap_lgbeat'] = [lgbeat, len(lgc)]
    with open(f'{d}/gap_{s}.json', 'w') as f: json.dump(rows, f)
    with open(f'{d}/summary_{s}.json', 'w') as f: json.dump(out, f, indent=1)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('d'); ap.add_argument('--seeds', default=''); ap.add_argument('--label', default='')
    ap.add_argument('--sets', default='dev,dev2')
    a = ap.parse_args(); seeds = {int(x) for x in a.seeds.split(',') if x}
    for s in a.sets.split(','): run(a.d, s, seeds, a.label)


if __name__ == '__main__':
    main()
