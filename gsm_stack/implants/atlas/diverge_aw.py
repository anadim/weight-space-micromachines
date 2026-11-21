"""Divergence categories on the all-weights atlas (reuses atlas/diverge.py heuristic unchanged).

usage: python3 atlas/diverge_aw.py DATA_DIR [--seeds 7,11,13,17] [--sets dev,dev2]
  -> DATA_DIR/diverge_{set}.jsonl.gz (rec=att per attempt of a mixed item, rec=item per item) + markdown tables on stdout.
"""
import os, sys, json, gzip, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diverge as DV                     # noqa: E402
from parse import Tok                    # noqa: E402

OLD_HAND = dict(operand=6, relation=43, dropped=15, unit=11, target=8, plan=8, arithmetic=4, early_stop=3, format=3)  # notes/atlas.md
OLD_HEUR = dict(operand=13, relation=48, dropped=19, unit=2, target=7, plan=5, arithmetic=1, early_stop=3, format=0, other=1)


def run(d, s, seeds, tk):
    A = [a for a in DV.rd(f'{d}/attempts_{s}.jsonl.gz') if not seeds or a['seed'] in seeds]
    by = collections.defaultdict(list)
    for a in A: by[a['idx']].append(a)
    rows, items = [], []
    for it in DV.rd(f'{d}/items_{s}.jsonl.gz'):
        L = by.get(it['idx'], [])
        r = sum(a['ok'] for a in L)
        if not L or r == 0 or r == len(L): continue
        it = dict(it, n_right=r, n_att=len(L))
        try:
            rr, item = DV.analyse_item(it, L, tk)
        except Exception as e:                                  # keep going; report
            print(f'  ! {s} {it["idx"]}: {type(e).__name__} {e}', file=sys.stderr); continue
        rows += rr; items.append(item)
    with gzip.open(f'{d}/diverge_{s}.jsonl.gz', 'wt') as f:
        for r in rows + items: f.write(json.dumps(r) + '\n')
    return rows, items


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('d'); ap.add_argument('--seeds', default=''); ap.add_argument('--sets', default='dev,dev2')
    a = ap.parse_args(); seeds = {int(x) for x in a.seeds.split(',') if x}; tk = Tok()
    R = {}
    for s in a.sets.split(','):
        R[s] = run(a.d, s, seeds, tk)
    print('\n| category | ' + ' | '.join(f'{s} att | {s} item' for s in R) + ' | old S3 heur att % | old S3 hand % |')
    print('|---|' + '---|---|' * len(R) + '---|---|')
    for c in DV.CATS:
        cells = []
        for s, (rows, items) in R.items():
            wr = [r for r in rows if not r['ok']]; na = sum(r['cat'] == c for r in wr); ni = sum(i['dominant_cat'] == c for i in items)
            cells.append(f'{na} ({100 * na / max(1, len(wr)):.0f}%) | {ni} ({100 * ni / max(1, len(items)):.0f}%)')
        print(f'| {c} | ' + ' | '.join(cells) + f' | {OLD_HEUR.get(c, 0)}% | {OLD_HAND.get(c, "-")}{"%" if c in OLD_HAND else ""} |')
    for s, (rows, items) in R.items():
        wr = [r for r in rows if not r['ok']]
        print(f'\n{s}: mixed items {len(items)}, wrong attempts {len(wr)}; top subs: ' +
              ', '.join(f'{c}/{u} {n}' for (c, u), n in collections.Counter((r['cat'], r['sub']) for r in wr).most_common(12)))


if __name__ == '__main__':
    main()
