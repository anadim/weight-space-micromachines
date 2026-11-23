"""[32](b) 1.7B matched-compute lists: N_set = round(row-4 tok/q / row-2 tok/q) (seed means, uniform/items.py), SCH = N independent host-only
samples (row-2 protocol: KIND H, T=1 seed 7) + plain majority, for P17 s0/s1/s2 (row 2) and Base17 (row 1).
usage: python3 uniform/rl28/gsm8k/mk32.py  -> fe/lists/q32b_test.list, fe/lists/q32b_hard.list; prints N."""
import json, os, sys
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import items   # uniform/items.py: reduced per-item rows (bundles) or your own full rows via $FE_ROWS


def tokq(st, names):
    v = [r['ntok'] for n in names for r in items.load(st, n)]
    return sum(v) / len(v)


for st in ('test', 'hard'):
    t4 = tokq(st, [f'r2_CW17_s{s}' for s in range(3)]); t2 = tokq(st, [f'P17_s{s}' for s in range(3)])
    N = round(t4 / t2)
    L = [f'# [32](b) 1.7B matched compute ({st}): N = round(row-4 tok/q {t4:.0f} / row-2 tok/q {t2:.1f}) = {N}; SCH, T=1 seed 7, once']
    L += [f'scH{N}_P17_s{s} P17_s{s} {st} SCH {N} 0 1 1 0' for s in range(3)]
    L += [f'scH{N}_base17 base {st} SCH {N} 0 1 1 0']
    open(os.path.join(H, 'fe', 'lists', f'q32b_{st}.list'), 'w').write('\n'.join(L) + '\n')
    print(st, 'N', N, 'row4 tok/q', round(t4), 'row2 tok/q', round(t2, 1))
