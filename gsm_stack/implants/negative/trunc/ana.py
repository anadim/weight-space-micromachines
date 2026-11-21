"""Analyse implants/trunc runs vs the S3-serial baseline (same seeds).

usage: python3.11 ana.py [--exact]   (reads job/out/*.jsonl.gz and gsm_stack/implants/gen/out baselines)
Per config x set (mean over seeds 7, 11): vote acc, per-attempt acc (atlas/parse.py attempt answers), oracle@8,
distinct answers / item, tokens / question, fixes / breaks vs the same-seed baseline (item-level vote ok).
"""
import os, re, sys, json, gzip, glob, argparse, collections
HERE = os.path.dirname(os.path.abspath(__file__)); WS = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(WS, 'atlas'))
import parse as PA

GEN = os.path.join(WS, 'gen', 'out'); OUT = os.path.join(HERE, 'job', 'out')
SEEDS = (7, 11)


def rd(fn): return [json.loads(l) for l in gzip.open(fn, 'rt')]


CLS = {}


def cls_of(set_):
    if set_ not in CLS:
        CLS[set_] = {d['idx']: d['class'] for d in rd(os.path.join(WS, 'atlas', 'data', f'items_{set_}.jsonl.gz'))}
    return CLS[set_]


def stats(rows, tk, set_, seed):
    att_ok = att_n = orc = nd = mo = mn = 0; C = cls_of(set_)
    for r in rows:
        P = PA.parse_stream(r, tk); A = PA.attempt_rows(r, P, tk, set_, seed)
        att_ok += sum(a['ok'] for a in A); att_n += len(A); orc += any(a['ok'] for a in A)
        if C[r['idx']] == 'mixed': mo += sum(a['ok'] for a in A); mn += len(A)
        nd += len({a['ans'] for a in A if a['ans'] is not None})
    n = len(rows)
    return dict(vote=100 * sum(r['ok'] for r in rows) / n, att=100 * att_ok / att_n, orc=100 * orc / n, nd=nd / n,
                tok=sum(r['ntok'] for r in rows) / n, natt=att_n / n, mix=100 * mo / mn)


def exact():
    for tag in ('', '_mp0'):
        f = f'{OUT}/dev_N8_R1_v1_s1_m0_T1.0{tag}_seed7_n128.jsonl.gz'
        if not os.path.exists(f): print('missing', f); continue
        A = rd(f); B = rd(f'{GEN}/dev_N8_R1_v1_s1_m0_T1.0_seed7.jsonl.gz')[:len(A)]
        same = {k: sum(a[k] == b[k] for a, b in zip(A, B)) for k in ('tokens', 'wf', 'logp', 'val', 'lg', 'ok')}
        print(f'exactness {tag or "orig"}: {len(A)} rows vs baseline rows 0..{len(A)-1}:', same)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--exact', action='store_true'); a = ap.parse_args()
    if a.exact: exact()
    tk = PA.Tok()
    cfgs = collections.OrderedDict([('baseline T1.0', 'T1.0')])
    for f in sorted(glob.glob(f'{OUT}/dev_N8_*_seed7.jsonl.gz')):
        c = re.search(r'_m0_(.*)_seed7', f).group(1)
        if c != 'T1.0': cfgs[c] = c
    print('| config | set | vote | per-att | per-att (mixed items) | oracle@8 | distinct ans | tok/q | fixes | breaks | net |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for name, c in cfgs.items():
        for set_ in ('dev', 'dev2'):
            S = []; fx = br = 0; ok = True
            for sd in SEEDS:
                fn = f'{GEN if name.startswith("baseline") else OUT}/{set_}_N8_R1_v1_s1_m0_{c}_seed{sd}.jsonl.gz'
                if not os.path.exists(fn): ok = False; break
                R = rd(fn); B = rd(f'{GEN}/{set_}_N8_R1_v1_s1_m0_T1.0_seed{sd}.jsonl.gz')
                assert [r['q'] for r in R] == [b['q'] for b in B]
                fx += sum(r['ok'] and not b['ok'] for r, b in zip(R, B)); br += sum(b['ok'] and not r['ok'] for r, b in zip(R, B))
                S.append(stats(R, tk, set_, sd))
            if not ok: continue
            m = {k: sum(s[k] for s in S) / len(S) for k in S[0]}
            print(f"| {name} | {set_} | {m['vote']:.2f} | {m['att']:.2f} | {m['mix']:.2f} | {m['orc']:.2f} | {m['nd']:.2f} | {m['tok']:.0f} | "
                  f"{fx} | {br} | {fx - br:+d} |")
    print(f'(means over seeds {SEEDS}; fixes/breaks summed over seeds, item-level vote ok vs same-seed baseline)')


if __name__ == '__main__':
    main()
