"""Analysis of the forced re-read implant runs (implants/reread).
usage: python3 analyze.py [OUTDIR] [TAGS]   TAGS: comma list of variant tags (default rrfull,rrask,rrphrase)
Per variant x set (mean over seeds 7, 11): per-attempt accuracy (k=0, k>=1; attempt ok = register valid & value == gold),
vote acc, oracle@8, item-level fixes / breaks vs the same-seed base run (gen/out), tokens/q (all, host-generated only).
Exactness: wave-0 rows (idx < 128) must have attempt-0 tokens (through register end[0]) bit-identical to the base run.
Breakdown by atlas item category (atlas/data/diverge_{set}.jsonl.gz, dominant_cat of mixed items) of per-attempt acc k>=1.
"""
import gzip, json, os, sys, collections, statistics as S

W = os.path.expanduser(os.environ.get('WSM_ROOT', '.') + '/gsm_stack/implants')
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'job', 'out')
TAGS = (sys.argv[2] if len(sys.argv) > 2 else 'rrfull,rrask,rrphrase').split(',')
SEEDS = (7, 11)


def correct(pred, gold):
    try: return pred is not None and abs(float(pred) - float(gold)) <= 1e-6 * max(1.0, abs(float(gold)))
    except ValueError: return False


def load(f):
    if not os.path.exists(f): return None
    return {r['idx']: r for r in map(json.loads, gzip.open(f, 'rt'))}


def att_ok(r):
    n = r['natt']
    return [bool(r['rok'][k]) and correct(r['val'][k], r['gold']) for k in range(min(n, 8))]


def stats(R):
    a0 = [att_ok(r)[0] for r in R.values() if r['natt'] >= 1]
    a1 = [x for r in R.values() for x in att_ok(r)[1:]]
    return dict(a0=100 * S.mean(a0), a1=100 * S.mean(a1), vote=100 * S.mean(r['ok'] for r in R.values()),
                orc=100 * S.mean(any(att_ok(r)) for r in R.values()), tok=S.mean(r['ntok'] for r in R.values()),
                gtok=S.mean(sum(1 for w in r['wf'] if not (w & 2)) for r in R.values()),
                natt=S.mean(r['natt'] for r in R.values()))


def exact(B, R):
    n = same = 0
    for i in range(128):
        if i not in B or i not in R: continue
        b, r = B[i], R[i]; e = b['end'][0]
        n += 1; same += r['end'][0] == e and r['tokens'][:e + 1] == b['tokens'][:e + 1]
    return same, n


def main():
    cats = {}
    for s in ('dev', 'dev2'):
        for l in gzip.open(f'{W}/atlas/data/diverge_{s}.jsonl.gz', 'rt'):
            r = json.loads(l)
            if r['rec'] == 'item': cats[(s, r['idx'])] = r['dominant_cat']
    rows = []; bycat = collections.defaultdict(lambda: [0, 0, 0, 0])
    print('| variant | set | seeds | att acc k=0 | att acc k>=1 | vote | oracle@8 | fixes/breaks | tok/q | gen tok/q | exact k=0 (wave 0) |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for tag in ['base'] + TAGS:
        for s in ('dev', 'dev2'):
            acc = []; fx = br = 0; ex = [0, 0]; seeds = []
            for sd in SEEDS:
                B = load(f'{W}/gen/out/{s}_N8_R1_v1_s1_m0_T1.0_seed{sd}.jsonl.gz')
                R = B if tag == 'base' else load(f'{OUT}/{s}_N8_R1_v1_s1_m0_T1.0_seed{sd}_{tag}.jsonl.gz')
                if R is None: continue
                seeds.append(sd); B = {i: B[i] for i in R}; acc.append(stats(R))
                fx += sum(R[i]['ok'] and not B[i]['ok'] for i in R); br += sum(B[i]['ok'] and not R[i]['ok'] for i in R)
                if tag != 'base':
                    a, b = exact(B, R); ex[0] += a; ex[1] += b
                for i in R:
                    c = cats.get((s, i), 'not-mixed')
                    nb = att_ok(B[i])[1:]; nr = att_ok(R[i])[1:]
                    x = bycat[(tag, c)]; x[0] += sum(nr); x[1] += len(nr); x[2] += sum(nb); x[3] += len(nb)
            if not acc: continue
            m = {k: S.mean(a[k] for a in acc) for k in acc[0]}
            n = len(R)
            print(f"| {tag} | {s} (n={n}) | {','.join(map(str, seeds))} | {m['a0']:.2f} | {m['a1']:.2f} | {m['vote']:.2f} | {m['orc']:.2f} | "
                  f"{fx}/{br if tag != 'base' else '-'} | {m['tok']:.0f} | {m['gtok']:.0f} | " + (f"{ex[0]}/{ex[1]}" if tag != 'base' else '-') + ' |')
    print('\nPer-attempt acc k>=1 by atlas item category (dev+dev2 pooled, both seeds): variant vs same-item base')
    CATS = ['relation', 'dropped', 'operand', 'target', 'unit', 'plan', 'early_stop', 'arithmetic', 'format', 'other', 'not-mixed']
    print('| category | ' + ' | '.join(f'{t} (base -> {t})' for t in TAGS) + ' |')
    print('|---|' + '---|' * len(TAGS))
    for c in CATS:
        cells = []
        for t in TAGS:
            x = bycat.get((t, c))
            cells.append(f'{100 * x[2] / x[3]:.1f} -> {100 * x[0] / x[1]:.1f} (n={x[1]})' if x and x[1] and x[3] else '-')
        print(f'| {c} | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main()
