"""eq_cmp.py REF_DIR DIR [DIR ...] : per matching seed/set file, token identity of rows vs REF (row-by-row idx),
accuracy, tok/q, fixes/breaks (ok 0->1 / 1->0). Files matched on (set, seed) parsed from the tag."""
import gzip, json, glob, re, sys, os


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, '*.jsonl.gz'))):
        b = os.path.basename(f)
        m = re.match(r'(dev2|dev|val)_.*_seed(\d+)', b)
        if not m: continue
        rows = [json.loads(l) for l in gzip.open(f, 'rt')]
        out[(m.group(1), int(m.group(2)))] = {r['idx']: r for r in rows}
    return out


ref = load(sys.argv[1])
for d in sys.argv[2:]:
    cur = load(d)
    for k in sorted(cur):
        if k not in ref: print('EQCMP', d, k, 'no ref'); continue
        R, C = ref[k], cur[k]
        ids = sorted(set(R) & set(C)); n = len(ids)
        same = sum(R[i]['tokens'] == C[i]['tokens'] for i in ids)
        pref = [next((j for j, (a, b) in enumerate(zip(R[i]['tokens'], C[i]['tokens'])) if a != b), min(len(R[i]['tokens']), len(C[i]['tokens'])))
                for i in ids if R[i]['tokens'] != C[i]['tokens']]
        fx = sum(C[i]['ok'] and not R[i]['ok'] for i in ids); br = sum(R[i]['ok'] and not C[i]['ok'] for i in ids)
        ar = 100 * sum(R[i]['ok'] for i in ids) / n; ac = 100 * sum(C[i]['ok'] for i in ids) / n
        print(f"EQCMP {os.path.basename(d.rstrip('/'))} vs {os.path.basename(sys.argv[1].rstrip('/'))} {k[0]} s{k[1]} n={n} "
              f"tok-identical={same}/{n} first-diff-median={sorted(pref)[len(pref)//2] if pref else '-'} "
              f"acc {ar:.2f}->{ac:.2f} fixes={fx} breaks={br} tok/q {sum(R[i]['ntok'] for i in ids)/n:.1f}->{sum(C[i]['ntok'] for i in ids)/n:.1f}")
