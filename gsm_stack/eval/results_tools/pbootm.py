"""Seed-pooled paired bootstrap, numpy-free. usage: python3 pbootm.py A1.jsonl,A2.jsonl[,..] B1.jsonl,B2.jsonl[,..]
Each group is averaged over its seed files per item; the delta (B − A) is resampled over items (4000, percentile).
The CI covers item sampling only, not seed variance: also prints per-seed accs and their sd."""
import sys, os, json, random, statistics as st
from pboot import L, boot


def grp(arg):
    fs = arg.split(','); R = [L(f) for f in fs]
    assert all(len(r) == len(R[0]) for r in R)
    assert all(str(a['gold']) == str(b['gold']) for r in R[1:] for a, b in zip(r, R[0]))
    accs = [100 * sum(float(x['ok']) for x in r) / len(r) for r in R]
    m = [sum(float(r[i]['ok']) for r in R) / len(R) for i in range(len(R[0]))]
    return fs, accs, m, R[0]


fa, aa, ma, ra = grp(sys.argv[1]); fb, ab, mb, rb = grp(sys.argv[2])
assert len(ma) == len(mb) and all(str(a['gold']) == str(b['gold']) for a, b in zip(ra, rb))
d = [b - a for a, b in zip(ma, mb)]
sd = lambda x: round(st.stdev(x), 2) if len(x) > 1 else None
print(json.dumps(dict(A=[os.path.basename(f) for f in fa], A_accs=[round(x, 2) for x in aa], A_mean=round(st.mean(aa), 2),
                      A_sd=sd(aa), B=[os.path.basename(f) for f in fb], B_accs=[round(x, 2) for x in ab],
                      B_mean=round(st.mean(ab), 2), B_sd=sd(ab), delta=round(100 * sum(d) / len(d), 2), delta_ci=boot(d))))
