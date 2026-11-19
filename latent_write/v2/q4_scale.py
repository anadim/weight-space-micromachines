"""Q4: same slot write on a larger frozen model (set LW_MODEL). Task '(a*b)+c' (mid / hard ranges of latent_write),
written at the span 'a*b' (right-aligned, fill with the oracle ' (' state), slot-only and slot+suffix, every 2nd layer.
Usage: LW_MODEL=Qwen/Qwen3-1.7B-Base python q4_scale.py <mid|hard> [layers]
"""
import json, random, sys, time
from lw2 import *

RN = sys.argv[1]
load()
NL = nlayers()
LAYS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else list(range(0, NL, 2))
NTE, NFIT = 300, 2000
rng = random.Random(11)
res = {"range": RN, "model": MODEL, "n_layers": NL}
OUT = f"results/q4_{RN}_{MODEL.split('/')[-1]}.json"
t0 = time.time()


def log(k, v):
    res[k] = v
    print(f"[{time.time()-t0:6.0f}s] {k}: {v}", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)


R = {"mid": ((11, 29), (3, 9), (10, 99), 3), "hard": ((12, 49), (12, 29), (100, 899), 4)}[RN]
W = R[3]


def sample(r):
    while True:
        a, b, c = r.randint(*R[0]), r.randint(*R[1]), r.randint(*R[2])
        if len(str(a * b)) == W:
            return a, b, c


pool = sorted({a * b for a in range(R[0][0], R[0][1] + 1) for b in range(R[1][0], R[1][1] + 1) if len(str(a * b)) == W})
rs = random.Random(1234)
shots = [sample(rs) for _ in range(4)]
prefix = shots_prefix([(f"What is ({a}*{b})+{c}?", f" {a*b+c}") for a, b, c in shots])
prefix_tool = "".join(f"Q: What is ({a}*{b})+{c}?\nTool: {a}*{b} = {a*b}\nA: {a*b+c}\n\n" for a, b, c in shots)
T = lambda it: prefix + f"Q: What is ([[{it[0]}*{it[1]}]])+{it[2]}?\nA:"
O = lambda it, v: prefix + f"Q: What is ([[{v}]])+{it[2]}?\nA:"
test = [sample(rng) for _ in range(NTE)]
TP = [a * b for a, b, c in test]
CP = []
for v in TP:
    w = rng.choice(pool)
    while w == v:
        w = rng.choice(pool)
    CP.append(w)
cands = [[f" {TP[i]+c}", f" {CP[i]+c}"] for i, (a, b, c) in enumerate(test)]
plain = lambda ps: [unmark(p)[0] for p in ps]
tm = [T(it) for it in test]
log("none", rates(tf_score(plain(tm), cands)))
log("text_oracle", rates(tf_score(plain([O(it, TP[i]) for i, it in enumerate(test)]), cands)))
log("text_oracle_corr", rates(tf_score(plain([O(it, CP[i]) for i, it in enumerate(test)]), cands)))
log("text_tool_line", rates(tf_score([prefix_tool + f"Q: What is ({a}*{b})+{c}?\nTool: {a}*{b} = {a*b}\nA:" for a, b, c in test], cands)))
log("mul_alone", rates(tf_score([prefix.replace("+", "#") + f"Q: What is {a}*{b}?\nA:" for a, b, c in test], [[f" {a*b}", " x"] for a, b, c in test])))
fit = []
for _ in range(NFIT):
    it = sample(rng); v = rng.randint(10 ** (W - 1), 10 ** W - 1)
    fit.append((O(it, v), v))
feat = lambda v: f_num(v, W)
sw = SpanWrite(tm[0], O(test[0], TP[0]), [0], feat, fit, LAYS)
assert all(sw.ok(tm[i], O(it, TP[i])) for i, it in enumerate(test))
log("pairs", sw.pairs)
log("r2", sw.r2)
for L in LAYS:
    los = {0: L}
    log(f"L{L}:slot:true", rates(tf_score(plain(tm), cands, edits=sw.synth(TP, los))))
    log(f"L{L}:slot:corr", rates(tf_score(plain(tm), cands, edits=sw.synth(CP, los))))
    log(f"L{L}:slot:mean", rates(tf_score(plain(tm), cands, edits=sw.mean(NTE, los))))
# slot + suffix (window extension); writer also gets c (suffix positions hold c)
CW = len(str(R[2][0]))
feat2 = lambda v: f_num(v[0], W) + f_num(v[1], CW)
fit2 = [(om, (v, int(unmark(om)[0].split("+")[-1].split("?")[0]))) for om, v in fit]
sws = SpanWrite(tm[0], O(test[0], TP[0]), [0], feat2, fit2, [L for L in LAYS if L >= NL // 3], with_suffix=True)
for L in sws.lays:
    los = {0: L, -1: L}
    log(f"L{L}:slot+suffix:true", rates(tf_score(plain(tm), cands, edits=sws.synth([(TP[i], it[2]) for i, it in enumerate(test)], los))))
    log(f"L{L}:slot+suffix:corr", rates(tf_score(plain(tm), cands, edits=sws.synth([(CP[i], it[2]) for i, it in enumerate(test)], los))))
print("DONE", flush=True)
