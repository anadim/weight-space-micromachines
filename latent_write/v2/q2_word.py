"""Q2: natural word problems with no explicit '(a*b)' span. Where must the product be written?
Templates (answer immediately, no CoT):
 lost : Each box has {a} pens. There are {b} boxes. {c} pens are lost. How many pens are left?   (P-c)
 rows : A farmer has {b} rows of trees. Each row has {a} trees. He plants {c} more trees. How many trees does he have now? (P+c)
Locations (task span <- oracle span; the oracle is a rewritten problem that states P in text):
 sentA / sentB : the sentence that holds a / b         <- ' There are P pens.' style sentence
 both          : both operand sentences                <- one sentence stating P
 numA          : only the number tokens of a (' 23')    <- the digits of P
Usage: python q2_word.py <template> [layers]
"""
import json, random, sys, time
from lw2 import *

TPL = sys.argv[1]
LAYS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,4,8,12,14,16,18").split(",")]
NTE, NFIT = 300, 2000
load()
rng = random.Random(21)
res = {"template": TPL, "model": MODEL}
import os
LOCS = os.environ.get("LOCS", "sentA,sentB,both,numA").split(",")
OUT = f"results/q2_{TPL}_{MODEL.split('/')[-1]}" + ("" if "LOCS" not in os.environ else "_" + "-".join(LOCS) + "_L" + "-".join(map(str, LAYS))) + ".json"
t0 = time.time()


def log(k, v):
    res[k] = v
    print(f"[{time.time()-t0:6.0f}s] {k}: {v}", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)


def sample(r):
    while True:
        a, b, c = r.randint(11, 29), r.randint(3, 9), r.randint(10, 99)
        if a * b >= 100:
            return a, b, c


pool = [a * b for a in range(11, 30) for b in range(3, 10) if a * b >= 100]

if TPL == "lost":
    S_A = lambda a: f" Each box has {a} pens."
    S_B = lambda b: f" There are {b} boxes."
    REST = lambda c: f" {c} pens are lost. How many pens are left?"
    ANS = lambda P, c: P - c
    S_P = lambda P: f" There are {P} pens."
    order = "AB"
elif TPL == "rows":
    S_A = lambda a: f" Each row has {a} trees."
    S_B = lambda b: f" A farmer has {b} rows of trees."
    REST = lambda c: f" He plants {c} more trees. How many trees does he have now?"
    ANS = lambda P, c: P + c
    S_P = lambda P: f" A farmer has {P} trees."
    order = "BA"


def body(a, b, c, mark=None, P=None):
    """mark in {None,'sentA','sentB','both','numA'}; if P is given build the ORACLE for that location"""
    sa, sb = S_A(a), S_B(b)
    if P is None:  # task
        if mark == "sentA": sa = f"[[{sa}]]"
        if mark == "sentB": sb = f"[[{sb}]]"
        if mark == "numA": sa = sa.replace(f" {a}", f"[[ {a}]]", 1)
        first, second = (sa, sb) if order == "AB" else (sb, sa)
        if mark == "both":
            return f"[[{first}{second}]]" + REST(c)
        return first + second + REST(c)
    # oracle: the location's sentence(s) replaced by the P sentence
    sp = S_P(P)
    if mark == "sentA":
        sa = f"[[{sp}]]"; first, second = (sa, sb) if order == "AB" else (sb, sa)
        return first + second + REST(c)
    if mark == "numA":
        sa = sp.replace(f" {P}", f"[[ {P}]]", 1); first, second = (sa, sb) if order == "AB" else (sb, sa)
        return first + second + REST(c)
    if mark == "sentB":
        sb = f"[[{sp}]]"; first, second = (sa, sb) if order == "AB" else (sb, sa)
        return first + second + REST(c)
    if mark == "both":
        return f"[[{sp}]]" + REST(c)


rs = random.Random(1234)
shots = [sample(rs) for _ in range(4)]
qa = lambda s: (unmark(body(*s).strip())[0], f" {ANS(s[0]*s[1], s[2])}")
prefix = shots_prefix([qa(s) for s in shots])
prefix_tool = "".join(f"Q: {qa(s)[0]}\nTool: {s[0]}*{s[1]} = {s[0]*s[1]}\nA:{qa(s)[1]}\n\n" for s in shots)
Q = lambda txt: prefix + "Q:" + txt + "\nA:"
test = [sample(rng) for _ in range(NTE)]
TP = [a * b for a, b, c in test]
CP = []
for v in TP:
    w = rng.choice(pool)
    while w == v:
        w = rng.choice(pool)
    CP.append(w)
cands = [[f" {ANS(TP[i], c)}", f" {ANS(CP[i], c)}"] for i, (a, b, c) in enumerate(test)]
plain = lambda ps: [unmark(p)[0] for p in ps]
task = [Q(body(*it)) for it in test]
log("example_task", unmark(task[0])[0][len(prefix):])
log("none", rates(tf_score(plain(task), cands)))
log("text_tool_line", rates(tf_score([prefix_tool + f"Q: {qa(it)[0]}\nTool: {it[0]}*{it[1]} = {it[0]*it[1]}\nA:" for it in test], cands)))
for loc in LOCS:
    orc_t = [Q(body(*it, mark=loc, P=TP[i])) for i, it in enumerate(test)]
    orc_c = [Q(body(*it, mark=loc, P=CP[i])) for i, it in enumerate(test)]
    log(f"{loc}:example_oracle", unmark(orc_t[0])[0][len(prefix):])
    log(f"{loc}:text_oracle(true,follow)", rates(tf_score(plain(orc_t), cands)))
    log(f"{loc}:text_oracle_corr(true,follow)", rates(tf_score(plain(orc_c), cands)))
    tm = [Q(body(*it, mark=loc)) for it in test]
    fit = []
    for _ in range(NFIT):
        it = sample(rng); v = rng.randint(100, 999)
        fit.append((Q(body(*it, mark=loc, P=v)), v))
    LL = LAYS + ([L for L in (18, 20, 22) if L not in LAYS] if loc == "both" and "LOCS" not in os.environ else [])
    sw = SpanWrite(tm[0], orc_t[0], [0], lambda v: f_num(v, 3), fit, LL)
    assert all(sw.ok(tm[i], orc_t[i]) for i in range(NTE))
    log(f"{loc}:pairs", sw.pairs)
    log(f"{loc}:r2", sw.r2)
    for L in LL:
        los = {0: L}
        log(f"{loc}:L{L}:synth:true", rates(tf_score(plain(tm), cands, edits=sw.synth(TP, los))))
        log(f"{loc}:L{L}:synth:corr", rates(tf_score(plain(tm), cands, edits=sw.synth(CP, los))))
        log(f"{loc}:L{L}:mean", rates(tf_score(plain(tm), cands, edits=sw.mean(NTE, los))))
        if L in (4, 12):
            log(f"{loc}:L{L}:donor:true", rates(tf_score(plain(tm), cands, edits=sw.donor(orc_t, los))))
print("DONE", flush=True)
