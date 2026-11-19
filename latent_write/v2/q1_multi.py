"""Q1: several latent values.
 sum2 : What is (a*b)+(c*d)?        two independent intermediates, each written at its own span
 sum3 : What is (a*b)+(c*d)+(e*f)?  three
 chain: What is ((a*b)+c)*d?        nested: P1=a*b (inner span) and P2=P1+c (outer span)
Usage: python q1_multi.py <task> [layers]
"""
import json, random, sys, time, torch
from lw2 import *

TASK = sys.argv[1]
LAYS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,4,8,10,12,14,16,18").split(",")]
NTE, NFIT = 300, 2000
load()
rng = random.Random(7)
res = {"task": TASK, "model": MODEL}
OUT = f"results/q1_{TASK}_{MODEL.split('/')[-1]}.json"
t0 = time.time()


def log(k, v):
    res[k] = v
    print(f"[{time.time()-t0:6.0f}s] {k}: {v}", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)


def prod_item(r):
    while True:
        a, b = r.randint(11, 29), r.randint(3, 9)
        if a * b >= 100:
            return a, b


pool = [a * b for a in range(11, 30) for b in range(3, 10) if a * b >= 100]


def corrupt(v, r):
    w = r.choice(pool)
    while w == v:
        w = r.choice(pool)
    return w


if TASK in ("sum2", "sum3"):
    K = 2 if TASK == "sum2" else 3
    def sample(r):
        return [prod_item(r) for _ in range(K)]
    def q_task(it, mark=None):  # mark: set of span ids to mark
        parts = []
        for k, (a, b) in enumerate(it):
            s = f"{a}*{b}"
            parts.append(f"([[{s}]])" if mark is None or k in mark else f"({s})")
        return "What is " + "+".join(parts) + "?"
    def q_orc(it, vals, which):  # spans in `which` replaced by their values (marked)
        parts = []
        for k, (a, b) in enumerate(it):
            parts.append(f"([[{vals[k]}]])" if k in which else f"({a}*{b})")
        return "What is " + "+".join(parts) + "?"
    ans = lambda it, vals: sum(vals)
    rs = random.Random(1234)
    shots = [sample(rs) for _ in range(4)]
    prefix = shots_prefix([(q_task(s).replace("[[", "").replace("]]", ""), f" {sum(a*b for a,b in s)}") for s in shots])
    prefix_tool = "".join(f"Q: {unmark(q_task(s))[0]}\nTool: " + ", ".join(f"{a}*{b} = {a*b}" for a, b in s)
                          + f"\nA: {sum(a*b for a,b in s)}\n\n" for s in shots)
    test = [sample(rng) for _ in range(NTE)]
    tv = [[a * b for a, b in it] for it in test]
    cv = [[corrupt(v, rng) for v in vs] for vs in tv]
    T = lambda it: prefix + f"Q: {q_task(it)}\nA:"
    O = lambda it, vals, which: prefix + f"Q: {q_orc(it, vals, which)}\nA:"
    task_m = [T(it) for it in test]
    # candidate sets: true, each single corruption, all corrupted
    patterns = [[]] + [[k] for k in range(K)] + [list(range(K))]
    vals_of = lambda i, pat: [cv[i][k] if k in pat else tv[i][k] for k in range(K)]
    cands = [[f" {sum(vals_of(i, p))}" for p in patterns] for i in range(NTE)]
    pname = ["true"] + [f"corr{k}" for k in range(K)] + ["corrall"]
    plain = lambda ps: [unmark(p)[0] for p in ps]
    log("none", rates(tf_score(plain(task_m), cands)))
    log("text_oracle_all(true,..)", rates(tf_score(plain([O(test[i], tv[i], range(K)) for i in range(NTE)]), cands)))
    log("text_oracle_all_corrall", rates(tf_score(plain([O(test[i], cv[i], range(K)) for i in range(NTE)]),
                                                  [[cands[i][0], cands[i][-1]] for i in range(NTE)])))
    log("text_tool_line", rates(tf_score([prefix_tool + f"Q: {unmark(q_task(it))[0]}\nTool: " +
                                          ", ".join(f"{a}*{b} = {a*b}" for a, b in it) + "\nA:" for it in test], cands)))
    configs = {"all": list(range(K))}
    for k in range(K):
        configs[f"only{k}"] = [k]
    for cname, which in configs.items():
        fit = []
        for _ in range(NFIT):
            it = sample(rng); vals = [rng.randint(100, 999) for _ in range(K)]
            fit.append((O(it, vals, which), [vals[k] for k in which]))
        feat = lambda v: sum((f_num(x, 3) for x in v), [])
        ref_o = O(test[0], tv[0], which)
        sw = SpanWrite(task_m[0], ref_o, which, feat, fit, LAYS, with_suffix=(cname == "all"))
        sw_ns = SpanWrite(task_m[0], ref_o, which, feat, fit, LAYS) if cname == "all" else sw
        assert all(sw.ok(task_m[i], O(test[i], tv[i], which)) for i in range(NTE))
        log(f"{cname}:r2", sw_ns.r2)
        for L in LAYS:
            los = {k: L for k in range(len(which))}
            r = rates(tf_score(plain(task_m), cands, edits=sw_ns.synth([[tv[i][k] for k in which] for i in range(NTE)], los)))
            log(f"{cname}:L{L}:synth:true", dict(zip(pname, r)))
            for p, pn in zip(patterns[1:], pname[1:]):
                if not set(p) <= set(which):
                    continue
                if cname != "all" and len(p) != 1:
                    continue
                vals = [[vals_of(i, p)[k] for k in which] for i in range(NTE)]
                r = rates(tf_score(plain(task_m), cands, edits=sw_ns.synth(vals, los)))
                log(f"{cname}:L{L}:synth:{pn}", dict(zip(pname, r)))
            if cname == "all":
                r = rates(tf_score(plain(task_m), cands, edits=sw_ns.mean(NTE, los)))
                log(f"{cname}:L{L}:mean", dict(zip(pname, r)))
                if L >= 12:  # slot + suffix refresh (window extension)
                    los2 = dict(los); los2[-1] = L
                    for pn, p in [("true", []), ("corrall", list(range(K)))]:
                        r = rates(tf_score(plain(task_m), cands, edits=sw.synth([vals_of(i, p) for i in range(NTE)], los2)))
                        log(f"{cname}+suffix:L{L}:synth:{pn}", dict(zip(pname, r)))
        if cname == "all":
            # different layers per span (early span written early, later span late, and reverse)
            for lo in ([4, 12], [12, 4], [0, 16], [16, 0]) if K == 2 else ([4, 8, 12], [12, 8, 4]):
                los = {k: lo[k] for k in range(K)}
                if not all(l in LAYS for l in lo):
                    continue
                for pn, p in [("true", []), ("corrall", list(range(K)))]:
                    r = rates(tf_score(plain(task_m), cands, edits=sw_ns.synth([vals_of(i, p) for i in range(NTE)], los)))
                    log(f"{cname}:mixed{lo}:synth:{pn}", dict(zip(pname, r)))
            for L in [4, 12]:
                if L in LAYS:
                    los = {k: L for k in range(K)}
                    r = rates(tf_score(plain(task_m), cands, edits=sw_ns.donor([O(test[i], tv[i], which) for i in range(NTE)], los)))
                    log(f"{cname}:L{L}:donor:true", dict(zip(pname, r)))

elif TASK == "chain":
    def sample(r):
        while True:
            a, b, c, d = r.randint(2, 9), r.randint(2, 9), r.randint(10, 89), r.randint(3, 9)
            if a * b >= 10 and 20 <= a * b + c <= 99:
                return a, b, c, d
    Tin = lambda it: f"What is (([[{it[0]}*{it[1]}]])+{it[2]})*{it[3]}?"
    Tout = lambda it: f"What is ([[({it[0]}*{it[1]})+{it[2]}]])*{it[3]}?"
    Oin = lambda it, v1: f"What is (([[{v1}]])+{it[2]})*{it[3]}?"
    Oout = lambda it, v2: f"What is ([[{v2}]])*{it[3]}?"
    rs = random.Random(1234)
    shots = [sample(rs) for _ in range(4)]
    A = lambda it: (it[0] * it[1] + it[2]) * it[3]
    prefix = shots_prefix([(unmark(Tin(s))[0], f" {A(s)}") for s in shots])
    tl = lambda s: f"{s[0]}*{s[1]} = {s[0]*s[1]}, {s[0]*s[1]}+{s[2]} = {s[0]*s[1]+s[2]}"
    tl1 = lambda s: f"{s[0]}*{s[1]} = {s[0]*s[1]}"
    prefix_tool = "".join(f"Q: {unmark(Tin(s))[0]}\nTool: {tl(s)}\nA: {A(s)}\n\n" for s in shots)
    prefix_tool1 = "".join(f"Q: {unmark(Tin(s))[0]}\nTool: {tl1(s)}\nA: {A(s)}\n\n" for s in shots)
    test = [sample(rng) for _ in range(NTE)]
    P1 = [a * b for a, b, c, d in test]; P2 = [p + it[2] for p, it in zip(P1, test)]
    p1pool = sorted({a * b for a in range(2, 10) for b in range(2, 10) if a * b >= 10})
    C1, C2 = [], []
    for i, it in enumerate(test):
        while True:
            v = rng.choice(p1pool)
            if v != P1[i] and 20 <= v + it[2] <= 99:
                break
        C1.append(v)
        while True:
            w = rng.randint(20, 99)
            if w != P2[i]:
                break
        C2.append(w)
    cands = [[f" {P2[i]*it[3]}", f" {(C1[i]+it[2])*it[3]}", f" {C2[i]*it[3]}"] for i, it in enumerate(test)]
    pname = ["true", "follow_inner_corr", "follow_outer_corr"]
    pfx = lambda s: prefix + f"Q: {s}\nA:"
    plain = lambda ps: [unmark(p)[0] for p in ps]
    task_in = [pfx(Tin(it)) for it in test]; task_out = [pfx(Tout(it)) for it in test]
    log("none", dict(zip(pname, rates(tf_score(plain(task_in), cands)))))
    log("text_oracle_P1", dict(zip(pname, rates(tf_score(plain([pfx(Oin(it, P1[i])) for i, it in enumerate(test)]), cands)))))
    log("text_oracle_P1corr", dict(zip(pname, rates(tf_score(plain([pfx(Oin(it, C1[i])) for i, it in enumerate(test)]), cands)))))
    log("text_oracle_P2", dict(zip(pname, rates(tf_score(plain([pfx(Oout(it, P2[i])) for i, it in enumerate(test)]), cands)))))
    log("text_oracle_P2corr", dict(zip(pname, rates(tf_score(plain([pfx(Oout(it, C2[i])) for i, it in enumerate(test)]), cands)))))
    log("text_tool_both", dict(zip(pname, rates(tf_score([prefix_tool + f"Q: {unmark(Tin(it))[0]}\nTool: {tl(it)}\nA:" for it in test], cands)))))
    log("text_tool_P1", dict(zip(pname, rates(tf_score([prefix_tool1 + f"Q: {unmark(Tin(it))[0]}\nTool: {tl1(it)}\nA:" for it in test], cands)))))
    fin = [(pfx(Oin(it, v)), [v]) for it, v in ((sample(rng), rng.randint(10, 99)) for _ in range(NFIT))]
    fout = [(pfx(Oout(it, v)), [v]) for it, v in ((sample(rng), rng.randint(20, 99)) for _ in range(NFIT))]
    feat = lambda v: f_num(v[0], 2)
    sw_in = SpanWrite(task_in[0], pfx(Oin(test[0], P1[0])), [0], feat, fin, LAYS)
    sw_out = SpanWrite(task_out[0], pfx(Oout(test[0], P2[0])), [0], feat, fout, LAYS)
    fouts = [(om, [v[0], int(unmark(om)[0].split(")*")[-1].split("?")[0])]) for om, v in fout]
    sw_outs = SpanWrite(task_out[0], pfx(Oout(test[0], P2[0])), [0], lambda v: f_num(v[0], 2) + f_num(v[1], 1), fouts,
                        [L for L in LAYS if L >= 12], with_suffix=True)
    assert all(sw_in.ok(task_in[i], pfx(Oin(it, P1[i]))) for i, it in enumerate(test))
    assert all(sw_out.ok(task_out[i], pfx(Oout(it, P2[i]))) for i, it in enumerate(test))
    log("r2_in", sw_in.r2); log("r2_out", sw_out.r2)
    for L in LAYS:
        los = {0: L}
        for nm, sw, tv_, cv_ in [("inner_P1", sw_in, P1, C1), ("outer_P2", sw_out, P2, C2)]:
            log(f"{nm}:L{L}:synth:true", dict(zip(pname, rates(tf_score(plain(task_in), cands, edits=sw.synth([[v] for v in tv_], los))))))
            log(f"{nm}:L{L}:synth:corr", dict(zip(pname, rates(tf_score(plain(task_in), cands, edits=sw.synth([[v] for v in cv_], los))))))
            log(f"{nm}:L{L}:mean", dict(zip(pname, rates(tf_score(plain(task_in), cands, edits=sw.mean(NTE, los))))))
        # both: P1 at the inner span AND P2 at the outer span (outer overwrites the inner positions at its layer)
        # -> the 2-value version: P1 written early at the inner span, P2 written at the outer span's tail tokens only
        if L >= 12:
            los2 = {0: L, -1: L}
            for vn, vv in [("true", P2), ("corr", C2)]:
                log(f"outer_P2+suffix:L{L}:synth:{vn}", dict(zip(pname, rates(tf_score(plain(task_in), cands, edits=sw_outs.synth([[v, it[3]] for v, it in zip(vv, test)], los2))))))
json.dump(res, open(OUT, "w"), indent=1)
print("DONE", flush=True)
