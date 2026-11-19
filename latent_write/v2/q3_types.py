"""Q3: non-numeric / non-product latent values, written at the span they stand for.
 bool : What is not ({E})?  E = (x o y) o (z o w) over True/False.  value = truth of E (2 classes), written at E's span
        oracle: What is not (v)?
 wk   : Today is {S}. What day of the week will it be {N} days from today?  N 10-69.  value = N mod 7 (date offset,
        6 classes), written at N's span; oracle: ... {k} days from today?
 cmp  : Condition: {a}*{b} > {c}. If the condition is true, say w1; otherwise say w2.  |a*b-c| <= 25.
        value = comparison result (2 classes), written at 'a*b > c'; oracle 'Condition: true.' (literal)
 cmpE : If {a}*{b} is greater than {c}, say w1; otherwise say w2.  value written from an EASY comparison oracle
        'If x is greater than y' with random 1-digit x,y; the writer sees only the truth bit (x,y averaged out).
Usage: python q3_types.py <task> [layers]
"""
import json, random, sys, time
from lw2 import *

TASK = sys.argv[1]
LAYS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,4,8,12,14,16,18,20").split(",")]
NTE, NFIT = 300, 2000
load()
rng = random.Random(31)
res = {"task": TASK, "model": MODEL}
OUT = f"results/q3_{TASK}_{MODEL.split('/')[-1]}.json"
t0 = time.time()
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WORDS = [" red", " blue", " green", " cat", " dog", " sun", " tree", " fish"]


def log(k, v):
    res[k] = v
    print(f"[{time.time()-t0:6.0f}s] {k}: {v}", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)


# each task defines: sample(r)->item ; T(item) marked task ; O(item, v) marked oracle ; val(item) ; corr(item, v) ;
# ans(item, v) -> answer string implied by value v ; cats (value classes) ; plain-text tool line
if TASK == "bool":
    def sample(r):
        v = [r.choice([True, False]) for _ in range(4)]
        o = [r.choice(["and", "or"]) for _ in range(3)]
        return v, o
    E = lambda it: f"({it[0][0]} {it[1][0]} {it[0][1]}) {it[1][1]} ({it[0][2]} {it[1][2]} {it[0][3]})"
    T = lambda it: f"What is not ([[{E(it)}]])?"
    O = lambda it, v: f"What is not ([[{v}]])?"
    val = lambda it: bool(eval(E(it)))
    cats = [True, False]
    ans = lambda it, v: f" {not v}"
    tool = lambda it: f"Tool: {E(it)} = {val(it)}"
elif TASK == "wk":
    def sample(r):
        while True:
            s, N = r.randrange(7), r.randint(10, 69)
            if N % 7:
                return s, N
    T = lambda it: f"Today is {DAYS[it[0]]}. What day of the week will it be [[{it[1]}]] days from today?"
    O = lambda it, v: f"Today is {DAYS[it[0]]}. What day of the week will it be [[{v}]] days from today?"
    val = lambda it: it[1] % 7
    cats = [1, 2, 3, 4, 5, 6]
    ans = lambda it, v: " " + DAYS[(it[0] + v) % 7]
    tool = lambda it: f"Tool: {it[1]} mod 7 = {it[1] % 7}"
elif TASK in ("cmp", "cmpE"):
    def sample(r):
        while True:
            a, b = r.randint(11, 29), r.randint(3, 9)
            c = a * b + r.choice([d for d in range(-25, 26) if d])
            if 100 <= c <= 999 and a * b >= 100:
                return a, b, c, tuple(r.sample(WORDS, 2))
    val = lambda it: it[0] * it[1] > it[2]
    cats = [True, False]
    ans = lambda it, v: it[3][0] if v else it[3][1]
    tool = lambda it: f"Tool: {it[0]}*{it[1]} = {it[0]*it[1]}"
    if TASK == "cmp":
        T = lambda it: f"Condition:[[ {it[0]}*{it[1]} > {it[2]}]]. If the condition is true, say{it[3][0]}; otherwise say{it[3][1]}."
        O = lambda it, v: f"Condition:[[ {'true' if v else 'false'}]]. If the condition is true, say{it[3][0]}; otherwise say{it[3][1]}."
    else:
        T = lambda it: f"If[[ {it[0]}*{it[1]} is greater than {it[2]}]], say{it[3][0]}; otherwise say{it[3][1]}."
        def O(it, v, r=random.Random(77)):
            x, y = r.sample(range(1, 10), 2)
            if (x > y) != v:
                x, y = y, x
            return f"If[[ {x} is greater than {y}]], say{it[3][0]}; otherwise say{it[3][1]}."

if TASK == "cmpN":  # comparison result written at the comparison span, consumed by a negation
    def sample(r):
        while True:
            a, b = r.randint(11, 29), r.randint(3, 9)
            c = a * b + r.choice([d for d in range(-25, 26) if d])
            if 100 <= c <= 999 and a * b >= 100:
                return a, b, c
    T = lambda it: f"What is not ([[{it[0]}*{it[1]} > {it[2]}]])?"
    O = lambda it, v: f"What is not ([[{v}]])?"
    val = lambda it: it[0] * it[1] > it[2]
    cats = [True, False]
    ans = lambda it, v: f" {not v}"
    tool = lambda it: f"Tool: {it[0]}*{it[1]} = {it[0]*it[1]}"
if TASK == "yn":  # numeric write (the product) feeding a comparison done by the model
    PP = sorted({a * b for a in range(11, 30) for b in range(3, 10) if a * b >= 100})
    def sample(r):
        while True:
            a, b = r.randint(11, 29), r.randint(3, 9)
            c = a * b + r.choice([d for d in range(-25, 26) if d])
            if 105 <= c <= 255 and a * b >= 100:
                return a, b, c
    T = lambda it: f"Is [[{it[0]}*{it[1]}]] greater than {it[2]}? Answer yes or no."
    O = lambda it, v: f"Is [[{v}]] greater than {it[2]}? Answer yes or no."
    val = lambda it: it[0] * it[1]
    ans = lambda it, v: " yes" if v > it[2] else " no"
    tool = lambda it: f"Tool: {it[0]}*{it[1]} = {it[0]*it[1]}"
    cats = None
FEAT = (lambda v: f_num(v, 3)) if TASK == "yn" else (lambda v: f_cat(v, cats))


def corrupt(it, v, r):
    if TASK != "yn":
        return r.choice([c for c in cats if c != v])
    other = sorted([p for p in PP if (p > it[2]) != (v > it[2])], key=lambda p: abs(p - it[2]))
    return r.choice(other[:8])


def randval(r, it):
    return r.randint(100, 999) if TASK == "yn" else r.choice(cats)


rs = random.Random(1234)
shots = []
while len(shots) < 4:  # balanced shots
    it = sample(rs)
    if TASK == "wk":  # few-shot examples with small offsets (1-6 days); test items use N 10-69
        it = (it[0], rs.randint(1, 6))
    if len(shots) % 2 == 0 or ans(it, val(it)) != ans(shots[-1], val(shots[-1])) or TASK == "wk":
        shots.append(it)
prefix = shots_prefix([(unmark(T(s))[0], ans(s, val(s))) for s in shots])
prefix_tool = "".join(f"Q: {unmark(T(s))[0]}\n{tool(s)}\nA:{ans(s, val(s))}\n\n" for s in shots)
Q = lambda s: prefix + f"Q: {s}\nA:"
test = [sample(rng) for _ in range(NTE)]
TV = [val(it) for it in test]
CV = [corrupt(it, TV[i], rng) for i, it in enumerate(test)]
cands = [[ans(it, TV[i]), ans(it, CV[i])] for i, it in enumerate(test)]
plain = lambda ps: [unmark(p)[0] for p in ps]
tm = [Q(T(it)) for it in test]
log("example_task", unmark(tm[0])[0][len(prefix):])
log("example_oracle", unmark(Q(O(test[0], TV[0])))[0][len(prefix):])
log("frac_first_answer", sum(1 for i in range(NTE) if cands[i][0] == ans(test[i], TV[i]) and TV[i] in (True,)) / NTE)
log("none(true,follow)", rates(tf_score(plain(tm), cands)))
log("text_oracle", rates(tf_score(plain([Q(O(it, TV[i])) for i, it in enumerate(test)]), cands)))
log("text_oracle_corr", rates(tf_score(plain([Q(O(it, CV[i])) for i, it in enumerate(test)]), cands)))
log("text_tool_line", rates(tf_score([prefix_tool + f"Q: {unmark(T(it))[0]}\n{tool(it)}\nA:" for it in test], cands)))
feat = FEAT
fit = []
for _ in range(NFIT):
    it = sample(rng); v = randval(rng, it)
    fit.append((Q(O(it, v)), v))
# layouts may vary across items (e.g. True/False token counts): keep the items whose layout matches the writer's
sw = SpanWrite(tm[0], Q(O(test[0], TV[0])), [0], feat, fit, LAYS)
keep = [i for i in range(NTE) if sw.ok(tm[i], Q(O(test[i], TV[i]))) and sw.ok(tm[i], Q(O(test[i], CV[i])))]
log("n_kept", len(keep)); log("nfit", sw.nfit); log("pairs", sw.pairs); log("r2", sw.r2)
K = lambda xs: [xs[i] for i in keep]
tmk, ck = plain(K(tm)), K(cands)
log("none_kept", rates(tf_score(tmk, ck)))
for L in LAYS:
    los = {0: L}
    log(f"L{L}:synth:true", rates(tf_score(tmk, ck, edits=sw.synth(K(TV), los))))
    log(f"L{L}:synth:corr", rates(tf_score(tmk, ck, edits=sw.synth(K(CV), los))))
    log(f"L{L}:mean", rates(tf_score(tmk, ck, edits=sw.mean(len(keep), los))))
    if L in (4, 12):
        log(f"L{L}:donor:true", rates(tf_score(tmk, ck, edits=sw.donor(K([Q(O(it, TV[i])) for i, it in enumerate(test)]), los))))
print("DONE", flush=True)
