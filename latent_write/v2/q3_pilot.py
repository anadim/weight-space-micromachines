"""Q3 pilot: text baselines for candidate value-type task formats (no writes). n=200 each."""
import random, json
from lw2 import *

load()
rng = random.Random(5)
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WORDS = [" red", " blue", " green", " cat", " dog", " sun", " tree", " fish"]
res = {}


def run(name, mk, n=200):
    rs = random.Random(1234)
    shots = [mk(rs) for _ in range(4)]
    prefix = "".join(f"Q: {q}\nA:{a}\n\n" for q, a, _ in shots)
    items = [mk(rng) for _ in range(n)]
    r = rates(tf_score([prefix + f"Q: {q}\nA:" for q, _, _ in items], [[a] + alts for _, a, alts in items]))
    res[name] = r
    print(name, r, "| e.g.", items[0][0], "->", items[0][1], flush=True)


def wk(N0, N1, oracle=False):
    def mk(r):
        s = r.randrange(7)
        while True:
            N = r.randint(N0, N1)
            if N % 7: break
        k = N % 7
        nn = k if oracle else N
        return (f"Today is {DAYS[s]}. What day of the week will it be {nn} days from today?",
                " " + DAYS[(s + N) % 7], [" " + DAYS[(s + j) % 7] for j in range(1, 7) if j != k])
    return mk


run("wk_task_N10-69", wk(10, 69))
run("wk_oracle_k", wk(10, 69, oracle=True))


def boolexpr(r):
    v = [r.choice([True, False]) for _ in range(4)]
    o = [r.choice(["and", "or"]) for _ in range(3)]
    e = f"({v[0]} {o[0]} {v[1]}) {o[1]} ({v[2]} {o[2]} {v[3]})"
    return e, bool(eval(e))


def bt(kind):
    def mk(r):
        e, val = boolexpr(r)
        if kind == "not_task":
            q, a = f"What is not ({e})?", not val
        elif kind == "not_oracle":
            q, a = f"What is not ({val})?", not val
        elif kind == "plain_task":
            q, a = f"What is {e}?", val
        return q, f" {a}", [f" {not a}"]
    return mk


for k in ["plain_task", "not_task", "not_oracle"]:
    run("bool_" + k, bt(k))


def cmp(kind):
    def mk(r):
        while True:
            a, b, c = r.randint(11, 29), r.randint(3, 9), r.randint(100, 260)
            if a * b != c: break
        w1, w2 = r.sample(WORDS, 2)
        t = a * b > c
        if kind == "task":
            cond = f"{a}*{b} is greater than {c}"
        elif kind == "easy":
            x, y = r.sample(range(1, 10), 2)
            if (x > y) != t: x, y = y, x
            cond = f"{x} is greater than {y}"
        elif kind == "literal":
            cond = "true" if t else "false"
            return f"Condition: {cond}. If the condition is true, say{w1}; otherwise say{w2}.", w1 if t else w2, [w2 if t else w1]
        elif kind == "task_c":
            return f"Condition: {a}*{b} > {c}. If the condition is true, say{w1}; otherwise say{w2}.", w1 if t else w2, [w2 if t else w1]
        return f"If {cond}, say{w1}; otherwise say{w2}.", w1 if t else w2, [w2 if t else w1]
    return mk


for k in ["task", "easy", "literal", "task_c"]:
    run("cmp_" + k, cmp(k))


def cmpnum(kind):  # comparison feeding arithmetic: add 10 to the larger
    def mk(r):
        while True:
            a, b, c = r.randint(11, 29), r.randint(3, 9), r.randint(100, 260)
            if a * b != c: break
        t = a * b > c
        if kind == "task":
            return f"Is {a}*{b} greater than {c}? Answer yes or no.", " yes" if t else " no", [" no" if t else " yes"]
        return f"Is {a*b} greater than {c}? Answer yes or no.", " yes" if t else " no", [" no" if t else " yes"]
    return mk


run("yn_task", cmpnum("task"))
run("yn_oracleP", cmpnum("oracle"))
json.dump(res, open("results/q3_pilot.json", "w"), indent=1)
