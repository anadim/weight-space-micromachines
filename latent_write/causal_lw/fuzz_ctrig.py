"""fuzz: compiled trigger (ctrig.Trig table scan) == independent Python reference finder (ctrig.ref_find), exact match on
positions, slots and P, over random expression streams (prompt-style and generated-style, with near-miss decoys) and
random token soup. usage: python3.11 fuzz_ctrig.py [N]  (sandbox: tokenizers only)"""
import sys, random, os, glob
from tokenizers import Tokenizer
import ctrig as CT

TJ = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/*/tokenizer.json"))[0]
tk = Tokenizer.from_file(TJ)
V = tk.get_vocab_size()
txt = [tk.decode([i]) for i in range(V)]
enc = lambda s: tk.encode(s).ids


def expr(r):
    k = r.random()
    a = r.choice([r.randint(11, 29), r.randint(1, 9), r.randint(100, 999), r.randint(1000, 9999)]) if k < 0.3 else r.randint(11, 29)
    b = r.choice([r.randint(3, 9), r.randint(10, 99), r.randint(100, 300), 0]) if k < 0.5 else r.randint(3, 9)
    c = r.choice([r.randint(10, 99), r.randint(1, 9), r.randint(100, 9999)]) if k < 0.3 else r.randint(10, 99)
    op = r.choice(["+", "+", "+", "-", "*"])
    s = f"({a}*{b}){op}{c}"
    if r.random() < 0.1: s = s.replace("*", r.choice([" * ", "x", "*("]))
    if r.random() < 0.05: s = "(" + s
    return s, a * b + c


def stream(r):
    parts = []
    for _ in range(r.randint(1, 6)):
        e, v = expr(r)
        f = r.random()
        if f < 0.35: parts.append(f"Q: What is {e}?\nA: {v}\n\n")
        elif f < 0.6: parts.append(f"Q: What is {e}?\nA: {e} = {v}\n\n")
        elif f < 0.75: parts.append(f"Q: What is {e}?\nA:")
        elif f < 0.85: parts.append(f"A: {e} = {v}")
        elif f < 0.92: parts.append(f"The cost is {e} dollars, so{r.choice([':', ' :', ''])} {e}?\nA: ")
        else: parts.append(r.choice(["\n\n", ":", "A:", "?\nA: ", " (", "(", ")+", "12*", "*7)+", "Q: What is ("]))
    return "".join(parts)


def main(N):
    r = random.Random(0); n = fires = 0
    cfgs = [dict(colon=False, tail=True, nsep=0), dict(colon=False, tail=True, nsep=4), dict(colon=True, tail=False, nsep=0),
            dict(colon=True, tail=False, nsep=4), dict(colon=False, tail=False, nsep=0)]
    trigs = [CT.Trig(txt, **c) for c in cfgs]
    for it in range(N):
        if it % 5 == 4:   # random token soup over the pattern alphabet + random vocab
            alpha = enc(" (") + enc("(") + [enc(str(d))[0] for d in range(10)] + enc("*") + enc(")+") + enc("?\n") + enc("A") + enc(":") + enc(" ") + enc("\n\n")
            ids = [r.choice(alpha) if r.random() < 0.85 else r.randrange(V) for _ in range(r.randint(1, 80))]
        else:
            ids = enc(stream(r))
        toks = [txt[i] for i in ids]
        for c, t in zip(cfgs, trigs):
            a = CT.dfa_find(t, ids); b = CT.ref_find(toks, **c)
            if a != b:
                print("MISMATCH", c, repr("".join(toks)), a, b); sys.exit(1)
            fires += len(a); n += 1
    print(f"fuzz OK: {N} streams x {len(cfgs)} gate configs = {n} scans, {fires} fires, exact match (pos, slot, P)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000)
