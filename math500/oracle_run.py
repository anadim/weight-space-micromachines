"""Oracle injection: cut the model's own solution where a perfect tool would fire, insert the exact tool output,
continue greedily with the frozen model, repeat (up to R rounds), grade. Measurement only."""
import json, sys, time, argparse, torch, re
from transformers import AutoTokenizer, AutoModelForCausalLM
from oracle_lib import fire_point, loop_point, LOOP_TEXT

CHECK_BAD = "\n\nLet me verify this answer by checking it against the conditions of the problem. It does not check out, so the answer above is wrong. Let me redo the problem carefully.\n"
CHECK_OK = "\n\nLet me verify this answer by checking it against the conditions of the problem. It checks out.\n"
RETHINK = "\n\nHmm, wait. Let me rethink this.\n"


def end_point(text):
    """Where an end-of-solution implant fires: just before 'Final Answer:' if present, else after the last \\boxed{...}."""
    k = text.rfind("Final Answer:")
    if k >= 0:
        return k
    m = [x for x in re.finditer(r"\\boxed\s*\{", text)]
    if not m:
        return None
    i = m[-1].end(); d = 1
    while i < len(text) and d:
        d += {"{": 1, "}": -1}.get(text[i], 0); i += 1
    j = text.find("\n", i)
    return len(text) if j < 0 else j


def fire_any(text, kind, start, gold):
    if kind in ("loop", "loopcommit"):
        c = loop_point(text, start)
        return None if c is None else (c, LOOP_TEXT[kind], dict(type=kind, fix="loop-break"))
    if kind in ("check", "rethink"):
        c = end_point(text)
        if c is None or c < start:
            return None
        if kind == "rethink":
            return c, RETHINK, dict(type="rethink", fix="rethink")
        pred, _ = extract(text)
        if is_correct(pred, gold):
            return c, CHECK_OK, dict(type="check_ok", fix="ok")
        return c, CHECK_BAD, dict(type="check_bad", fix="bad")
    if kind == "all":  # loop breaker + solver + CAS (sym+num), whichever fires first
        cands = []
        c = loop_point(text, start)
        if c is not None:
            cands.append((c, LOOP_TEXT["loop"], dict(type="loop", fix="loop-break")))
        for k in ("solve", "cas"):
            fp = fire_point(text, k, start)
            if fp:
                cands.append(fp)
        return min(cands, key=lambda t: t[0]) if cands else None
    return fire_point(text, kind, start)
from grade import extract, is_correct
from gen import SHOTS  # noqa  (gen.py guarded below)

ap = argparse.ArgumentParser()
ap.add_argument("--model"); ap.add_argument("--graded"); ap.add_argument("--kind"); ap.add_argument("--out")
ap.add_argument("--rounds", type=int, default=4); ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--noop", action="store_true", help="control: cut at the same point, insert nothing, 1 round")
a = ap.parse_args()
data = json.load(open(a.graded))
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()


def head(x):
    return SHOTS + "Problem:\n" + x["problem"] + "\n\nSolution:\n"


def generate(items):
    """items: list of (key, prompt, max_new). returns dict key -> continuation"""
    res = {}
    items = sorted(items, key=lambda t: len(t[1]))
    for b in range(0, len(items), a.bs):
        chunk = items[b:b + a.bs]
        enc = tok([c[1] for c in chunk], return_tensors="pt", padding=True).to("cuda")
        mx = max(c[2] for c in chunk)
        with torch.no_grad():
            g = m.generate(**enc, max_new_tokens=mx, do_sample=False, temperature=None, top_p=None, top_k=None,
                           stop_strings=["\nProblem:"], tokenizer=tok, pad_token_id=tok.pad_token_id)
        for j, c in enumerate(chunk):
            new = g[j, enc.input_ids.shape[1]:][:c[2]]
            res[c[0]] = tok.decode(new, skip_special_tokens=True).split("\nProblem:")[0]
    return res


state = {x["i"]: dict(text=x["solution"], start=0, fires=[], active=True) for x in data}
byi = {x["i"]: x for x in data}
t0 = time.time()
for r in range(1 if a.noop else a.rounds):
    todo = []
    for i, st in state.items():
        if not st["active"]:
            continue
        try:
            fp = fire_any(st["text"], a.kind, st["start"], byi[i]["gold"])
        except Exception:
            fp = None
        if fp is None or (a.kind == "check" and fp[2]["type"] == "check_ok" and st["fires"]):
            st["active"] = False; continue
        cut, ins, info = fp
        if a.noop:
            ins = ""
        prefix = st["text"][:cut] + ins
        info.update(round=r, cut=cut, orig_after=st["text"][cut:cut + 80])
        st["fires"].append(info); st["start"] = len(prefix); st["prefix"] = prefix
        ntok = len(tok(prefix).input_ids)
        todo.append((i, head(byi[i]) + prefix, max(384, 1024 - ntok)))
    print(f"round {r}: {len(todo)} fires  {time.time()-t0:.0f}s", flush=True)
    if not todo:
        break
    cont = generate(todo)
    for i, c in cont.items():
        state[i]["text"] = state[i]["prefix"] + c
out = []
for x in data:
    st = state[x["i"]]
    pred, how = extract(st["text"])
    ok = is_correct(pred, x["gold"])
    out.append(dict(i=x["i"], subject=x["subject"], level=x["level"], gold=x["gold"], orig_correct=x["correct"],
                    orig_pred=x["pred"], fired=len(st["fires"]) > 0, fires=st["fires"], new_pred=pred, new_correct=ok,
                    text=st["text"] if st["fires"] else None))
json.dump(out, open(a.out, "w"), indent=1)
f = [o for o in out if o["fired"]]
print(f"KIND {a.kind}{' NOOP' if a.noop else ''}: fired on {len(f)} ({sum(not o['orig_correct'] for o in f)} wrong, "
      f"{sum(o['orig_correct'] for o in f)} right); fixed {sum(o['new_correct'] and not o['orig_correct'] for o in f)}, "
      f"broke {sum(o['orig_correct'] and not o['new_correct'] for o in f)}; "
      f"acc {sum(o['orig_correct'] for o in out)} -> {sum(o['new_correct'] for o in out)} / {len(out)}")
