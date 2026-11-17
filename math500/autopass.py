"""Automatic first pass over graded failures: loop detection, truncation, no-answer, extraction issues."""
import json, re, sys
from collections import Counter


def loop_score(txt, n=12):
    toks = txt.split()
    if len(toks) < 3 * n:
        return 0.0
    grams = [" ".join(toks[i:i + n]) for i in range(len(toks) - n)]
    c = Counter(grams)
    rep = sum(v - 1 for v in c.values() if v > 1)
    return rep / max(1, len(grams))


def tail_loop(txt):
    """True if the last ~40% of the text is dominated by repeated lines/ngrams."""
    t = txt[int(len(txt) * 0.6):]
    return loop_score(t, 8) > 0.5


data = json.load(open(sys.argv[1]))
for d in data:
    s = d["solution"]
    d["auto"] = dict(loop=round(loop_score(s), 3), tail_loop=tail_loop(s), hit_limit=d.get("hit_limit", False),
                     no_answer=d["pred"] is None)
json.dump(data, open(sys.argv[1], "w"), indent=1)
fails = [d for d in data if not d["correct"]]
print("fails", len(fails))
print("hit_limit", sum(d["auto"]["hit_limit"] for d in fails), "tail_loop", sum(d["auto"]["tail_loop"] for d in fails),
      "no_answer", sum(d["auto"]["no_answer"] for d in fails),
      "limit&loop", sum(d["auto"]["hit_limit"] and d["auto"]["tail_loop"] for d in fails))
