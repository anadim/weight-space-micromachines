import json, random, sys, collections
from autopass_lib import squeeze
d = json.load(open(sys.argv[1])); n = int(sys.argv[2]); out = sys.argv[3]
fails = [x for x in d if not x["correct"]]
cells = collections.defaultdict(list)
for x in fails: cells[(x["subject"], x["level"])].append(x)
random.seed(1)
for v in cells.values(): random.shuffle(v)
frac = n / len(fails); pick = []
for k, v in sorted(cells.items()):
    pick += v[:max(1, round(len(v) * frac))]
pick = sorted(pick, key=lambda x: x["i"])[:n] if len(pick) >= n else pick
print(len(pick))
with open(out, "w") as f:
    for x in pick:
        f.write(f"\n######## id={x['i']} | {x['subject']} L{x['level']} | gold={x['gold']} | pred={x['pred']} | ntok={x['ntok']}\nPROBLEM: {x['problem']}\n---SOLUTION---\n{squeeze(x['solution'])}\n")
json.dump([x["i"] for x in pick], open(out + ".ids", "w"))
