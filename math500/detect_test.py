import json, sys, collections
from oracle_lib import fire_point
d = json.load(open(sys.argv[1]))
kinds = ["solve", "sym", "num"]
cnt = collections.Counter(); ex = collections.defaultdict(list)
for x in d:
    for k in kinds:
        try:
            fp = fire_point(x["solution"], k)
        except Exception as e:
            fp = None
        if fp:
            cnt[(k, x["correct"])] += 1
            ex[k].append((x["i"], x["correct"], fp[2]))
print(cnt)
for k in kinds:
    print("=====", k)
    for e in ex[k][:int(sys.argv[2])]:
        print(e)
