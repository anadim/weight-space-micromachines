import json, glob, random
from lm_eval.tasks.ifeval import utils as U
import langdetect; langdetect.DetectorFactory.seed = 0
docs = {json.loads(l)["key"]: json.loads(l) for l in open("data/ifeval.jsonl")}
def load(p): return {json.loads(l)["key"]: json.loads(l) for f in glob.glob(p + ".s*.jsonl") for l in open(f)}
var = {}
for tag in ["q06", "q17", "q4"]:
    for m in ["plain", "imp"]:
        R = load(f"runs/{tag}_ifeval_{m}"); ks = sorted(R)
        res = []
        for rep in range(3):
            random.seed(rep); order = ks[:]; random.shuffle(order); out = {}
            for k in order:
                d = docs[k]; o = U.process_results({"key": k, "instruction_id_list": d["instruction_id_list"], "prompt": d["prompt"], "kwargs": d["kwargs"]}, [R[k]["response"]])
                out[k] = tuple(o["inst_level_strict_acc"])
            res.append(out)
        bad = [k for k in ks if len({r[k] for r in res}) > 1]
        for k in bad: var.setdefault(k, set()).add(f"{tag}_{m}")
        print(tag, m, "keys with order-dependent strict scores:", [(k, [i for i, x in enumerate(zip(*[r[k] for r in res])) if len(set(x)) > 1], [docs[k]["instruction_id_list"][i] for i, x in enumerate(zip(*[r[k] for r in res])) if len(set(x)) > 1]) for k in bad], flush=True)
