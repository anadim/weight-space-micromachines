import json, glob, sys, random
from lm_eval.tasks.ifeval import utils as U
import langdetect; langdetect.DetectorFactory.seed = 0
docs = {json.loads(l)["key"]: json.loads(l) for l in open("data/ifeval.jsonl")}
def load(p): return {json.loads(l)["key"]: json.loads(l) for f in glob.glob(p + ".s*.jsonl") for l in open(f)}
tag = sys.argv[1]; P, I = load(f"runs/{tag}_ifeval_plain"), load(f"runs/{tag}_ifeval_imp")
def sc(k, r):
    random.seed(k); d = docs[k]; return U.process_results({"key": k, "instruction_id_list": d["instruction_id_list"], "prompt": d["prompt"], "kwargs": d["kwargs"]}, [r])
for k in sorted(P):
    a, b = sc(k, P[k]["response"]), sc(k, I[k]["response"])
    if a["prompt_level_strict_acc"] and not b["prompt_level_strict_acc"]:
        c = I[k]["circ"]; lat = [x for x in ["nc","lower","caps","quote","end","ps","forbid","repeat","two","para","nth","title","high","bul"] if c.get(x)]
        failed = [i for i, s in zip(docs[k]["instruction_id_list"], b["inst_level_strict_acc"]) if not s]
        print(k, "ODD" if k % 2 else "EVEN", "latched", lat, "fires", c["fires"], "failed", failed, "finish", P[k]["finish"], "->", I[k]["finish"], "ntok", P[k]["ntok"], "->", I[k]["ntok"])
