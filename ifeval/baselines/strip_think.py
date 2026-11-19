import json,sys,glob
# stripped answer = text after </think>; if no </think> (truncated mid-think), no answer was produced -> empty.
for pref in sys.argv[1:]:
    for f in (sorted(glob.glob(pref+".s*.jsonl")) or [pref]):
        out=f.replace(".jsonl",".ans.jsonl")
        with open(out,"w") as w:
            for l in open(f):
                r=json.loads(l); resp=r["response"]
                r["response"]=(resp.split("</think>",1)[1].lstrip("\n") if "</think>" in resp else "")
                w.write(json.dumps(r)+"\n")
        print("wrote",out)
