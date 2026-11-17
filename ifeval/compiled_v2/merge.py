import json,glob,sys
for p in sys.argv[1:]:
    rows={}
    for f in sorted(glob.glob(f"runs/{p}_s*.jsonl")):
        for l in open(f): r=json.loads(l); rows[r["key"]]=r
    open(f"runs/{p}.jsonl","w").write("".join(json.dumps(rows[k])+"\n" for k in sorted(rows)))
    print(p,len(rows))
