"""Per-run summary of oracle_run.py outputs. Run inside a directory of or_*.json files (e.g. math500/results/runs)."""
import json, sys, glob, os
rows = []
for f in sorted(glob.glob("or_*.json")):
    tag = os.path.basename(f)[3:-5]
    d = json.load(open(f)); d = d["rows"] if isinstance(d, dict) else d
    fired = [o for o in d if o["fired"]]
    fixed = [o["i"] for o in fired if o["new_correct"] and not o["orig_correct"]]
    broke = [o["i"] for o in fired if o["orig_correct"] and not o["new_correct"]]
    base = sum(o["orig_correct"] for o in d); new = sum(o["new_correct"] for o in d)
    rows.append((tag, len(fired), sum(not o["orig_correct"] for o in fired), sum(o["orig_correct"] for o in fired),
                 len(fixed), len(broke), base, new, fixed, broke))
print(f"{'run':22s} fired(wrong/right) fixed broke  acc")
for r in rows:
    print(f"{r[0]:22s} {r[1]:4d} ({r[2]:3d}/{r[3]:3d})  {r[4]:4d} {r[5]:4d}  {r[6]}->{r[7]}  ({100*r[6]/500:.1f}->{100*r[7]/500:.1f}%)")
    print(f"    fixed={r[8]}\n    broke={r[9]}")
