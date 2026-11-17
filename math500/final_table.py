"""Build the oracle-injection results table (raw and noise-adjusted with the NOOP controls).
python math500/final_table.py [RUN_DIR]   (default: math500/results/runs, the reduced files)"""
import json, os, sys
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "runs")
KINDS = [("num", "calc: LaTeX exact calculator", "num"), ("solve", "equation/system solver", "solve"),
         ("sym", "algebra: expand/factor/simplify", "sym"), ("all", "combined: loop+solver+calc+algebra", "all"),
         ("loop", "loop breaker -> different approach", None), ("loopcommit", "loop breaker -> commit to answer", None),
         ("check", "ORACLE verifier (uses gold) + redo", "rethink"), ("rethink", "unconditional rethink (no oracle)", "rethink")]


def load(m, k):
    f = f"{R}/or_{m}_{k}.json"
    if not os.path.exists(f): return None
    d = json.load(open(f))
    return d["rows"] if isinstance(d, dict) else d      # reduced result file (results/runs) or a full oracle_run.py output


def sets(d):
    fx = {o["i"] for o in d if o["fired"] and o["new_correct"] and not o["orig_correct"]}
    br = {o["i"] for o in d if o["fired"] and o["orig_correct"] and not o["new_correct"]}
    return fx, br


out = []
out.append("| implant (oracle) | model | fires on fail/right | fixed | broke | acc | noise-adj fixed | noise-adj broke | net adj | fixed ids |")
out.append("|---|---|---|---|---|---|---|---|---|---|")
for k, name, nk in KINDS:
    for m in ["0.6b", "1.7b"]:
        d = load(m, k)
        if d is None:
            out.append(f"| {name} | {m} | pending | | | | | | | |"); continue
        fx, br = sets(d)
        base = sum(o["orig_correct"] for o in d); new = sum(o["new_correct"] for o in d)
        fw = sum(o["fired"] and not o["orig_correct"] for o in d); fr = sum(o["fired"] and o["orig_correct"] for o in d)
        nd = load(m, nk + "_noop") if nk else None
        if nd is not None:
            nfx, nbr = sets(nd); afx = fx - nfx; abr = br - nbr; adj = f"{len(afx)} | {len(abr)} | {len(afx) - len(abr):+d}"
        else:
            afx = fx; adj = "n/a | n/a | n/a"
        out.append(f"| {name} | {m} | {fw}/{fr} | {len(fx)} | {len(br)} | {base}->{new} ({100*base/500:.1f}->{100*new/500:.1f}%) | {adj} | "
                   f"{sorted(afx)}{' broke ' + str(sorted(br)) if br else ''} |")
print("\n".join(out))
